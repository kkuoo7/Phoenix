import argparse
import json
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from itertools import islice
import math

# --- 시작: HASS 프로젝트의 핵심 모듈 import ---
# 실제 HASS 프로젝트의 정확한 경로에서 EAModel과 유틸리티 함수들을 가져와야 합니다.
from Phoenix.model.ea_model import EaModel
from Phoenix.model.kv_cache import initialize_past_key_values
from Phoenix.model.utils import initialize_tree
from Phoenix.model.cnets import Model as CNet
# --- 끝: HASS 프로젝트 모듈 import ---

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, IterableDataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# --- 시작: 분석을 위한 핵심 로직 (linguistic_collapse_analysis.py에서 가져옴) ---
# GeometricAnalyzer와 calculate_final_metrics_from_stats는 이전과 동일합니다.
class GeometricAnalyzer:
    """ 메모리 효율적인 LNC 통계량 계산 클래스 """
    def __init__(self, tokenizer=None, min_samples_per_token=2, device='cpu'):
        self.tokenizer = tokenizer
        self.min_samples_per_token = min_samples_per_token
        self.device = device
        self.special_token_ids = set(self.tokenizer.all_special_ids) if self.tokenizer else set()
        self.reset()

    def reset(self):
        self.token_sums = {}
        self.token_counts = {}

    def update_batch(self, features: torch.Tensor, token_ids: torch.Tensor):
        unique_tokens, inverse_indices, counts = torch.unique(token_ids, return_inverse=True, return_counts=True)
        for i, token_id in enumerate(unique_tokens):
            token_id_item = token_id.item()
            if token_id_item in self.special_token_ids: continue
            token_features_sum = features[inverse_indices == i].sum(dim=0)
            if token_id_item not in self.token_counts:
                self.token_counts[token_id_item] = 0
                self.token_sums[token_id_item] = torch.zeros_like(token_features_sum, device=self.device)
            self.token_counts[token_id_item] += counts[i].item()
            self.token_sums[token_id_item] += token_features_sum.to(self.device)

    def _get_class_means(self):
        class_means = {}
        for token_id, count in self.token_counts.items():
            if count >= self.min_samples_per_token:
                class_means[token_id] = self.token_sums[token_id] / count
        return class_means

    def calculate_and_get_stats(self, classifier_weights):
        class_means_map = self._get_class_means()
        gnc2_stats = self._compute_gnc2_stats_from_means(class_means_map)
        unc3_stats = self._compute_unc3_stats_from_means(class_means_map, classifier_weights)
        return gnc2_stats, unc3_stats

    def _compute_gnc2_stats_from_means(self, class_means_map):
        try:
            if len(class_means_map) < 2: return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
            class_means = torch.stack(list(class_means_map.values())).to(self.device)
            global_mean = class_means.mean(dim=0, keepdim=True)
            normalized_means = torch.nn.functional.normalize(class_means - global_mean, p=2, dim=1)
            pdist = torch.pdist(normalized_means.float(), p=2)
            log_inv_distances = torch.log(1.0 / (pdist + 1e-8))
            return {'n': len(log_inv_distances), 'sum': log_inv_distances.sum().item(), 'sum_sq': (log_inv_distances ** 2).sum().item()}
        except Exception as e:
            logger.error(f"GNC2 통계량 계산 실패: {e}")
            return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}

    def _compute_unc3_stats_from_means(self, class_means_map, classifier_weights):
        try:
            if len(class_means_map) < 2: return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
            common_ids = sorted(list(class_means_map.keys() & set(range(classifier_weights.shape[0]))))
            if len(common_ids) < 2: return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
            class_means = torch.stack([class_means_map[cid] for cid in common_ids]).to(self.device)
            weights = classifier_weights[common_ids, :].to(self.device)
            global_mean = class_means.mean(dim=0, keepdim=True)
            normalized_means = torch.nn.functional.normalize(class_means - global_mean, p=2, dim=1)
            normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
            similarities = (normalized_weights * normalized_means).sum(dim=1)
            return {'n': len(similarities), 'sum': similarities.sum().item(), 'sum_sq': (similarities ** 2).sum().item()}
        except Exception as e:
            logger.error(f"UNC3 통계량 계산 실패: {e}")
            return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}

def calculate_final_metrics_from_stats(total_stats):
    n = total_stats['n']
    if n == 0: return {'avg': 0.0, 'cov': 0.0}
    sum_x, sum_x_sq = total_stats['sum'], total_stats['sum_sq']
    mean = sum_x / n
    variance = (sum_x_sq / n) - (mean ** 2)
    if variance < 0: variance = 0
    std = math.sqrt(variance)
    cov = (std / abs(mean)) if mean != 0 else 0.0
    return {'avg': mean, 'cov': cov}

# --- 끝: 분석을 위한 핵심 로직 ---


@torch.inference_mode()
def capture_draft_features(base_model, draft_model, input_ids):
    """
    Base 모델과 Draft 모델을 직접 받아, Draft 토큰과 그 피처를 반환하는 가장 안정적인 함수.
    """
    # 1. Base 모델을 통과시켜 Draft 모델에 입력으로 들어갈 hidden_states를 얻습니다.
    base_outputs = base_model(input_ids, use_cache=False, output_hidden_states=True)
    hidden_states = base_outputs.hidden_states[-1]

    # --- 시작: 이 부분이 핵심 수정 사항입니다 ---

    # 2. topK_genrate가 내부적으로 input_ids[:, 1:]를 수행하므로,
    #    hidden_states도 동일하게 마지막 토큰의 피처를 제외하고 전달하여 길이를 맞춥니다.
    hidden_states_for_draft = hidden_states[:, :-1, :]

    device = draft_model.embed_tokens.weight.device

    hidden_states_for_draft = hidden_states_for_draft.to(device)
    input_ids = input_ids.to(device)
    output_embedding = base_model.get_output_embeddings().to(device)

    print("[DEBUG] hidden_states_for_draft.device:", hidden_states_for_draft.device)
    print("[DEBUG] input_ids.device:", input_ids.device)
    print("[DEBUG] base_model.get_output_embeddings().weight.device:", base_model.get_output_embeddings().weight.device)
    print("[DEBUG] draft_model.embed_tokens.weight.device:", draft_model.embed_tokens.weight.device)

    # 3. 얻은 hidden_states를 Draft 모델의 topK_genrate에 전달하여 draft_tokens를 생성합니다.
    draft_tokens, _, _, _ = draft_model.topK_genrate(
        hidden_states_for_draft, #!<- 길이가 맞춰진 hidden_states 전달
        input_ids,
        output_embedding,
        None # logits_processor,
    )
    
    # --- 끝: 핵심 수정 완료 ---

    if draft_tokens is None or draft_tokens.shape[1] == 0:
        return None, None
    
    # 3. 생성된 draft_tokens를 다시 Draft 모델에 통과시켜 최종 피처를 얻습니다.
    draft_outputs = draft_model(
        hidden_states=draft_model.embed_tokens(draft_tokens),
        input_ids=draft_tokens
    )
    final_draft_features = draft_outputs[0] if isinstance(draft_outputs, tuple) else draft_outputs

    # 4. 1차원으로 펼쳐서 반환합니다.
    return final_draft_features.view(-1, final_draft_features.shape[-1]).cpu(), draft_tokens.view(-1).cpu()

@torch.inference_mode()
def run_analysis(args):
    setup_seed(args.seed)

    # 1단계: Base 모델을 로드합니다. (이전과 동일)
    logger.info(f"Base 모델 로딩: {args.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    # 2단계: Draft 모델(CNet)을 수동으로 로드합니다. (이전과 동일)
    logger.info(f"Draft 모델(CNet) 로딩: {args.ea_model_path}")
    try:
        draft_config = AutoConfig.from_pretrained(args.ea_model_path)
        draft_model = CNet(draft_config)
        weights_path = hf_hub_download(repo_id=args.ea_model_path, filename="pytorch_model.bin")
        draft_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        draft_model.to(base_model.device) # Base 모델과 같은 디바이스로 이동
        logger.info("Draft 모델 가중치를 성공적으로 로드했습니다.")
    except Exception as e:
        logger.error(f"Draft 모델 로딩 실패: {e}")
        return

    # 1. Base 모델의 그래디언트 체크포인팅을 비활성화합니다.
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
        
    # 2. Draft 모델(CNet)의 그래디언트 체크포인팅 플래그를 비활성화합니다.
    if hasattr(draft_model, "gradient_checkpointing"):
        draft_model.gradient_checkpointing = False

    # Base 모델과 동일한 데이터 타입 및 디바이스로 이동
    draft_model.to(base_model.dtype)
    draft_model.to(base_model.device)
    base_model.eval()
    draft_model.eval()

    # 1. Draft 모델의 트리 생성을 위한 필수 초기화 함수를 호출합니다.
    draft_model.init_tree()
    logger.info("Draft 모델 트리 초기화 완료.")

    tokenizer_path = args.tokenizer_path or args.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Target 모델(Base)의 분류기 가중치를 미리 가져옵니다.
    target_classifier_weights = base_model.get_output_embeddings().weight.detach().cpu()
    logger.info("Target (Base) model의 classifier weights를 성공적으로 가져왔습니다.")

    analyzer = GeometricAnalyzer(tokenizer, min_samples_per_token=args.min_samples_per_token)
    
    full_dataset_iterable = load_dataset(args.dataset, split="train", streaming=True)
    full_dataset_iterator = iter(full_dataset_iterable)
    
    total_processed_tokens = 0
    with tqdm(total=args.num_samples, desc="전체 샘플 처리 중") as pbar:
        while total_processed_tokens < args.num_samples:
            batch_texts = []
            try:
                for _ in range(args.batch_size):
                    # Pile 데이터셋의 경우 'text' 필드를 사용합니다.
                    batch_texts.append(next(full_dataset_iterator)['text'])
            except StopIteration:
                # 데이터셋의 끝에 도달하면 루프를 종료합니다.
                break
            
            # 텍스트 배치를 토큰화하여 'inputs' 텐서를 생성합니다.
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            for i in range(inputs.input_ids.shape[0]):
                    input_id = inputs.input_ids[i:i+1]
                    
                    # 3단계: 더 이상 EAModel이 아닌, 로드된 두 모델을 직접 전달합니다.
                    draft_features, draft_tokens = capture_draft_features(
                        base_model,
                        draft_model,
                        input_id
                    )
                    
                    if draft_features is not None and draft_tokens is not None:
                        analyzer.update_batch(draft_features, draft_tokens)

                        # 처리한 배치 크기만큼 진행률 표시줄(progress bar)을 업데이트합니다.
            pbar.update(len(batch_texts))
            # 전체 처리된 샘플 수를 증가시킵니다.
            total_processed_tokens += len(batch_texts)

    logger.info("모든 샘플 처리 완료. 최종 메트릭 계산 중...")
    gnc2_stats, unc3_stats = analyzer.calculate_and_get_stats(target_classifier_weights)
    final_gnc2 = calculate_final_metrics_from_stats(gnc2_stats)
    final_unc3 = calculate_final_metrics_from_stats(unc3_stats)

    metrics = {
        'model_name': args.ea_model_path,
        'gnc2_avg': final_gnc2['avg'],
        'gnc2_cov': final_gnc2['cov'],
        'unc3_avg': final_unc3['avg'],
        'unc3_cov': final_unc3['cov'],
        'total_analyzed_tokens': total_processed_tokens 
    }

    # 결과를 파일에 저장
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as fout:
        json.dump(metrics, fout, indent=2)

    # 최종 결과 요약 출력
    logger.info("=== Final Draft Model Collapse Metrics Summary ===")
    logger.info(f"GNC2 Avg (Log Distances): {metrics.get('gnc2_avg', 0):.4f}")
    logger.info(f"GNC2 CoV (Uniformity): {metrics.get('gnc2_cov', 0):.4f}")
    logger.info(f"UNC3 Avg (Similarity): {metrics.get('unc3_avg', 0):.4f}")
    logger.info(f"UNC3 CoV (Duality): {metrics.get('unc3_cov', 0):.4f}")
    logger.info(f"분석에 사용된 총 토큰 수: {metrics.get('total_analyzed_tokens', 0)}")
    logger.info(f"결과가 {args.output_file} 파일에 저장되었습니다.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Draft Model의 Linguistic Collapse 분석")

    # --- 필수 인자 ---
    parser.add_argument("--ea-model-path", type=str, required=True, help="Draft 모델을 포함한 EA 모델의 경로")
    parser.add_argument("--base-model-path", type=str, required=True, help="Target(Base) 모델의 경로")
    parser.add_argument("--output-file", type=str, required=True, help="분석 결과를 저장할 JSON 파일 경로")

    # --- 선택 인자 ---
    parser.add_argument("--tokenizer-path", type=str, default=None, help="토크나이저 경로 (기본값: base-model-path)")
    parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted", help="분석에 사용할 데이터셋")
    parser.add_argument("--num-samples", type=int, default=10000, help="분석에 사용할 총 샘플 수")
    parser.add_argument("--batch-size", type=int, default=4, help="한 번에 처리할 배치 크기")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="eagenerate에서 생성할 최대 토큰 수")
    parser.add_argument("--min-samples-per-token", type=int, default=10, help="LNC 계산을 위한 토큰별 최소 샘플 수")
    parser.add_argument("--seed", type=int, default=42, help="재현성을 위한 랜덤 시드")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="모델을 로드할 장치")
    parser.add_argument("--total-token", type=int, default=60, help="추측 디코딩에 사용할 총 토큰 수")
    parser.add_argument("--depth", type=int, default=5, help="추측 디코딩 트리의 깊이")
    parser.add_argument("--top-k", type=int, default=9, help="추측 디코딩의 top-k 샘플링")
    parser.add_argument("--max-length", type=int, default=512, help="토큰화 시 최대 길이")
    parser.add_argument("--threshold", type=float, default=0.09, help="추측 디코딩 posterior threshold")
    
    args = parser.parse_args()
    
    def setup_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    setup_seed(args.seed)
    
    run_analysis(args)