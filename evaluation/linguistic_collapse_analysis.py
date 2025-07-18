import argparse
import json
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from itertools import islice

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, IterableDataset

# geometric_analyzer.py의 코드를 여기에 직접 포함하거나 import합니다.
# --- 시작: geometric_analyzer.py 내용 ---

logger = logging.getLogger(__name__)

def to_tensor_safe(f):
    if isinstance(f, torch.Tensor):
        return f.detach().clone()
    elif isinstance(f, list):
        return torch.stack([to_tensor_safe(x) for x in f])
    else:
        return torch.tensor(f, dtype=torch.float32)

class GeometricAnalyzer:
    """
    Linguistic Collapse 논문에 기반하여 GNC2/UNC3 등 기하학적 collapse 메트릭을 계산하는 클래스.
    """
    def __init__(self, tokenizer=None, min_samples_per_token=2):
        self.tokenizer = tokenizer
        self.min_samples_per_token = min_samples_per_token

    def group_features_by_token(self, features, token_ids):
        token_features = {}
        special_token_ids = set()
        if self.tokenizer is not None:
            # 토크나이저의 모든 스페셜 토큰 ID를 가져옵니다.
            special_token_ids = set(self.tokenizer.all_special_ids)

        for i, token_id in enumerate(token_ids):
            token_id = token_id.item() if hasattr(token_id, 'item') else int(token_id)
            if token_id in special_token_ids:
                continue
            if token_id not in token_features:
                token_features[token_id] = []
            token_features[token_id].append(features[i])

        for token_id in list(token_features.keys()):
            if len(token_features[token_id]) >= self.min_samples_per_token:
                feature_list = token_features[token_id]
                try:
                    # 리스트 내 모든 피처를 텐서로 변환하여 스택합니다.
                    tensor_features = [to_tensor_safe(f) for f in feature_list]
                    token_features[token_id] = torch.stack(tensor_features)
                except Exception as e:
                    logger.error(f"토큰 {token_id}의 텐서 스택 실패: {e}")
                    del token_features[token_id]
            else:
                # 샘플 수가 부족한 토큰은 분석에서 제외합니다.
                del token_features[token_id]
        return token_features

    def compute_gnc2(self, token_features):
        """GNC2 (Hyperspherical Uniformity)를 계산합니다."""
        try:
            class_means = []
            for token_id, features in token_features.items():
                if features.shape[0] > 1:
                    class_means.append(features.mean(dim=0))

            if len(class_means) < 2:
                return 0.0

            class_means = torch.stack(class_means)
            global_mean = class_means.mean(dim=0, keepdim=True)
            centered_means = class_means - global_mean
            normalized_means = torch.nn.functional.normalize(centered_means, p=2, dim=1)
            pdist = torch.pdist(normalized_means.float(), p=2)
            # 0으로 나누는 것을 방지하기 위해 작은 값을 더합니다.
            log_inv_distances = torch.log(1.0 / (pdist + 1e-8))
            
            mean_dist = log_inv_distances.mean()
            std_dist = log_inv_distances.std()
            
            # CoV (Coefficient of Variation)를 반환합니다.
            return (std_dist / mean_dist).item() if mean_dist != 0 else 0.0
        except Exception as e:
            logger.error(f"GNC2 계산 실패: {e}")
            return 0.0

    def compute_unc3(self, token_features, classifier_weights):
        """ UNC3 (Uniform Duality)를 계산합니다. """
        try:
            class_means_map = {
                token_id: features.mean(dim=0)
                for token_id, features in token_features.items()
                if features.shape[0] > 1
            }

            if len(class_means_map) < 2:
                return 0.0
            
            # class_means와 classifier_weights에 공통으로 존재하는 토큰 ID를 찾습니다.
            common_ids = sorted(list(
                class_means_map.keys() &
                set(range(classifier_weights.shape[0]))
            ))

            if len(common_ids) < 2:
                return 0.0

            class_means = torch.stack([class_means_map[cid] for cid in common_ids])
            weights = classifier_weights[common_ids, :]
            
            global_mean = class_means.mean(dim=0, keepdim=True)
            normalized_means = torch.nn.functional.normalize(class_means - global_mean, p=2, dim=1)
            normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
            
            # 코사인 유사도를 계산합니다.
            similarities = (normalized_weights * normalized_means).sum(dim=1)
            
            mean_sim = similarities.mean()
            std_sim = similarities.std()
            
            # CoV (Coefficient of Variation)를 반환합니다.
            return (std_sim / abs(mean_sim)).item() if mean_sim != 0 else 0.0
        except Exception as e:
            logger.error(f"UNC3 계산 실패: {e}")
            return 0.0

    def get_gnc2_unc3_metrics(self, features, token_ids, classifier_weights=None):
        token_features = self.group_features_by_token(features, token_ids)
        gnc2 = self.compute_gnc2(token_features)
        
        unc3 = None
        if classifier_weights is not None:
            unc3 = self.compute_unc3(token_features, classifier_weights)
            
        return {'gnc2': gnc2, 'unc3': unc3, 'analyzed_token_classes': len(token_features)}

# --- 끝: geometric_analyzer.py 내용 ---


# Pile 데이터셋 설정
PILE_DATASET_ID = "monology/pile-uncopyrighted"
HIGH_QUALITY_SUBSETS = {
    "Pile-CC", "PubMed Central", "arXiv", "Wikipedia"
}

def is_high_quality(example):
    return example['meta']['pile_set_name'] in HIGH_QUALITY_SUBSETS

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_pile_batch_iterator(
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_samples_to_yield: int = 10000
):
    logger.info("데이터셋 스트리밍 및 필터링 설정 중...")
    dataset_stream = load_dataset(PILE_DATASET_ID, split="train", streaming=True)
    filtered_stream = dataset_stream.filter(is_high_quality)
    limited_stream = islice(filtered_stream, num_samples_to_yield)

    def collate_fn(examples):
        texts = [example["text"] for example in examples]
        tokenized_inputs = tokenizer.batch_encode_plus(  # type: ignore
            texts,
            padding='longest',
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        return tokenized_inputs

    def batch_iterator():
        batch = []
        for example in limited_stream:
            batch.append(example)
            if len(batch) == batch_size:
                yield collate_fn(batch)
                batch = []
        if batch:
            yield collate_fn(batch)

    logger.info(f"Pile 배치 이터레이터 생성 완료: 배치 크기 {batch_size}, 최대 길이 {max_length}")
    return batch_iterator()

@torch.inference_mode()
def run_pile_analysis(args):
    """Pile 데이터셋을 사용한 collapse 메트릭 측정"""
    setup_seed(args.seed)
    
    logger.info(f"모델 로딩: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"{args.num_samples}개 샘플로 Pile 데이터 로더 생성 중")
    pile_batch_iterator = create_pile_batch_iterator(
        tokenizer, 
        batch_size=args.batch_size, 
        max_length=args.max_length,
        num_samples_to_yield=args.num_samples
    )
    
    all_features = []
    all_token_ids = []
    
    logger.info("피처 벡터 수집 시작...")
    for batch in tqdm(pile_batch_iterator, desc="Pile 배치 처리 중"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        # 모델 포워딩 및 마지막 히든 상태 추출
        outputs = model(**batch, output_hidden_states=True)
        # 마지막 레이어의 히든 상태가 penultimate 피처에 해당
        last_hidden_states = outputs.hidden_states[-1] 

        # 다음 토큰 예측에 사용되는 피처는 각 시퀀스의 마지막 토큰을 제외한 모든 토큰의 피처
        # 라벨(정답)은 첫번째 토큰을 제외한 모든 토큰
        # 따라서, attention_mask를 고려하여 유효한 피처와 토큰 ID만 수집
        attention_mask = batch['attention_mask']
        valid_indices = attention_mask[:, :-1].bool().flatten()
        
        # (batch, seq_len, dim) -> (batch * seq_len, dim)
        features = last_hidden_states[:, :-1, :].reshape(-1, model.config.hidden_size)[valid_indices]
        token_ids = batch['input_ids'][:, 1:].reshape(-1)[valid_indices]
        
        all_features.append(features.cpu())
        all_token_ids.append(token_ids.cpu())
    
    logger.info("피처 벡터 수집 완료!")

    # 수집된 피처들을 하나의 텐서로 결합
    final_features = torch.cat(all_features, dim=0)
    final_token_ids = torch.cat(all_token_ids, dim=0)
    
    logger.info(f"총 {final_features.shape[0]}개의 토큰 피처 수집됨.")
    
    # Collapse 메트릭 계산
    logger.info("Collapse 메트릭 계산 중...")
    analyzer = GeometricAnalyzer(tokenizer, min_samples_per_token=args.min_samples_per_token)
    
    # UNC3 계산을 위해 classifier weights (output embedding) 가져오기
    classifier_weights = model.get_output_embeddings().weight.detach().cpu()
    
    collapse_metrics = analyzer.get_gnc2_unc3_metrics(
        final_features, 
        final_token_ids,
        classifier_weights
    )
    
    # 결과 저장 및 출력
    os.makedirs(os.path.dirname(args.collapse_file), exist_ok=True)
    with open(args.collapse_file, "w") as fout:
        json.dump(collapse_metrics, fout, indent=2)
    
    logger.info("=== Collapse Metrics Summary ===")
    logger.info(f"GNC2 (Hyperspherical Uniformity): {collapse_metrics.get('gnc2', 0):.4f}")
    logger.info(f"UNC3 (Uniform Duality): {collapse_metrics.get('unc3', 0):.4f}")
    logger.info(f"분석에 사용된 토큰 종류 수: {collapse_metrics.get('analyzed_token_classes', 0)}")
    logger.info(f"결과가 {args.collapse_file} 파일에 저장되었습니다.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Linguistic Collapse Metric (GNC2, UNC3) Evaluation on Pile Dataset")
    parser.add_argument("--model-path", type=str, required=True, help="HuggingFace 모델 경로 (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="토크나이저 경로 (기본값: model-path와 동일)")
    parser.add_argument("--num-samples", type=int, default=10000, help="처리할 샘플 수")
    parser.add_argument("--batch-size", type=int, default=8, help="배치 크기")
    parser.add_argument("--max-length", type=int, default=512, help="최대 시퀀스 길이")
    parser.add_argument("--collapse-file", type=str, required=True, help="Collapse 메트릭 저장 파일 경로")
    parser.add_argument("--min-samples-per-token", type=int, default=10, help="클래스(토큰)별 최소 샘플 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    
    args = parser.parse_args()
    
    run_pile_analysis(args)