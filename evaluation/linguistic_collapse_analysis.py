import argparse
import json
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from itertools import islice
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, IterableDataset

logger = logging.getLogger(__name__)

# --- 시작: 수정된 GeometricAnalyzer 내용 ---

class GeometricAnalyzer:
    """
    Linguistic Collapse 논문에 기반하여 GNC2/UNC3 등 기하학적 collapse 메트릭을 계산하는 클래스.
    """
    def __init__(self, tokenizer=None, min_samples_per_token=2, device='cpu'):
        self.tokenizer = tokenizer
        self.min_samples_per_token = min_samples_per_token
        self.device = device
        self.special_token_ids = set(self.tokenizer.all_special_ids) if self.tokenizer else set()
        self.reset()

    def reset(self):
        """내부 통계량을 초기화합니다."""
        self.token_sums = {}
        self.token_counts = {}

    def update_batch(self, features: torch.Tensor, token_ids: torch.Tensor):
        """배치 단위로 들어온 피처의 합계와 카운트를 누적합니다."""
        unique_tokens, inverse_indices, counts = torch.unique(token_ids, return_inverse=True, return_counts=True)
        
        for i, token_id in enumerate(unique_tokens):
            token_id_item = token_id.item()
            if token_id_item in self.special_token_ids:
                continue
            
            # 해당 토큰에 대한 피처들을 합산
            token_features_sum = features[inverse_indices == i].sum(dim=0)
            
            if token_id_item not in self.token_counts:
                self.token_counts[token_id_item] = 0
                self.token_sums[token_id_item] = torch.zeros_like(token_features_sum, device=self.device)
            
            self.token_counts[token_id_item] += counts[i].item()
            self.token_sums[token_id_item] += token_features_sum

    def _get_class_means(self):
        """누적된 합계와 카운트로부터 클래스 평균(class means)을 계산합니다."""
        class_means = {}
        for token_id, count in self.token_counts.items():
            if count >= self.min_samples_per_token:
                class_means[token_id] = self.token_sums[token_id] / count
        return class_means


    def calculate_and_get_stats(self, classifier_weights):
        """누적된 통계량으로 GNC2와 UNC3 중간 통계량을 계산합니다."""
        class_means_map = self._get_class_means()
        
        gnc2_stats = self._compute_gnc2_stats_from_means(class_means_map)
        unc3_stats = self._compute_unc3_stats_from_means(class_means_map, classifier_weights)
        
        return gnc2_stats, unc3_stats

    def _compute_gnc2_stats_from_means(self, class_means_map):
        try:
            if len(class_means_map) < 2:
                return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}

            class_means = torch.stack(list(class_means_map.values())).to(self.device)
            global_mean = class_means.mean(dim=0, keepdim=True)
            centered_means = class_means - global_mean
            normalized_means = torch.nn.functional.normalize(centered_means, p=2, dim=1)
            
            pdist = torch.pdist(normalized_means.float(), p=2)
            log_inv_distances = torch.log(1.0 / (pdist + 1e-8))
            
            return {
                'n': len(log_inv_distances),
                'sum': log_inv_distances.sum().item(),
                'sum_sq': (log_inv_distances ** 2).sum().item()
            }
        except Exception as e:
            logger.error(f"GNC2 통계량 계산 실패: {e}")
            return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}

    def _compute_unc3_stats_from_means(self, class_means_map, classifier_weights):
        try:
            if len(class_means_map) < 2:
                return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
            
            common_ids = sorted(list(class_means_map.keys() & set(range(classifier_weights.shape[0]))))
            if len(common_ids) < 2:
                return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}

            class_means = torch.stack([class_means_map[cid] for cid in common_ids]).to(self.device)
            weights = classifier_weights[common_ids, :].to(self.device)
            
            global_mean = class_means.mean(dim=0, keepdim=True)
            normalized_means = torch.nn.functional.normalize(class_means - global_mean, p=2, dim=1)
            normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
            
            similarities = (normalized_weights * normalized_means).sum(dim=1)
            
            return {
                'n': len(similarities),
                'sum': similarities.sum().item(),
                'sum_sq': (similarities ** 2).sum().item()
            }
        except Exception as e:
            logger.error(f"UNC3 통계량 계산 실패: {e}")
            return {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}

# --- 끝: 수정된 GeometricAnalyzer 내용 ---

# Pile 데이터셋 설정 (기존 코드와 동일)
PILE_DATASET_ID = "monology/pile-uncopyrighted"
HIGH_QUALITY_SUBSETS = {"Pile-CC", "PubMed Central", "arXiv", "Wikipedia"}

def is_high_quality(example):
    return example['meta']['pile_set_name'] in HIGH_QUALITY_SUBSETS

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_pile_batch_iterator(
    tokenizer: AutoTokenizer,
    full_dataset_stream,
    batch_size: int = 8,
    max_length: int = 512,
):
    # (기존 코드와 유사하나, 전체 데이터셋 스트림을 인자로 받도록 수정)
    def collate_fn(examples):
        texts = [example["text"] for example in examples]
        tokenized_inputs = tokenizer(
            texts,
            padding='longest',
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        return tokenized_inputs

    def batch_iterator():
        batch = []
        for example in full_dataset_stream:
            batch.append(example)
            if len(batch) == batch_size:
                yield collate_fn(batch)
                batch = []
        if batch:
            yield collate_fn(batch)

    return batch_iterator()
    
def calculate_final_metrics_from_stats(total_stats):
    """ 누적된 통계량으로 최종 avg, CoV를 계산합니다. """
    n = total_stats['n']
    if n == 0:
        return {'avg': 0.0, 'cov': 0.0}
    
    sum_x = total_stats['sum']
    sum_x_sq = total_stats['sum_sq']
    
    mean = sum_x / n
    # 분산 = E[X^2] - (E[X])^2
    variance = (sum_x_sq / n) - (mean ** 2)
    if variance < 0: variance = 0 # 부동소수점 오류 방지
    std = math.sqrt(variance)
    
    cov = (std / abs(mean)) if mean != 0 else 0.0
    return {'avg': mean, 'cov': cov}

@torch.inference_mode()
def run_pile_analysis(args):
    setup_seed(args.seed)
    
    logger.info(f"모델 로딩: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    chunk_size = 10000
    num_chunks = math.ceil(args.num_samples / chunk_size)
    
    total_gnc2_stats = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
    total_unc3_stats = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
    total_analyzed_tokens = 0
    
    logger.info(f"총 {args.num_samples}개 샘플을 {chunk_size}개 단위, {num_chunks}개 chunk로 나누어 처리합니다.")
    
    full_dataset_iterable = load_dataset(PILE_DATASET_ID, split="train", streaming=True).filter(is_high_quality)
    full_dataset_iterator = iter(full_dataset_iterable)
    
    # --- 시작: 메모리 효율적인 처리 로직 ---
    for i in range(num_chunks):
        logger.info(f"--- Chunk {i+1}/{num_chunks} 처리 시작 ---")
        
        chunk_stream = islice(full_dataset_iterator, chunk_size)
        pile_batch_iterator = create_pile_batch_iterator(
            tokenizer, chunk_stream, batch_size=args.batch_size, max_length=args.max_length
        )
        
        # Analyzer를 각 청크마다 새로 생성하여 상태 관리
        analyzer = GeometricAnalyzer(tokenizer, min_samples_per_token=args.min_samples_per_token, device='cpu')
        
        chunk_total_tokens = 0
        for batch in tqdm(pile_batch_iterator, desc=f"Chunk {i+1} 처리 중"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            attention_mask = batch['attention_mask']
            valid_indices = attention_mask[:, :-1].bool().flatten()
            features = last_hidden_states[:, :-1, :].reshape(-1, model.config.hidden_size)[valid_indices]
            token_ids = batch['input_ids'][:, 1:].reshape(-1)[valid_indices]
            
            # 피처를 리스트에 저장하는 대신, 즉시 통계량 업데이트
            analyzer.update_batch(features.cpu(), token_ids.cpu())
            chunk_total_tokens += len(features)

        if chunk_total_tokens == 0:
            logger.warning(f"Chunk {i+1}에서 처리할 데이터가 없습니다.")
            continue
        
        total_analyzed_tokens += chunk_total_tokens
        logger.info(f"Chunk {i+1}: {chunk_total_tokens}개 토큰 피처 처리 완료. Collapse 통계량 계산 중...")
        
        classifier_weights = model.get_output_embeddings().weight.detach().cpu()
        
        # 청크의 누적된 통계량으로 GNC2/UNC3 중간 결과 계산
        gnc2_stats, unc3_stats = analyzer.calculate_and_get_stats(classifier_weights)
        
        # 전체 통계량에 누적
        total_gnc2_stats['n'] += gnc2_stats['n']
        total_gnc2_stats['sum'] += gnc2_stats['sum']
        total_gnc2_stats['sum_sq'] += gnc2_stats['sum_sq']
        
        total_unc3_stats['n'] += unc3_stats['n']
        total_unc3_stats['sum'] += unc3_stats['sum']
        total_unc3_stats['sum_sq'] += unc3_stats['sum_sq']

    # --- 끝: 메모리 효율적인 처리 로직 ---
    
    logger.info("모든 chunk 처리 완료. 최종 메트릭 계산 중...")
    
    final_gnc2_metrics = calculate_final_metrics_from_stats(total_gnc2_stats)
    final_unc3_metrics = calculate_final_metrics_from_stats(total_unc3_stats)

    collapse_metrics = {
        'gnc2_cov': final_gnc2_metrics['cov'], 
        'gnc2_avg': final_gnc2_metrics['avg'],
        'unc3_cov': final_unc3_metrics['cov'],
        'unc3_avg': final_unc3_metrics['avg'],
        'total_analyzed_tokens': total_analyzed_tokens
    }

    
    os.makedirs(os.path.dirname(args.collapse_file), exist_ok=True)
    with open(args.collapse_file, "w") as fout:
        json.dump(collapse_metrics, fout, indent=2)
    
    logger.info("=== Final Collapse Metrics Summary ===")
    logger.info(f"GNC2 Avg (Log Distances): {collapse_metrics.get('gnc2_avg', 0):.4f}")
    logger.info(f"GNC2 CoV (Uniformity): {collapse_metrics.get('gnc2_cov', 0):.4f}")
    logger.info(f"UNC3 Avg (Similarity): {collapse_metrics.get('unc3_avg', 0):.4f}")
    logger.info(f"UNC3 CoV (Duality): {collapse_metrics.get('unc3_cov', 0):.4f}")
    logger.info(f"분석에 사용된 총 토큰 수: {total_analyzed_tokens}")
    logger.info(f"결과가 {args.collapse_file} 파일에 저장되었습니다.")

if __name__ == "__main__":
    # (기존 코드와 동일)
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Linguistic Collapse Metric (GNC2, UNC3) Evaluation on Pile Dataset")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--collapse-file", type=str, required=True)
    parser.add_argument("--min-samples-per-token", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pile_analysis(args)