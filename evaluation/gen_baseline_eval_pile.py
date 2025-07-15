"""Generate baseline evaluation with Pile dataset and measure collapse metrics (GNC2, UNC3, SVD entropy).

Usage:
python3 gen_baseline_eval_pile.py --model-path meta-llama/Llama-2-7b-hf --num-samples 10000 --batch-size 8 --max-length 512 --collapse-file results/pile_collapse_metrics.json
"""
import argparse
import json
import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from itertools import islice

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
hass_dir = os.path.dirname(current_dir)
if hass_dir not in sys.path:
    sys.path.insert(0, hass_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, IterableDataset

# 기존 collapse 모듈들 import
from collapse_collector import CollapseCollector
from collapse_config import CollapseConfig
from collapse_analyzer import CollapseAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pile 데이터셋 설정
PILE_DATASET_ID = "EleutherAI/pile"

# 고품질 서브셋 정의
HIGH_QUALITY_SUBSETS = {
    "Pile-CC", "PubMed Central", "Books3", "OpenWebText2", "arXiv", "Wikipedia"
}

def is_high_quality(example):
    """주어진 샘플이 고품질 서브셋에 속하는지 확인하는 필터 함수"""
    return example['meta']['pile_set_name'] in HIGH_QUALITY_SUBSETS

def setup_seed(seed):
    """시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_collapse_config(args) -> CollapseConfig:
    """CollapseConfig 객체 생성"""
    return CollapseConfig(
        save_every=1024,
        max_buffer_size=10000,
        device=args.device if hasattr(args, 'device') else "cpu",
        min_samples_per_token=2
    )

def create_pile_dataloader(
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    num_samples_to_yield: int = 10000
) -> torch.utils.data.DataLoader:
    """
    The Pile의 고품질 서브셋을 스트리밍하고 배치로 묶어주는 데이터 로더를 생성합니다.

    Args:
        tokenizer: 사용할 토크나이저
        batch_size: 한 번에 처리할 배치 크기
        max_length: 토크나이저의 최대 길이
        num_samples_to_yield: 전체 데이터셋에서 처리할 샘플의 총 개수

    Returns:
        torch.utils.data.DataLoader: PyTorch 모델에 바로 입력할 수 있는 데이터 로더
    """
    logger.info("데이터셋 스트리밍 및 필터링 설정 중...")
    
    # 1. 스트리밍으로 데이터셋 열기
    dataset_stream = load_dataset(PILE_DATASET_ID, split="train", streaming=True)
    
    # 2. 고품질 서브셋 필터링
    filtered_stream = dataset_stream.filter(is_high_quality)
    logger.info("고품질 서브셋 필터링 완료")

    # 3. 필요한 만큼의 샘플만 가져오도록 스트림 제한
    limited_stream = islice(filtered_stream, num_samples_to_yield)
    
    # 4. 실시간으로 토크나이징 및 포맷팅하는 함수
    def collate_fn(examples):
        texts = [example["text"] for example in examples]
        # padding='longest'는 배치 내 가장 긴 샘플에 맞춰 패딩합니다.
        tokenized_inputs = tokenizer(
            texts, 
            padding='longest', # 'longest' 또는 True
            max_length=max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        return tokenized_inputs

    # islice로 생성된 iterator를 IterableDataset으로 한번 감싸줍니다.
    iterable_dataset = IterableDataset.from_generator(lambda: limited_stream)

    # 5. 배치 단위로 묶어주는 DataLoader 생성
    # collate_fn을 사용하여 배치 생성 방식을 커스터마이징합니다.
    dataloader = torch.utils.data.DataLoader(
        iterable_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    logger.info(f"Pile 데이터 로더 생성 완료: 배치 크기 {batch_size}, 최대 길이 {max_length}")
    return dataloader

class SimpleModelWrapper:
    """허깅페이스 모델을 HASS collapse 시스템과 호환되도록 래핑"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def __call__(self, input_ids, attention_mask=None, **kwargs):
        """forward pass for feature extraction"""
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

@torch.inference_mode()
def get_model_answers_pile(
    model_path,
    tokenizer_path,
    num_samples,
    batch_size,
    max_length,
    collapse_file,
    args
):
    """Pile 데이터셋을 사용한 collapse 메트릭 측정"""
    
    # 모델 및 토크나이저 로드
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 토크나이저 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 래핑
    wrapped_model = SimpleModelWrapper(model, tokenizer)
    
    # Pile 데이터 로더 생성
    logger.info(f"Creating Pile dataloader with {num_samples} samples")
    pile_dataloader = create_pile_dataloader(
        tokenizer, 
        batch_size=batch_size, 
        max_length=max_length,
        num_samples_to_yield=num_samples
    )
    
    # Collapse 수집기 초기화
    collapse_config = create_collapse_config(args)
    collapse_collector = CollapseCollector(wrapped_model, tokenizer, collapse_config)
    
    # 피처 벡터 수집
    logger.info("피처 벡터 수집 시작...")
    processed_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(pile_dataloader, desc="Processing Pile batches"):
            # 배치를 GPU로 이동
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # CollapseCollector를 사용하여 특성 수집
            features, token_ids = collapse_collector.collect_features_efficient(
                batch['input_ids'], batch['attention_mask']
            )
            
            if features is not None and token_ids is not None:
                collapse_collector.add_batch_features(features, token_ids)
                processed_batches += 1
                
                if processed_batches % 10 == 0:
                    logger.info(f"  - 처리된 배치: {processed_batches}")
    
    logger.info(f"피처 벡터 수집 완료! 총 {processed_batches}개 배치 처리")
    
    # Collapse 메트릭 계산 및 저장
    logger.info("Computing collapse metrics...")
    try:
        collapse_metrics = collapse_collector.get_collapse_metrics()
        if collapse_metrics:
            # 기본 메트릭 저장
            os.makedirs(os.path.dirname(collapse_file), exist_ok=True)
            with open(collapse_file, "w") as fout:
                json.dump(collapse_metrics, fout, indent=2)
            logger.info(f"Collapse metrics saved to {collapse_file}")
            logger.info(
                f"Total tokens analyzed: {collapse_metrics.get('total_tokens', 0)}"
            )
            
            # 추가 분석 리포트 생성
            analyzer = CollapseAnalyzer(collapse_config)
            analysis_file = collapse_file.replace('.json', '_analysis.json')
            analysis = analyzer.generate_report(collapse_metrics, analysis_file)
            if analysis:
                logger.info(f"Collapse analysis saved to {analysis_file}")
                
            # 결과 요약 출력
            overall_metrics = collapse_metrics.get("overall_metrics", {})
            logger.info("=== Collapse Metrics Summary ===")
            logger.info(f"GNC2: {overall_metrics.get('gnc2', 0):.4f}")
            logger.info(f"UNC3: {overall_metrics.get('unc3', 0):.4f}")
            logger.info(f"SVD Entropy: {overall_metrics.get('svd_entropy', 0):.4f}")
            logger.info(f"Total tokens: {collapse_metrics.get('total_tokens', 0)}")
            
        else:
            logger.warning("No collapse metrics computed")
    except Exception as e:
        logger.error(f"Failed to compute collapse metrics: {e}")
        if not collapse_config.continue_on_error:
            raise
    
    # 메모리 정리
    collapse_collector.clear()
    del model
    torch.cuda.empty_cache()

def run_eval(
    model_path,
    tokenizer_path,
    num_samples,
    batch_size,
    max_length,
    collapse_file,
    args
):
    """평가 실행 함수"""
    get_model_answers_pile(
        model_path,
        tokenizer_path,
        num_samples,
        batch_size,
        max_length,
        collapse_file,
        args
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                       help="허깅페이스 모델 경로 (예: meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                       help="토크나이저 경로 (기본값: model-path와 동일)")
    parser.add_argument("--num-samples", type=int, default=10000,
                       help="처리할 샘플 수")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="배치 크기")
    parser.add_argument("--max-length", type=int, default=512,
                       help="최대 시퀀스 길이")
    parser.add_argument("--collapse-file", type=str, required=True,
                       help="collapse 메트릭 저장 파일 경로")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    
    args = parser.parse_args()
    
    # 토크나이저 경로가 지정되지 않으면 모델 경로와 동일하게 설정
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    
    setup_seed(args.seed)
    
    logger.info("=== Pile Dataset Collapse Evaluation ===")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tokenizer: {args.tokenizer_path}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Output file: {args.collapse_file}")
    
    run_eval(
        args.model_path,
        args.tokenizer_path,
        args.num_samples,
        args.batch_size,
        args.max_length,
        args.collapse_file,
        args
    )