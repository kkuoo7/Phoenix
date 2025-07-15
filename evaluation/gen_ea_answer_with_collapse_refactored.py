"""Generate answers with EA model and measure collapse metrics (with draft model).

Usage:
python3 gen_ea_answer_with_collapse_refactored.py --base-model-path meta-llama/Llama-3.1-8B-Instruct --ea-model-path path/to/ea_model --bench-name html_to_tsv --question-begin 0 --question-end 2 --max-new-token 128
"""
import argparse
import json
import os
import sys
import logging
from typing import Dict, Any

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
hass_dir = os.path.dirname(current_dir)
if hass_dir not in sys.path:
    sys.path.insert(0, hass_dir)

import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import time

from fastchat.llm_judge.common import load_questions
from model.ea_model import EaModel
from model.utils import *

# 새로운 모듈들 import
from collapse_collector import CollapseCollector
from collapse_config import CollapseConfig
from collapse_analyzer import CollapseAnalyzer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_collapse_config(args) -> CollapseConfig:
    """설정 객체 생성"""
    return CollapseConfig(
        save_every=1024,
        max_buffer_size=10000,
        device=args.device if hasattr(args, 'device') else "cpu",
        min_samples_per_token=2,
        continue_on_error=True
    )

@torch.inference_mode()
def get_model_answers(
    base_model_path,
    ea_model_path,
    model_id,
    questions,
    answer_file,
    collapse_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    args
):
    """리팩토링된 EA 모델 답변 생성 함수"""
    
    # 모델 초기화
    logger.info("Initializing EaModel...")
    logger.info(f"Base model path: {base_model_path}")
    logger.info(f"EA model path: {ea_model_path}")
    
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path if ea_model_path else None,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=args.threshold,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    logger.info("EaModel loaded successfully")
    
    tokenizer = model.get_tokenizer()
    logger.info("Tokenizer loaded successfully")
    
    # 토크나이저 설정 수정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    model.eval()
    logger.info("Model set to eval mode")
    
    # 모델 디바이스 추출
    device = next(model.parameters()).device
    logger.info(f"Model device: {device}")
    
    # Collapse 수집기 초기화
    collapse_config = create_collapse_config(args)
    
    all_turn_collapse = []
    chunk_entropies_by_index = []  # List of lists: each sublist is all entropies for that chunk index

    for question in tqdm(questions, desc="Processing questions"):
        for turn_idx, turn in enumerate(question["turns"]):
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                ]
                for j in range(turn_idx + 1):
                    qs = question["turns"][j]
                    messages.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
                prompt_len = len(input_ids[0])
                input_tensor = torch.as_tensor(input_ids).cuda()
                attention_mask = torch.ones_like(input_tensor)
                
                # 1. 텍스트 생성 (prompt -> 생성)
                with torch.no_grad():
                    outputs = model.generate(
                        input_tensor,
                        max_new_tokens=max_new_token,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False,
                        output_hidden_states=True,
                    )
                    full_ids = outputs.sequences  # (batch, prompt+gen_len)
                    gen_len = full_ids.shape[1] - prompt_len
                    if gen_len == 0:
                        logger.warning(f"질문 {question['question_id']} 턴 {turn_idx}: 생성 토큰 없음")
                        continue
                    # past_key_values 얻기 위해 prompt만 forward
                    prompt_outputs = model(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = prompt_outputs.past_key_values
                    # 생성 구간만 forward (메모리 효율)
                    gen_input_ids = full_ids[:, prompt_len:]
                    gen_attention_mask = torch.ones_like(gen_input_ids)
                    gen_outputs = model(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hidden_states = gen_outputs.hidden_states[-1]  # (batch, gen_len, hidden)
                    # 안전장치: hidden_states를 항상 float32 + GPU로 변환
                    if not hidden_states.is_cuda or hidden_states.dtype != torch.float32:
                        hidden_states = hidden_states.to(dtype=torch.float32, device=model.device)
                    # CollapseCollector를 사용해 생성 토큰 구간의 hidden states만 추출
                    turn_collector = CollapseCollector(model, tokenizer, collapse_config)
                    features, token_ids = turn_collector._extract_features_accurate(
                        hidden_states, gen_input_ids, gen_attention_mask
                    )
                    if features is not None and (not features.is_cuda or features.dtype != torch.float32):
                        features = features.to(dtype=torch.float32, device=model.device)

                if features is not None and token_ids is not None:
                    turn_collector.add_batch_features(features, token_ids)
                    metrics = turn_collector.get_collapse_metrics(input_len=prompt_len, num_chunks=5)
                    all_turn_collapse.append({
                        "question_id": question["question_id"],
                        "turn_idx": turn_idx,
                        "chunk_svd_entropies": metrics['chunk_svd_entropies'],
                        "total_generated_tokens": metrics['total_generated_tokens']
                    })
                    # summary용 청크별 엔트로피 수집
                    chunk_entropies = metrics['chunk_svd_entropies']
                    for i, ent in enumerate(chunk_entropies):
                        if ent is not None:
                            if len(chunk_entropies_by_index) <= i:
                                chunk_entropies_by_index.append([])
                            chunk_entropies_by_index[i].append(ent)
                turn_collector.clear()
            except Exception as e:
                logger.error(f"Error in question {question['question_id']} turn {turn_idx}: {e}")
                continue
    # summary: 각 청크별 평균 SVD entropy
    num_chunks = 5
    avg_chunk_svd_entropies = []
    for i in range(num_chunks):
        chunk_vals = chunk_entropies_by_index[i] if i < len(chunk_entropies_by_index) else []
        avg = float(np.mean(chunk_vals)) if chunk_vals else None
        avg_chunk_svd_entropies.append(avg)
    summary = {f'avg_chunk{i+1}_svd_entropy': avg_chunk_svd_entropies[i] for i in range(num_chunks)}
    summary['num_turns'] = len(all_turn_collapse)
    # 저장
    try:
        os.makedirs(os.path.dirname(collapse_file), exist_ok=True)
        with open(collapse_file, "w") as fout:
            json.dump({'per_turn': all_turn_collapse, 'summary': summary}, fout, indent=2)
        logger.info(f"Per-turn collapse metrics and summary saved to {collapse_file}")
    except Exception as e:
        logger.error(f"Failed to save collapse metrics: {e}")
    
    # 메모리 정리
    del model
    torch.cuda.empty_cache()

def run_eval(
    base_model_path,
    ea_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    collapse_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    args
):
    """평가 실행 함수"""
    questions = load_questions(question_file, question_begin, question_end)
    
    # 단일 GPU 실행 (Ray 없이)
    get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        collapse_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
    )

def reorg_answer_file(answer_file):
    """답변 파일 재구성"""
    # 기존 함수와 동일하게 유지
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path", type=str, required=True)  # EA 모델은 필수
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--bench-name", type=str, required=True)
    parser.add_argument("--question-begin", type=int, default=0)
    parser.add_argument("--question-end", type=int, default=-1)
    parser.add_argument("--answer-file", type=str, required=True)
    parser.add_argument("--collapse-file", type=str, required=True)
    parser.add_argument("--max-new-token", type=int, default=512)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str, default="13GiB")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--total-token", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.1)
    
    args = parser.parse_args()
    
    setup_seed(0)
    
    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        args.answer_file,
        args.collapse_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    ) 