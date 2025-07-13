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
    collapse_collector = CollapseCollector(model, tokenizer, collapse_config)
    
    # Warmup
    logger.info("Starting warmup...")
    for warmup_iter in range(3):
        logger.info(f"Warmup iteration {warmup_iter + 1}/3")
        torch.manual_seed(0)
        question = questions[0]
        
        messages = [
            {"role": "system", 
            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
        
        for j in range(len(question["turns"])):
            logger.info(f"Processing turn {j + 1}/{len(question['turns'])}")
            qs = question["turns"][j]
            messages.append({"role": "user", "content": qs})
            
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                logger.info(f"Prompt length: {len(prompt)} characters")
                
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
                logger.info(f"Input IDs length: {len(input_ids[0])} tokens")
                
                # Warmup에서는 collapse 벡터 수집하지 않음
                input_tensor = torch.as_tensor(input_ids).cuda()
                logger.info(f"Input tensor shape: {input_tensor.shape}")
            
                # EA 모델 생성 (draft 포함) - 베이스 코드 스타일
                torch.cuda.synchronize()
                start_time = time.time()
                
                logger.info("Calling model.eagenerate()...")
                output_ids, new_token, idx, _ = model.eagenerate(
                    input_tensor, temperature=temperature, log=True, is_llama3=True
                )
                logger.info(f"Generation completed: {new_token} new tokens, idx: {idx}")
                
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                logger.info(f"Generation time: {total_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in warmup turn {j}: {e}")
                raise
            
            # 출력 처리
            output_ids = output_ids[0][len(input_ids[0]):]
            stop_token_ids = [
                tokenizer.eos_token_id, 
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            if stop_token_ids:
                stop_token_ids_index = [
                    i for i, id in enumerate(output_ids) if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[:stop_token_ids_index[0]]
            
            output = tokenizer.decode(
                output_ids, spaces_between_special_tokens=False
            )
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()
            
            messages.append({"role": "assistant", "content": output})
    
    logger.info("Warmup completed")
    
    # 메인 평가 루프
    logger.info(f"Starting main evaluation loop with {len(questions)} questions")
    for question in tqdm(questions, desc="Processing questions"):
        logger.info(f"Processing question {question['question_id']}: {question['turns'][0][:50]}...")
        choices = []
        
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
                
                # 베이스 코드와 동일한 방식으로 디바이스 처리
                torch.cuda.synchronize()
                start_time = time.time()
                
                output_ids, new_token, idx, accept_length_list = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),  # 베이스 코드와 동일
                    temperature=temperature,
                    log=True,
                    is_llama3=True
                )
                
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                
                # Collapse 벡터 수집 (메인 평가에서만)
                try:
                    # attention_mask 생성
                    attention_mask = torch.ones_like(torch.as_tensor(input_ids).cuda())
                    
                    # 특성 수집
                    features, token_ids = collapse_collector.collect_features_efficient(
                        input_ids=torch.as_tensor(input_ids).cuda(),
                        attention_mask=attention_mask
                    )
                    
                    if features is not None and token_ids is not None:
                        collapse_collector.add_batch_features(features, token_ids)
                        logger.info(f"Collected collapse features: {features.shape}, tokens: {token_ids.shape}")
                    else:
                        logger.warning("Failed to collect collapse features")
                        
                except Exception as e:
                    logger.warning(f"Failed to collect collapse features: {e}")
                
                # 출력 처리
                output_ids = output_ids[0][len(input_ids[0]):]
                stop_token_ids = [
                    tokenizer.eos_token_id, 
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
                
                if stop_token_ids:
                    stop_token_ids_index = [
                        i for i, id in enumerate(output_ids) if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[:stop_token_ids_index[0]]
                
                output = tokenizer.decode(
                    output_ids, spaces_between_special_tokens=False
                )
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                
                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({"role": "assistant", "content": output})
            
            choices.append({
                "index": i, "turns": turns, "idxs": idxs, 
                "new_tokens": new_tokens, "wall_time": wall_time
            })
        
        # 답변 저장
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
    
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