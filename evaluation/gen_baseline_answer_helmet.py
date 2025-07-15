"""Generate answers with baseline model and measure collapse metrics for HELMET/LongProc datasets.

Usage:
python3 gen_baseline_answer_helmet.py --base-model-path meta-llama/Llama-2-7b-hf --model-id llama2-7b --bench-name json_kv --test-file HASS/HELMET/data/json_kv/test.json --demo-file HASS/HELMET/data/json_kv/demo.json --answer-file results/helmet_json_kv_answers.jsonl --collapse-file results/helmet_json_kv_collapse.json
"""
import argparse
import json
import os
import sys
import logging
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


# HELMET 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
hass_dir = os.path.dirname(current_dir)
helmet_dir = os.path.join(hass_dir, 'HELMET')
if helmet_dir not in sys.path:
    sys.path.insert(0, helmet_dir)

# HASS 모듈 경로 추가
if hass_dir not in sys.path:
    sys.path.insert(0, hass_dir)

# HELMET 데이터 로딩만 사용
try:
    from data import load_data
except ImportError as e:
    print(f"HELMET data module import failed: {e}")
    print("Please check if HELMET dependencies are installed")
    sys.exit(1)

# HASS collapse 모듈들 import
from collapse_collector import CollapseCollector
from collapse_config import CollapseConfig
from collapse_analyzer import CollapseAnalyzer

logging.basicConfig(level=logging.DEBUG)  # INFO에서 DEBUG로 변경
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_collapse_config(args) -> CollapseConfig:
    """
    CollapseConfig 객체 생성 (SVD entropy만 측정)
    """
    return CollapseConfig(
        save_every=1024,
        max_buffer_size=10000,
        device=args.device if hasattr(args, 'device') else "cpu",
        min_samples_per_token=1  # 2에서 1로 변경하여 제약 완화
    )

@torch.inference_mode()
def get_model_answers(
    base_model_path,
    model_id,
    bench_name,
    test_file,
    demo_file,
    answer_file,
    collapse_file,
    max_new_token,
    num_choices,
    temperature,
    args
):
    print("[DEBUG] get_model_answers 진입")
    logger.info(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("[DEBUG] 모델 로딩 완료")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"데이터 로딩 시작: bench_name={bench_name}, test_file={test_file}, demo_file={demo_file}")
    data_args = argparse.Namespace()
    data_args.max_test_samples = args.max_test_samples if hasattr(args, 'max_test_samples') else None
    data_args.shots = args.shots if hasattr(args, 'shots') else 0
    data_args.seed = args.seed if hasattr(args, 'seed') else 42
    logger.info(f"data_args: max_test_samples={data_args.max_test_samples}, shots={data_args.shots}, seed={data_args.seed}")

    # longproc 계열 데이터셋 분기
    longproc_prefixes = [
        'html_to_tsv', 'pseudo_to_code', 'path_traversal', 'tom_tracking', 'countdown', 'travel_planning'
    ]
    is_longproc = any([bench_name.startswith(prefix) for prefix in longproc_prefixes])
    if is_longproc:
        print(f"[DEBUG] longproc 계열 데이터셋 감지: {bench_name}")
        with open(test_file, 'r') as f:
            samples = json.load(f)
        # max_test_samples 적용 (longproc 계열)
        if data_args.max_test_samples is not None:
            samples = samples[:data_args.max_test_samples]
        logger.info(f"longproc 데이터 로딩 완료: 샘플 수={len(samples)}")
        # prompts.yaml에서 USER_PROMPT 템플릿 로딩
        import yaml
        prompt_dir = os.path.dirname(test_file)
        with open(os.path.join(prompt_dir, 'prompts.yaml'), 'r') as f:
            user_prompt = yaml.safe_load(f)['USER_PROMPT']
        # 태스크별 필드 가공 (longproc_data.py 참고)
        if bench_name.startswith('html_to_tsv'):
            for d in samples:
                html_path = d['html_path']
                html_file_path = os.path.join(prompt_dir, html_path)
                with open(html_file_path, 'r') as html_file:
                    d['html_str'] = html_file.read()
        elif bench_name.startswith('pseudo_to_code'):
            for d in samples:
                d['pseudocode'] = '\n'.join(d['pseudocode_lines'])
                d['reference_output'] = '\n'.join(d['code_lines'])
        elif bench_name.startswith('path_traversal'):
            for d in samples:
                d['city_context'] = d['context_nl']
                d['src_city'] = d['question_repr'][0]
                d['dst_city'] = d['question_repr'][1]
                d['reference_output'] = d['answer_nl']
        elif bench_name.startswith('tom_tracking'):
            for d in samples:
                d['reference_output'] = d['solution']
        elif bench_name.startswith('countdown'):
            # demonstration 등 추가 가공 필요 (간단화)
            for d in samples:
                pass  # 실제 실험에서는 longproc_data.py의 build_countdown_demonstration 등 활용 필요
        elif bench_name.startswith('travel_planning'):
            for d in samples:
                pass  # 실제 실험에서는 build_icl_demonstration 등 활용 필요
    else:
        try:
            dataset = load_data(data_args, bench_name, path=test_file, demo_path=demo_file)
            logger.info(f"데이터 로딩 완료: dataset keys={list(dataset.keys())}")
            samples = dataset["data"]
            logger.info(f"샘플 수: {len(samples)}")
            if len(samples) > 0:
                logger.info(f"첫 번째 샘플 예시: {samples[0]}")
                logger.info(f"prompt_template: {dataset.get('prompt_template', 'Not found')}")
            prompt_template = dataset["prompt_template"]
        except Exception as e:
            logger.error(f"데이터 로딩 중 오류: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"[DEBUG] 데이터 로딩 중 오류: {e}")
            raise
    collapse_config = create_collapse_config(args)
    logger.info(f"총 {len(samples)}개 샘플 평가 시작 (샘플별+청크별 collapse)")
    print(f"[DEBUG] 총 {len(samples)}개 샘플 평가 시작")
    all_sample_collapse = []
    total_features_collected = 0
    total_tokens_collected = 0
    for idx, sample in enumerate(tqdm(samples)):
        print(f"[DEBUG] 샘플 {idx} 처리 시작")
        try:
            # longproc robust 분기: prompts.yaml 템플릿 적용 + 필드 체크
            if is_longproc:
                prompt = user_prompt.format(**sample)
            else:
                prompt = prompt_template.format(**sample)
            print(f"[DEBUG] 샘플 {idx} prompt: {prompt}")
            # 표준 토크나이저 사용
            inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            prompt_len = input_ids.shape[1]
            print(f"[DEBUG] 샘플 {idx} input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}, prompt_len: {prompt_len}")
            torch.cuda.empty_cache()
            # 1. 텍스트 생성 (prompt -> 생성)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_token,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                    output_hidden_states=True,
                )
                # Huggingface generate는 전체 시퀀스의 hidden_states를 반환하지 않으므로,
                # prompt를 past_key_values로 처리한 뒤, 생성 구간만 따로 forward
                full_ids = generated.sequences  # (batch, prompt+gen_len)
                gen_len = full_ids.shape[1] - prompt_len
                if gen_len == 0:
                    logger.warning(f"샘플 {idx}: 생성 토큰 없음")
                    all_sample_collapse.append({
                        'sample_id': idx,
                        'chunk_svd_entropies': [None]*5,
                        'total_generated_tokens': 0
                    })
                    continue
                # past_key_values 얻기 위해 prompt만 forward
                prompt_outputs = model(
                    input_ids=input_ids,
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
                print(f"[DEBUG] 샘플 {idx} hidden_states shape: {hidden_states.shape}")
                # CollapseCollector를 사용해 생성 토큰 구간의 hidden states만 추출
                sample_collector = CollapseCollector(model, tokenizer, collapse_config)
                features, token_ids = sample_collector._extract_features_accurate(
                    hidden_states, gen_input_ids, gen_attention_mask
                )
                # features도 안전장치
                if features is not None and (not features.is_cuda or features.dtype != torch.float32):
                    features = features.to(dtype=torch.float32, device=model.device)
            print(f"[DEBUG] 샘플 {idx} features is None? {features is None}, token_ids is None? {token_ids is None}")
            if features is not None and token_ids is not None:
                print(f"[DEBUG] 샘플 {idx} features shape: {features.shape}, token_ids shape: {token_ids.shape}")
                sample_collector.add_batch_features(features, token_ids)
                total_features_collected += features.shape[0] if hasattr(features, 'shape') else len(features)
                total_tokens_collected += token_ids.shape[0] if hasattr(token_ids, 'shape') else len(token_ids)
            else:
                logger.warning(f"샘플 {idx}: features 또는 token_ids가 None")
                print(f"[DEBUG] 샘플 {idx}: features 또는 token_ids가 None")
                all_sample_collapse.append({
                    'sample_id': idx,
                    'chunk_svd_entropies': [None]*5,
                    'total_generated_tokens': 0
                })
                continue
            # 샘플별 collapse 측정 (생성 토큰만, 5등분 청크)
            sample_metrics = sample_collector.get_collapse_metrics(input_len=prompt_len, num_chunks=5)
            print(f"[DEBUG] 샘플 {idx} sample_metrics: {sample_metrics}")
            all_sample_collapse.append({
                'sample_id': idx,
                'chunk_svd_entropies': sample_metrics['chunk_svd_entropies'],
                'total_generated_tokens': sample_metrics['total_generated_tokens']
            })
            sample_collector.clear()
        except Exception as e:
            logger.error(f"샘플 {idx} 처리 중 오류: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"[DEBUG] 샘플 {idx} 처리 중 오류: {e}")
            continue
    print(f"[DEBUG] 전체 샘플 처리 완료, 결과 저장 시작")
    # summary: 각 청크별 평균 SVD entropy
    chunk_lists = [s['chunk_svd_entropies'] for s in all_sample_collapse]
    num_chunks = 5
    avg_chunk_svd_entropies = []
    for i in range(num_chunks):
        chunk_vals = [chunks[i] for chunks in chunk_lists if chunks[i] is not None]
        avg = float(np.mean(chunk_vals)) if chunk_vals else None
        avg_chunk_svd_entropies.append(avg)
    summary = {f'avg_chunk{i+1}_svd_entropy': avg_chunk_svd_entropies[i] for i in range(num_chunks)}
    summary['num_samples'] = len(all_sample_collapse)
    # 샘플별 collapse 결과 + summary 저장
    try:
        os.makedirs(os.path.dirname(collapse_file), exist_ok=True)
        with open(collapse_file, "w") as fout:
            json.dump({'per_sample': all_sample_collapse, 'summary': summary}, fout, indent=2)
        logger.info(f"Sample-wise collapse metrics and summary saved to {collapse_file}")
        print(f"[DEBUG] 결과 저장 완료: {collapse_file}")
    except Exception as e:
        logger.error(f"Failed to save sample-wise collapse metrics: {e}")
        print(f"[DEBUG] 결과 저장 실패: {e}")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("[DEBUG] MAIN 진입")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--base-model-path", type=str, required=True)
        parser.add_argument("--model-id", type=str, required=True)
        parser.add_argument("--bench-name", type=str, required=True)
        parser.add_argument("--test-file", type=str, required=True)
        parser.add_argument("--demo-file", type=str, default=None)
        parser.add_argument("--answer-file", type=str, required=True)
        parser.add_argument("--collapse-file", type=str, required=True)
        parser.add_argument("--max-new-token", type=int, default=512)
        parser.add_argument("--num-choices", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--max-test-samples", type=int, default=None)
        parser.add_argument("--shots", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        logger.info("=== Starting HELMET Collapse Evaluation ===")
        print("[DEBUG] HELMET Collapse Evaluation 시작")
        logger.info(f"Model: {args.base_model_path}")
        logger.info(f"Dataset: {args.bench_name}")
        logger.info(f"Test file: {args.test_file}")
        logger.info(f"Max samples: {args.max_test_samples}")
        print(f"[DEBUG] Model: {args.base_model_path}, Dataset: {args.bench_name}, Test file: {args.test_file}, Max samples: {args.max_test_samples}")
        setup_seed(args.seed)
        print("[DEBUG] Seed 설정 완료")
        get_model_answers(
            args.base_model_path,
            args.model_id,
            args.bench_name,
            args.test_file,
            args.demo_file,
            args.answer_file,
            args.collapse_file,
            args.max_new_token,
            args.num_choices,
            args.temperature,
            args
        )
        logger.info("=== Evaluation completed successfully ===")
        print("[DEBUG] Evaluation completed successfully")
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        print("[DEBUG] Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"[DEBUG] Unexpected error occurred: {e}")
        sys.exit(1) 