"""Generate answers with EA model and measure collapse metrics for HELMET/LongProc datasets.

Usage:
python3 gen_ea_answer_helmet.py --base-model-path meta-llama/Llama-3.1-8B-Instruct --ea-model-path path/to/ea_model --model-id llama3-ea --bench-name html_to_tsv_0.5k --test-file ... --answer-file ... --collapse-file ... --max-new-token 512
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

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
hass_dir = os.path.dirname(current_dir)
helmet_dir = os.path.join(hass_dir, 'HELMET')
if helmet_dir not in sys.path:
    sys.path.insert(0, helmet_dir)
if hass_dir not in sys.path:
    sys.path.insert(0, hass_dir)

# HELMET 데이터 로딩만 사용
try:
    from data import load_data
except ImportError as e:
    print(f"HELMET data module import failed: {e}")
    print("Please check if HELMET dependencies are installed")
    sys.exit(1)

from collapse_collector import CollapseCollector
from collapse_config import CollapseConfig
from collapse_analyzer import CollapseAnalyzer

from model.ea_model import EaModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_collapse_config(args) -> CollapseConfig:
    return CollapseConfig(
        save_every=1024,
        max_buffer_size=10000,
        device=args.device if hasattr(args, 'device') else "cpu",
        min_samples_per_token=1
    )

@torch.inference_mode()
def get_model_answers(
    base_model_path,
    ea_model_path,
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
    logger.info(f"Loading EA model from {base_model_path}, {ea_model_path}")
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=getattr(args, 'total_token', 256),
        depth=getattr(args, 'depth', 32),
        top_k=getattr(args, 'top_k', 40),
        threshold=getattr(args, 'threshold', 0.1),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    data_args = argparse.Namespace()
    data_args.max_test_samples = args.max_test_samples if hasattr(args, 'max_test_samples') else None
    data_args.shots = args.shots if hasattr(args, 'shots') else 0
    data_args.seed = args.seed if hasattr(args, 'seed') else 42
    longproc_prefixes = [
        'html_to_tsv', 'pseudo_to_code', 'path_traversal', 'tom_tracking', 'countdown', 'travel_planning'
    ]
    is_longproc = any([bench_name.startswith(prefix) for prefix in longproc_prefixes])
    if is_longproc:
        with open(test_file, 'r') as f:
            samples = json.load(f)
        if data_args.max_test_samples is not None:
            samples = samples[:data_args.max_test_samples]
        import yaml
        prompt_dir = os.path.dirname(test_file)
        with open(os.path.join(prompt_dir, 'prompts.yaml'), 'r') as f:
            user_prompt = yaml.safe_load(f)['USER_PROMPT']
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
            for d in samples:
                pass
        elif bench_name.startswith('travel_planning'):
            for d in samples:
                pass
    else:
        dataset = load_data(data_args, bench_name, path=test_file, demo_path=demo_file)
        samples = dataset["data"]
        prompt_template = dataset["prompt_template"]
    collapse_config = create_collapse_config(args)
    logger.info(f"총 {len(samples)}개 샘플 평가 시작 (샘플별+청크별 collapse)")
    all_sample_collapse = []
    for idx, sample in enumerate(tqdm(samples)):
        try:
            if is_longproc:
                prompt = user_prompt.format(**sample)
            else:
                prompt = prompt_template.format(**sample)
            inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            prompt_len = input_ids.shape[1]
            # 답변 생성 (EA 모델 naivegenerate)
            generated = model.naivegenerate(input_ids, temperature=temperature, log=False, is_llama3=True)
            full_ids = generated[0]  # (batch, prompt+gen_len)
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
            # 생성 구간만 forward (output_hidden_states=True)
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
            if not hidden_states.is_cuda or hidden_states.dtype != torch.float32:
                hidden_states = hidden_states.to(dtype=torch.float32, device=model.device)
            sample_collector = CollapseCollector(model, tokenizer, collapse_config)
            features, token_ids = sample_collector._extract_features_accurate(
                hidden_states, gen_input_ids, gen_attention_mask
            )
            if features is not None and (not features.is_cuda or features.dtype != torch.float32):
                features = features.to(dtype=torch.float32, device=model.device)
            if features is not None and token_ids is not None:
                sample_collector.add_batch_features(features, token_ids)
                sample_metrics = sample_collector.get_collapse_metrics(input_len=0, num_chunks=5)
                all_sample_collapse.append({
                    'sample_id': idx,
                    'chunk_svd_entropies': sample_metrics['chunk_svd_entropies'],
                    'total_generated_tokens': sample_metrics['total_generated_tokens']
                })
                sample_collector.clear()
            else:
                logger.warning(f"샘플 {idx}: features 또는 token_ids가 None")
                all_sample_collapse.append({
                    'sample_id': idx,
                    'chunk_svd_entropies': [None]*5,
                    'total_generated_tokens': 0
                })
                continue
        except Exception as e:
            logger.error(f"샘플 {idx} 처리 중 오류: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            all_sample_collapse.append({
                'sample_id': idx,
                'chunk_svd_entropies': [None]*5,
                'total_generated_tokens': 0
            })
            continue
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
    try:
        os.makedirs(os.path.dirname(collapse_file), exist_ok=True)
        with open(collapse_file, "w") as fout:
            json.dump({'per_sample': all_sample_collapse, 'summary': summary}, fout, indent=2)
        logger.info(f"Sample-wise collapse metrics and summary saved to {collapse_file}")
    except Exception as e:
        logger.error(f"Failed to save sample-wise collapse metrics: {e}")
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path", type=str, required=True)
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
    parser.add_argument("--total-token", type=int, default=256)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()
    setup_seed(args.seed)
    get_model_answers(
        args.base_model_path,
        args.ea_model_path,
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