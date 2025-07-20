"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from accelerate.utils import set_seed
set_seed(0)

import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

from Phoenix.model.ea_model import EaModel
from Phoenix.model.kv_cache import initialize_past_key_values
from Phoenix.model.utils import *


import random
import torch
import numpy as np

from Phoenix.evaluation.collapse_collector import CollapseCollector
from Phoenix.evaluation.svd_collapse_analyzer import SVDCollapseAnalyzer

# LongProc 데이터 로더 임포트 
try: 
    from Phoenix.HELMET.longproc_addon.longproc.longproc.longproc_data import load_longproc_data 
except ImportError: 
    print("LongProc data module import failed: ")
    exit(1)

import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 스크립트 경로 설정 
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def load_benchmark_data(bench_name: str, data_path, question_begin, question_end): 
    """
    벤치마크 이름에 따라 적절한 데이터 로더를 호출하고, 
    'get_model_answers'가 사용할 수 있는 표준 형식으로 데이터를 반환한다. 
    """ 
    if bench_name.startswith('longproc'): 
        task_name = bench_name.split(':')[1]
        logger.info(f"LongProc 데이터셋 로딩 중: {task_name} from {data_path}")

        # LongProc 데이터 로더 호출 (평가 함수는 사용하지 않음)
        task_data, _ = load_longproc_data(dataset_name=task_name, path=data_path)

        # 표준 형식으로 변환 
        questions = [] 
        for i, item in enumerate(task_data): 
            questions.append({
                "question_id": f"{task_name}_{i}",
                # LongProc은 보통 단일 턴으로 구성
                "turns": [item["input_prompt"]],
            })
        
        # question_begin/end 인덱싱 적용 
        if question_begin is not None and question_end is not None: 
            return questions[question_begin:question_end]
    
        return questions

    else:
        raise ValueError(f"지원하지 않는 벤치마크 이름입니다: {bench_name}")



def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        bench_name,
        data_path, # LongProc 데이터 폴더 경로
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
        model_max_length,
        args
):
    questions = load_benchmark_data(bench_name, data_path, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                collapse_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                model_max_length,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


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
        model_max_length,
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    collector = CollapseCollector()
    analyzer = SVDCollapseAnalyzer(collector)
    all_turn_collapse_metrics = []

    question = questions[0]

    # LongProc 벤치마크에서는 Warmup 짧게 수행
    for _ in range(1):
        torch.manual_seed(0)

        # messages = [
        #     {"role": "system", "content": "You are a highly specialized agent tasked with executing complex, multi-step procedures based on a long context. Your primary goal is to follow the given instructions precisely and generate a complete, structured, long-form output. Key directives: 1.  **Follow Instructions Meticulously**: Adhere strictly to every step and constraint outlined in the user's request. Do not summarize or omit any required steps. 2.  **Synthesize All Information**: Your response must be based on a thorough synthesis of all relevant information dispersed throughout the provided context. 3.  **Generate a Complete and Structured Output**: Produce a long-form, well-structured response in the exact format specified. Ensure the output is fully completed before you stop. Do not end prematurely. 4.  **Reason Step-by-Step**: Think through the procedure logically to ensure accuracy and coherence from beginning to end."},
        # ]

        messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]

        turns = []
        idxs = []
        new_tokens = []
        wall_time = []

        user_content = question["turns"][0]
        messages.append({"role": "user", "content": user_content})

        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = tokenizer([prompt], add_special_tokens=False).input_ids 

        # try:
        torch.cuda.synchronize()
        start_time = time.time()

        output_ids, new_token, idx = model.naivegenerate(
            torch.as_tensor(input_ids).cuda(),
            max_new_tokens=max_new_token,
            temperature=temperature,
            max_length=model_max_length,
            log=True,
            is_llama3=True,
            hidden_state_collector=collector,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        output_ids = output_ids[0][len(input_ids[0]):]
        # be consistent with the template's stop_token_ids
        stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        if stop_token_ids:
            stop_token_ids_index = [
                i
                for i, id in enumerate(output_ids)
                if id in stop_token_ids
            ]
            if len(stop_token_ids_index) > 0:
                output_ids = output_ids[: stop_token_ids_index[0]]

        output = tokenizer.decode(
            output_ids,
            spaces_between_special_tokens=False,
        )
        # stop_str = "</s>"
        # if stop_str and output.find(stop_str) > 0:
        #     output = output[: output.find(stop_str)]
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
        messages.append({
            "role": "assistant",
            "content": output
        })

    print('Warmup done')

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)

            # messages = [
            #     {"role": "system",
            #     "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            # ]

            messages = [
                {"role": "system", "content": "You are a highly specialized agent tasked with executing complex, multi-step procedures based on a long context. Your primary goal is to follow the given instructions precisely and generate a complete, structured, long-form output. Key directives: 1.  **Follow Instructions Meticulously**: Adhere strictly to every step and constraint outlined in the user's request. Do not summarize or omit any required steps. 2.  **Synthesize All Information**: Your response must be based on a thorough synthesis of all relevant information dispersed throughout the provided context. 3.  **Generate a Complete and Structured Output**: Produce a long-form, well-structured response in the exact format specified. Ensure the output is fully completed before you stop. Do not end prematurely. 4.  **Reason Step-by-Step**: Think through the procedure logically to ensure accuracy and coherence from beginning to end."},
            ]

            turns = []
            idxs = []
            new_tokens = []
            wall_time = []

            # [수정] 샘플 단위로 collector clear
            collector.clear()
            for j, qs in enumerate(question["turns"]):

                messages.append({"role": "user", "content": qs})

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                input_ids = tokenizer([prompt], add_special_tokens=False, ).input_ids

                torch.cuda.synchronize()
                start_time = time.time()

                output_ids, new_token, idx = model.naivegenerate(
                    torch.as_tensor(input_ids).cuda(),
                    max_new_tokens=max_new_token,
                    temperature=temperature,
                    max_length=model_max_length,
                    log=True,
                    is_llama3=True,
                    hidden_state_collector=collector # collector에 계속 누적
                )

                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
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

                messages.append({
                    "role": "assistant",
                    "content": output
                })
                # [DEBUG] 턴별 피처 수집/생성 토큰 수 확인
                collected_features = len(collector.get_hidden_states_by_state('baseline_accepted'))
                print(f"[DEBUG] question_id={question['question_id']}, turn={j+1}, new_token={new_token}, collected_features={collected_features}")
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # [수정] 샘플 전체에 대해 collapse metric 계산
        all_features = collector.get_hidden_states_by_state('baseline_accepted')
        print(f"[DEBUG] sample question_id={question['question_id']}, total_feature_len={len(all_features)}, total_generated_tokens={sum(new_tokens)}")
        
        if len(all_features) > 0:
            all_features = torch.cat(all_features, dim=0)
            print(f"[DEBUG] sample concatenated_features_shape={all_features.shape}")
            
            metrics = analyzer.get_collapse_metrics_fixed_chunk(all_features, chunk_size=128)
            
            print(f"[DEBUG] sample metrics={metrics}")
            metrics['question_id'] = question['question_id']
            all_turn_collapse_metrics.append(metrics)

        # Dump answers
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

    if collapse_file and all_turn_collapse_metrics:
            summary = {}
            
            # --- 1. 모든 샘플로부터 통계치 수집 ---
            entropies_by_chunk_idx = {}
            all_avg_entropies = []
            all_cv_entropies = []
            all_slope_entropies = []

            for metrics in all_turn_collapse_metrics:
                # 청크별 엔트로피 수집
                if 'fixed_chunk_svd_entropies' in metrics:
                    for i, entropy in enumerate(metrics['fixed_chunk_svd_entropies']):
                        if entropy is not None:
                            if i not in entropies_by_chunk_idx:
                                entropies_by_chunk_idx[i] = []
                            entropies_by_chunk_idx[i].append(entropy)

                # 샘플별 종합 통계치 수집
                if metrics.get('avg_svd_entropy') is not None:
                    all_avg_entropies.append(metrics['avg_svd_entropy'])
                if metrics.get('cv_svd_entropy') is not None:
                    all_cv_entropies.append(metrics['cv_svd_entropy'])
                if metrics.get('slope_svd_entropy') is not None:
                    all_slope_entropies.append(metrics['slope_svd_entropy'])

            # --- 2. 종합 통계치 계산 ---
            summary['total_analyzed_samples'] = len(all_turn_collapse_metrics)
            summary['overall_avg_svd_entropy'] = float(np.mean(all_avg_entropies)) if all_avg_entropies else None
            summary['overall_avg_cv_svd_entropy'] = float(np.mean(all_cv_entropies)) if all_cv_entropies else None
            summary['overall_avg_slope_svd_entropy'] = float(np.mean(all_slope_entropies)) if all_slope_entropies else None

            # 청크별 평균 엔트로피 계산
            chunk_summary = {}
            for i, chunk_entropies in sorted(entropies_by_chunk_idx.items()):
                avg_entropy = np.mean(chunk_entropies) if chunk_entropies else None
                chunk_summary[f'avg_chunk_idx_{i}_svd_entropy'] = float(avg_entropy) if avg_entropy is not None else None
            
            summary.update(chunk_summary)

            # --- 3. 터미널에 상세 리포트 출력 ---
            print("\n" + "="*60)
            print(" " * 15 + "Collapse Analysis Summary")
            print("="*60)
            print(f"Total Analyzed Samples: {summary['total_analyzed_samples']}")
            print("-" * 60)
            print("📊 Overall Sample Statistics:")
            print(f"  - Average SVD Entropy      : {summary['overall_avg_svd_entropy']:.4f}")
            print(f"  - Average CV of Entropy    : {summary['overall_avg_cv_svd_entropy']:.4f}")
            print(f"  - Average Trend Slope      : {summary['overall_avg_slope_svd_entropy']:.4f}")
            print("-" * 60)
            print("📈 Chunk-wise Average SVD Entropy:")
            
            # 청크별 엔트로피를 표 형식으로 출력
            chunk_keys = sorted([key for key in summary.keys() if 'avg_chunk_idx' in key], key=lambda x: int(x.split('_')[3]))
            for key in chunk_keys:
                chunk_index = key.split('_')[3]
                value = summary[key]
                print(f"  - Chunk {int(chunk_index):<2}: {value:.4f}")
            print("="*60 + "\n")

            # --- 4. JSON 파일로 리포트 저장 ---
            report = {
                'per_sample_metrics': all_turn_collapse_metrics,
                'summary': summary
            }
            os.makedirs(os.path.dirname(collapse_file), exist_ok=True)
            with open(collapse_file, "w") as fout:
                json.dump(report, fout, indent=2)
            logger.info(f"Collapse 분석 결과를 {collapse_file}에 저장했습니다.")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/70B/",
                        help="1")

    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )

    parser.add_argument("--model-id", type=str, default="llama38b2_40")

    parser.add_argument("--data-path", type=str, default="data", help="LongProc 또는 다른 데이터셋의 폴더 경로")

    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )

    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )

    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--collapse-file", 
        type=str, 
        default=None,
        help="Collapse 분석 결과를 저장할 파일 경로"
    )

    parser.add_argument(
        "--model-max-length",
        type=int,
        default=4096,
        help="The maximum sequence length the model can handle."
    )   

    args = parser.parse_args()
    setup_seed(args.seed)

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    # collapse_file 설정
    if args.collapse_file:
        collapse_file = args.collapse_file
    else:
        collapse_file = f"{args.bench_name}/{args.model_id}_collapse.json"

    print(f"Output to {answer_file}")
    print(f"Collapse analysis to {collapse_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        args.bench_name,
        args.data_path,
        args.question_begin,
        args.question_end,
        args.answer_file,
        args.collapse_file,  # collapse_file 파라미터 추가
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.model_max_length,
        args
    )

    reorg_answer_file(answer_file)
