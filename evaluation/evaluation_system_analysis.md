# HASS 평가 시스템 상세 분석

## 개요

이 문서는 HASS (Hierarchical Attention with Selective Sampling) 프로젝트의 평가 시스템에 대한 상세한 분석을 제공합니다. 이 시스템은 EA 모델과 베이스라인 모델의 성능을 비교 평가하고, collapse 메트릭을 측정하는 4개의 핵심 스크립트로 구성되어 있습니다.

## 시스템 아키텍처

```
HASS/evaluation/
├── gen_ea_answer_with_collapse_refactored.py      # EA 모델 평가 (draft 포함)
├── gen_baseline_answer_with_collapse_refactored.py # 베이스라인 모델 평가
├── acceptance_length.py                           # 수용 길이 계산
├── speed.py                                      # 속도 비교 분석
└── collapse_analysis.md                          # Collapse 분석 문서
```

## 1. EA 모델 평가 스크립트 (gen_ea_answer_with_collapse_refactored.py)

### 목적
- EA 모델의 답변 생성 및 collapse 메트릭 측정
- Draft 모델을 포함한 전체 HASS 시스템 평가

### 핵심 기능

#### 1. 모델 초기화
```python
def get_model_answers(...):
    # EA 모델 초기화
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
```

#### 2. Collapse 수집기 통합
```python
# Collapse 수집기 초기화
collapse_config = create_collapse_config(args)
collapse_collector = CollapseCollector(model, tokenizer, collapse_config)

# 특성 수집 및 메트릭 계산
features, token_ids = collapse_collector.collect_features_efficient(
    input_tensor, attention_mask
)
if features is not None:
    collapse_collector.add_batch_features(features, token_ids)
```

#### 3. EA 모델 생성
```python
# EA 모델 생성 (draft 포함)
output_ids, new_token, idx, _ = model.eagenerate(
    input_tensor, temperature=temperature, log=True, is_llama3=True
)
```

### 워크플로우

#### 1. Warmup 단계
```python
logger.info("Starting warmup...")
for _ in range(3):
    # 3번의 워밍업 실행
    # 모델과 collapse 수집기 초기화
    # 메모리 최적화
```

#### 2. 메인 평가 루프
```python
for question in tqdm(questions, desc="Processing questions"):
    for i in range(num_choices):
        for j in range(len(question["turns"])):
            # 1. 특성 수집
            features, token_ids = collapse_collector.collect_features_efficient(...)
            
            # 2. EA 모델 생성
            output_ids, new_token, idx, _ = model.eagenerate(...)
            
            # 3. 결과 저장
            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
```

#### 3. Collapse 메트릭 계산
```python
# Collapse 메트릭 계산 및 저장
collapse_metrics = collapse_collector.get_collapse_metrics()
if collapse_metrics:
    # 기본 메트릭 저장
    with open(collapse_file, "w") as fout:
        json.dump(collapse_metrics, fout, indent=2)
    
    # 추가 분석 리포트 생성
    analyzer = CollapseAnalyzer(collapse_config)
    analysis = analyzer.generate_report(collapse_metrics, analysis_file)
```

### 성능 측정

#### 시간 측정
```python
# 시간 측정
torch.cuda.synchronize()
start_time = time.time()

# EA 모델 생성
output_ids, new_token, idx, _ = model.eagenerate(...)

torch.cuda.synchronize()
total_time = time.time() - start_time
```

#### 메모리 관리
```python
# 메모리 정리
collapse_collector.clear()
del model
torch.cuda.empty_cache()
```

## 2. 베이스라인 모델 평가 스크립트 (gen_baseline_answer_with_collapse_refactored.py)

### 목적
- 베이스라인 모델의 답변 생성 및 collapse 메트릭 측정
- EA 모델과의 성능 비교를 위한 기준점 제공

### 핵심 차이점

#### 1. 모델 선택 로직
```python
# Baseline 모델만 사용하는 경우
if ea_model_path is None or ea_model_path == "None":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Base model 직접 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
```

#### 2. Naive Generation 시뮬레이션
```python
def naivegenerate(input_ids, temperature=0.0, log=False, is_llama3=False):
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_token,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            repetition_penalty=1.1,  # 반복 방지
            no_repeat_ngram_size=3   # n-gram 반복 방지
        )
        generated_ids = outputs.sequences
        new_tokens = generated_ids.shape[1] - input_ids.shape[1]
        return generated_ids, new_tokens, 0  # idx는 0으로 설정

model.naivegenerate = naivegenerate
```

#### 3. 베이스라인 특성 수집
```python
# Baseline 모델: base model의 피처 벡터만 수집
features, token_ids = collapse_collector.collect_features_efficient(
    input_tensor, attention_mask
)
if features is not None:
    collapse_collector.add_batch_features(features, token_ids)
```

### 베이스라인 vs EA 모델 차이점

| 특징 | 베이스라인 모델 | EA 모델 |
|------|----------------|---------|
| 모델 구조 | 단일 베이스 모델 | 베이스 + Draft 모델 |
| 생성 방식 | `model.generate()` | `model.eagenerate()` |
| 특성 수집 | 베이스 모델만 | 베이스 + Draft 모델 |
| 성능 | 기준점 | 최적화된 성능 |

## 3. 수용 길이 계산 스크립트 (acceptance_length.py)

### 목적
- EA 모델의 수용 길이 (acceptance length) 계산
- Draft 모델이 생성한 토큰 중 실제로 수용된 비율 측정

### 핵심 기능

#### 1. 데이터 로딩
```python
f = open(args.input_file, 'r')
lines = f.readlines()
print('num of samples:', len(lines))
```

#### 2. 수용 길이 계산
```python
avg_accept_length = 0

for line in lines:
    data = json.loads(line)
    # 각 샘플의 평균 수용 길이 계산
    avg_accept_length += sum(data['choices'][0]['accept_length']) / len(data['choices'][0]['accept_length']) + 1

# 전체 평균 계산
avg_accept_length /= len(lines)
```

### 수용 길이의 의미

- **수용 길이**: Draft 모델이 생성한 토큰 중 실제로 베이스 모델이 수용한 토큰의 수
- **효율성 지표**: 높은 수용 길이는 Draft 모델의 품질을 나타냄
- **성능 최적화**: 수용 길이를 통해 Draft 모델의 성능을 평가하고 개선

## 4. 속도 비교 분석 스크립트 (speed.py)

### 목적
- EA 모델과 베이스라인 모델의 속도 비교
- Speedup ratio 계산

### 핵심 기능

#### 1. EA 모델 속도 계산
```python
speeds = []
for datapoint in data:
    qid = datapoint["question_id"]
    answer = datapoint["choices"][0]['turns']
    tokens = sum(datapoint["choices"][0]['new_tokens'])  # 실제 생성된 토큰 수
    times = sum(datapoint["choices"][0]['wall_time'])    # 실제 소요 시간
    speeds.append(tokens/times)  # 토큰/초
```

#### 2. 베이스라인 모델 속도 계산
```python
speeds0 = []
for datapoint in data:
    qid = datapoint["question_id"]
    answer = datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        # 토크나이저로 실제 토큰 수 계산
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
```

#### 3. Speedup Ratio 계산
```python
print("speedup ratio:", np.array(speeds).mean() / np.array(speeds0).mean())
```

### 속도 측정의 차이점

| 측정 방식 | EA 모델 | 베이스라인 모델 |
|-----------|---------|----------------|
| 토큰 수 | `new_tokens` (실제 생성) | 토크나이저로 계산 |
| 시간 | `wall_time` (실제 시간) | `wall_time` (실제 시간) |
| 의미 | Draft 모델의 효율성 | 순수 베이스 모델 성능 |

## 시스템 통합 및 평가 워크플로우

### 1. 평가 준비
```bash
# EA 모델 평가
python3 gen_ea_answer_with_collapse_refactored.py \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --ea-model-path path/to/ea_model \
    --bench-name html_to_tsv \
    --question-begin 0 \
    --question-end 2 \
    --max-new-token 128

# 베이스라인 모델 평가
python3 gen_baseline_answer_with_collapse_refactored.py \
    --base-model-path meta-llama/Llama-3.1-8B-Instruct \
    --bench-name html_to_tsv \
    --question-begin 0 \
    --question-end 2 \
    --max-new-token 128
```

### 2. 결과 분석
```bash
# 수용 길이 계산
python3 acceptance_length.py --input_file ea_results.jsonl

# 속도 비교
python3 speed.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --baseline_json baseline_results.jsonl \
    --hass_json ea_results.jsonl
```

### 3. Collapse 메트릭 분석
```python
# Collapse 분석기로 상세 분석
analyzer = CollapseAnalyzer(collapse_config)
analysis = analyzer.generate_report(collapse_metrics, "analysis.json")
```

## 성능 최적화 기법

### 1. 메모리 최적화
- **torch.inference_mode()**: 그래디언트 계산 비활성화
- **low_cpu_mem_usage=True**: CPU 메모리 사용량 최소화
- **device_map="auto"**: 자동 디바이스 매핑

### 2. 계산 최적화
- **Warmup**: 모델 초기화 및 메모리 최적화
- **배치 처리**: 효율적인 특성 수집
- **점진적 저장**: 메모리 사용량 제한

### 3. 에러 처리
- **예외 안전성**: 각 단계별 예외 처리
- **로깅**: 상세한 디버깅 정보 제공
- **복구 메커니즘**: 부분 실패 시에도 계속 진행

## 평가 메트릭

### 1. 성능 메트릭
- **Speedup Ratio**: EA 모델의 속도 향상 비율
- **Acceptance Length**: Draft 모델의 수용률
- **Wall Time**: 실제 소요 시간

### 2. 품질 메트릭
- **Collapse Metrics**: SVD 엔트로피, GNC2, UNC3
- **Token Quality**: 생성된 토큰의 품질
- **Response Quality**: 답변의 품질

### 3. 효율성 메트릭
- **Memory Usage**: 메모리 사용량
- **GPU Utilization**: GPU 활용률
- **Throughput**: 처리량 (토큰/초)

## 확장성 및 유지보수성

### 1. 모듈화 설계
- 각 스크립트의 명확한 책임 분리
- 재사용 가능한 컴포넌트 설계

### 2. 설정 기반 설계
- argparse를 통한 명령행 인수 처리
- 유연한 설정 옵션 제공

### 3. 로깅 시스템
- 각 단계별 상세한 로깅
- 디버깅 및 모니터링 지원

## 결론

이 평가 시스템은 다음과 같은 특징을 가지고 있습니다:

1. **포괄성**: 성능, 품질, 효율성을 모두 평가
2. **정확성**: 정확한 메트릭 계산과 비교 분석
3. **효율성**: 최적화된 메모리 사용과 계산
4. **확장성**: 새로운 모델과 메트릭 추가 용이

이 시스템은 EA 모델의 성능을 정량적으로 평가하여 모델의 개선 방향을 제시하는 데 중요한 역할을 합니다. 