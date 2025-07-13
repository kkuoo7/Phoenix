# HASS Collapse 분석 시스템 상세 분석

## 개요

이 문서는 HASS (Hierarchical Attention with Selective Sampling) 프로젝트의 collapse 분석 시스템에 
대한 상세한 분석을 제공합니다. 이 시스템은 언어 모델의 collapse 현상을 분석하기 위한 4개의 핵심 모듈로 구성되어 있습니다.

## 시스템 아키텍처

```
HASS/evaluation/
├── collapse_config.py      # 설정 관리
├── collapse_collector.py   # 특성 수집 및 메트릭 계산
├── collapse_analyzer.py    # 분석 결과 처리 및 리포트 생성
└── memory_manager.py       # 메모리 관리 및 점진적 처리
```

## 1. CollapseConfig (collapse_config.py)

### 목적
- collapse 분석을 위한 모든 설정을 중앙 집중식으로 관리
- dataclass를 사용하여 타입 안전성 보장

### 주요 설정 항목

#### 메모리 관리 설정
```python
save_every: int = 1024          # 중간 저장 주기
max_buffer_size: int = 10000     # 최대 버퍼 크기
device: str = "cpu"              # 처리 디바이스
```

#### 수집 설정
```python
min_samples_per_token: int = 2   # 토큰당 최소 샘플 수
max_tokens_to_analyze: Optional[int] = None  # 분석할 최대 토큰 수
```

#### 분석 설정
```python
compute_token_metrics: bool = True      # 토큰별 메트릭 계산
compute_overall_metrics: bool = True    # 전체 메트릭 계산
compute_class_metrics: bool = True      # GNC2, UNC3 계산
```

### 설계 특징
- **타입 안전성**: dataclass 사용으로 컴파일 타임 검증
- **유연성**: Optional 타입으로 선택적 설정 지원
- **확장성**: 새로운 메트릭 추가 시 쉽게 확장 가능

## 2. IncrementalMemoryManager (memory_manager.py)

### 목적
- 대용량 데이터의 효율적인 메모리 관리
- GPU 메모리 최적화를 위한 점진적 처리

### 핵심 기능

#### 메모리 관리 전략
```python
def add_features(self, features, token_ids):
    # GPU에서 배치로 처리 (CPU 이동 지연)
    self.buffer.append((
        features.detach(),  # GPU에 유지
        token_ids.detach()  # GPU에 유지
    ))
```

#### 점진적 저장
```python
def _save_and_clear(self):
    # GPU에서 배치로 통합
    combined_features = torch.cat([f[0] for f in self.buffer], dim=0)
    combined_tokens = torch.cat([f[1] for f in self.buffer], dim=0)
    
    # 한 번에 CPU로 이동 (배치 처리)
    combined_features = combined_features.cpu()
    combined_tokens = combined_tokens.cpu()
```

### 최적화 기법

1. **GPU 메모리 최적화**
   - 배치 단위로 CPU 이동 지연
   - `detach()` 사용으로 그래디언트 메모리 해제

2. **메모리 통합**
   - 버퍼 크기 초과 시 자동 통합
   - 가비지 컬렉션 강제 실행

3. **에러 처리**
   - 예외 상황에서도 메모리 누수 방지
   - 안전한 메모리 정리

## 3. CollapseCollector (collapse_collector.py)

### 목적
- HASS 모델 구조에 특화된 특성 수집
- collapse 메트릭 (SVD 엔트로피, GNC2, UNC3) 계산

### 핵심 기능

#### HASS 구조 지원
```python
def collect_features_efficient(self, input_ids, attention_mask):
    # 모델 타입에 따라 다른 처리
    if hasattr(self.model, 'base_model') 
        and hasattr(self.model.base_model, 'model'):
        # HASS EaModel 구조
        outputs = self.model.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False  # 메모리 절약
        )
    else:
        # 일반적인 transformers 모델 구조
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
```

#### 정확한 토큰-특성 매칭
```python
def _extract_features_accurate(self, hidden_states, input_ids, attention_mask):
    Y = input_ids[:, 1:]      # 다음 토큰 (타겟)
    X = hidden_states[:, :-1]  # 현재 토큰의 hidden states
    
    # 유효한 토큰만 선택
    valid_mask = attention_mask[:, 1:].bool()
    Y, X = Y[valid_mask], X[valid_mask]
```

### Collapse 메트릭 계산

#### 1. SVD 엔트로피
```python
def _compute_svd_entropy(self, features):
    # 특성 행렬의 SVD 계산
    U, S, V = torch.svd(features)
    # 정규화된 특이값
    S_norm = S / S.sum()
    # 엔트로피 계산
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
```

#### 2. GNC2 (Geometric Neural Collapse 2)
```python
def _compute_gnc2_class_means(self, token_features):
    # 각 토큰 클래스의 평균 벡터 계산
    class_means = []
    for token_id, features in token_features.items():
        class_mean = features.mean(dim=0)
        class_means.append(class_mean)
    
    # 정규화된 클래스 평균 간 거리 계산
    normed = centered / (centered.norm(dim=1, keepdim=True) + 1e-8)
    values.append(torch.log(1.0 / dist))
```

#### 3. UNC3 (Unified Neural Collapse 3)
```python
def _compute_unc3_duality(self, token_features):
    # classifier와 클래스 평균 벡터의 cosine similarity CoV
    cos_sim = (classifier_norm * means_norm).sum(dim=1)
    
    # Coefficient of Variation (CoV) 계산
    mean_sim = cos_sim.mean()
    std_sim = cos_sim.std()
    return (std_sim / mean_sim).item()
```

### 스페셜 토큰 처리
```python
def _group_features_by_token(self, features, token_ids):
    special_token_ids = set()
    if hasattr(self.tokenizer, 'eos_token_id'):
        special_token_ids.add(self.tokenizer.eos_token_id)
    if hasattr(self.tokenizer, 'pad_token_id'):
        special_token_ids.add(self.tokenizer.pad_token_id)
    
    for i, token_id in enumerate(token_ids):
        if token_id in special_token_ids:
            continue  # 스페셜 토큰은 제외
```

## 4. CollapseAnalyzer (collapse_analyzer.py)

### 목적
- 수집된 메트릭 데이터의 분석 및 리포트 생성
- JSON 형태로 결과 저장

### 분석 구조

#### 1. 전체 요약 통계
```python
def _compute_summary(self, metrics_data):
    summary = {
        "total_tokens_analyzed": total_tokens,
        "tokens_with_metrics": len(token_metrics),
        "avg_svd_entropy": np.mean(svd_entropies),
        "std_svd_entropy": np.std(svd_entropies),
        "avg_samples_per_token": np.mean(sample_counts),
        "min_samples": min(sample_counts),
        "max_samples": max(sample_counts)
    }
```

#### 2. 토큰별 분석
```python
def _analyze_token_metrics(self, metrics_data):
    token_analysis = {}
    for token_id, metrics in token_metrics.items():
        token_analysis[token_id] = {
            "svd_entropy": metrics['svd_entropy'],
            "num_samples": metrics['num_samples']
        }
```

#### 3. 전체 메트릭 분석
```python
def _analyze_overall_metrics(self, metrics_data):
    overall_analysis = {
        "overall_svd_entropy": svd_entropy,
        "overall_gnc2": gnc2,
        "overall_unc3": unc3
    }
```

### 결과 저장
```python
def save_analysis(self, analysis, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
```

## 시스템 통합 및 워크플로우

### 1. 초기화 단계
```python
config = CollapseConfig()
collector = CollapseCollector(model, tokenizer, config)
analyzer = CollapseAnalyzer(config)
```

### 2. 데이터 수집 단계
```python
# 배치별 특성 수집
features, token_ids = 
    collector.collect_features_efficient(input_ids, attention_mask)
collector.add_batch_features(features, token_ids)
```

### 3. 메트릭 계산 단계
```python
# collapse 메트릭 계산
metrics = collector.get_collapse_metrics()
```

### 4. 분석 및 리포트 생성
```python
# 분석 수행
analysis = analyzer.analyze_collapse_metrics(metrics)
analyzer.generate_report(metrics, "collapse_report.json")
```

## 성능 최적화 기법

### 1. 메모리 최적화
- **GPU 배치 처리**: CPU 이동을 최소화
- **점진적 저장**: 메모리 사용량 제한
- **자동 정리**: 가비지 컬렉션 최적화

### 2. 계산 최적화
- **torch.no_grad()**: 그래디언트 계산 비활성화
- **배치 처리**: 텐서 연산 최적화
- **조건부 계산**: 필요시에만 메트릭 계산

### 3. 에러 처리
- **예외 안전성**: 각 단계별 예외 처리
- **로깅**: 상세한 디버깅 정보 제공
- **복구 메커니즘**: 부분 실패 시에도 계속 진행

## 확장성 및 유지보수성

### 1. 모듈화 설계
- 각 모듈의 명확한 책임 분리
- 인터페이스 기반 설계로 확장 용이

### 2. 설정 기반 설계
- dataclass를 통한 중앙 집중식 설정 관리
- 런타임 설정 변경 가능

### 3. 로깅 시스템
- 각 모듈별 상세한 로깅
- 디버깅 및 모니터링 지원

## 결론

이 collapse 분석 시스템은 다음과 같은 특징을 가지고 있습니다:

1. **효율성**: GPU 메모리 최적화와 점진적 처리를 통한 대용량 데이터 처리
2. **정확성**: HASS 모델 구조에 특화된 정확한 특성 추출
3. **확장성**: 모듈화된 설계로 새로운 메트릭 추가 용이
4. **안정성**: 포괄적인 에러 처리와 로깅 시스템

이 시스템은 언어 모델의 collapse 현상을 정량적으로 분석하여 
모델의 품질과 성능을 평가하는 데 중요한 역할을 합니다. 