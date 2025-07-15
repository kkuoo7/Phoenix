# SVD 엔트로피 계산 방식 개선

## 개요

이 문서는 HASS 프로젝트의 representation collapse 측정을 위한 SVD 엔트로피 계산 방식의 개선 사항을 설명합니다.

## SVD 엔트로피 계산 방식

### 기본 방식 (공분산 행렬 기반)
```python
def _compute_svd_entropy(self, features):
    # 1. 공분산 행렬 계산
    centered_features = features - features.mean(dim=0, keepdim=True)
    covariance_matrix = torch.matmul(centered_features.T, centered_features) / (n_samples - 1)
    
    # 2. 특이값 분해
    singular_values = torch.linalg.svdvals(covariance_matrix)
    
    # 3. 정규화 및 엔트로피 계산
    normalized_singular_values = singular_values / torch.sum(singular_values)
    entropy = -torch.sum(normalized_singular_values * torch.log2(normalized_singular_values + 1e-8))
    
    return entropy.item()
```

### 고급 분석 방식
```python
def _compute_svd_entropy_advanced(self, features):
    # 기본 공분산 행렬 계산 + 추가 분석
    # - 유효 차원 수 계산
    # - 정규화된 엔트로피 (0~1 범위)
    # - 최대 가능한 엔트로피
    # - 특이값 분포
```

## 주요 개선사항

### 1. 수학적 정확성
- **기존**: 특성 벡터들 간의 직접적인 선형 관계 분석
- **새로운**: 특성들 간의 공분산 구조를 통한 정보량 측정

### 2. 민감도 향상
테스트 결과에서 확인된 개선사항:
- **공분산 행렬 기반**: collapse_level 0.6부터 명확한 감소 (6.53 → 6.50 → 6.33)
- **정확한 collapse 감지**: 완전한 붕괴 상태에서 -0.0000으로 정확한 감지

### 3. 계산 효율성
- **공분산 행렬 기반**: `O(d^3)` - 공분산 행렬 SVD (d << n인 경우 더 효율적)
- **수학적 정확성**: 특성들 간의 공분산 구조를 통한 정보량 측정

## 사용 방법

### 1. 설정 변경
```python
from collapse_config import CollapseConfig

config = CollapseConfig(
    svd_entropy_method="default",  # 기본값 (공분산 행렬 기반)
    use_advanced_analysis=True
)
```

### 2. 사용 가능한 방식들
- `"default"`: 공분산 행렬 기반 (기본값)
- `"advanced"`: 고급 분석 포함

### 3. 고급 분석 기능
```python
# 고급 분석 결과 예시
{
    'entropy': 6.5348,
    'effective_dimensions': 45,
    'max_entropy': 6.6439,
    'normalized_entropy': 0.9836,
    'singular_values': [...]
}
```

## 테스트 결과

### 비교 테스트 실행
```bash
cd HASS/evaluation
python test_svd_entropy_methods.py
```

### 결과 요약
| Collapse Level | Default | Advanced | Eff.Dims | Norm. |
|----------------|---------|----------|----------|-------|
| 0.0            | 6.5348  | 6.5348   | 45       | 0.984 |
| 0.2            | 6.5367  | 6.5367   | 47       | 0.984 |
| 0.4            | 6.5308  | 6.5308   | 45       | 0.983 |
| 0.6            | 6.5050  | 6.5050   | 45       | 0.979 |
| 0.8            | 6.3373  | 6.3373   | 41       | 0.954 |
| 1.0            | -0.0000 | -0.0000  | 1        | -0.000 |

## 이론적 배경

### 공분산 행렬의 의미
공분산 행렬 `C`는 다음과 같이 계산됩니다:

```
C = (1/(n-1)) * H_centered.T @ H_centered
```

여기서:
- `H_centered`: 평균이 0으로 조정된 특성 행렬
- `n`: 샘플 수
- `C`: (hidden_dim, hidden_dim) 형태의 대칭 행렬

### SVD 엔트로피의 해석
- **높은 엔트로피**: 여러 방향으로 정보가 고르게 분포
- **낮은 엔트로피**: 소수의 방향에만 정보가 치우침 (collapse)

### 정규화된 엔트로피
```
normalized_entropy = entropy / log2(min(n_samples, hidden_dim))
```
- 0~1 범위의 값
- 1에 가까울수록 표현력이 풍부함
- 0에 가까울수록 표현 공간이 붕괴됨

## 적용된 파일들

1. **collapse_collector.py**: 메인 SVD 엔트로피 계산 함수
2. **collapse_config.py**: 설정 옵션 추가
3. **gen_ea_answer_with_collapse_refactored.py**: EA 모델 평가
4. **gen_baseline_answer_with_collapse_refactored.py**: 베이스라인 모델 평가
5. **test_svd_entropy_methods.py**: 비교 테스트 스크립트

## 결론

공분산 행렬 기반 SVD 엔트로피 계산 방식은:

1. **정확한 collapse 감지**: 수학적으로 의미있는 측정
2. **향상된 민감도**: collapse 현상을 더 일찍, 더 명확하게 감지
3. **추가 분석 정보**: 유효 차원, 정규화된 엔트로피 등 제공
4. **코드 통일성**: 단일 표준 방식으로 일관된 측정

이 개선사항으로 HASS 모델의 representation collapse를 더 정확하고 민감하게 측정할 수 있습니다. 