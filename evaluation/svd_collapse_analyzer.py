import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SVDCollapseAnalyzer:
    """
    SVD 엔트로피 기반 청크-wise representation collapse 분석 클래스
    (공분산 행렬 기반 SVD 엔트로피 계산)
    """
    def __init__(self, config=None):
        self.config = config

    def get_generation_features(self, features):
        if features is None or features.numel() == 0:
            return torch.tensor([])
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features

    def get_chunk_svd_entropies(self, gen_features, num_chunks=5):
        n = gen_features.shape[0]
        if n == 0:
            return [None] * num_chunks
        chunk_size = max(1, n // num_chunks)
        entropies = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_chunks - 1 else n
            chunk = gen_features[start:end]
            if chunk.shape[0] < 2:
                entropies.append(None)
            else:
                entropies.append(self._compute_svd_entropy_cov(chunk))
        return entropies

    def get_collapse_metrics(self, features, num_chunks=5):
        gen_features = self.get_generation_features(features)
        entropies = self.get_chunk_svd_entropies(gen_features, num_chunks=num_chunks)
        stats = self._calculate_entropy_statistics(entropies)
        
        return {
            'total_generated_tokens': gen_features.shape[0],
            'chunk_svd_entropies': entropies,
            **stats
        }

    def get_collapse_metrics_fixed_chunk(self, features, chunk_size=64):
        """
        고정된 크기의 청크로 분할하여 SVD 엔트로피를 계산합니다.
        """
        gen_features = self.get_generation_features(features)
        entropies = self._get_fixed_chunk_svd_entropies(gen_features, chunk_size)
        stats = self._calculate_entropy_statistics(entropies)
        
        return {
            'total_generated_tokens': gen_features.shape[0],
            'chunk_size': chunk_size,
            'fixed_chunk_svd_entropies': entropies,
            'num_valid_chunks': len(entropies),
            **stats
        }

    def _get_fixed_chunk_svd_entropies(self, gen_features, chunk_size):
        n_total_tokens = gen_features.shape[0]
        if n_total_tokens < chunk_size:
            return []
        
        num_chunks = n_total_tokens // chunk_size
        entropies = []
        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = start_index + chunk_size
            chunk = gen_features[start_index:end_index]
            if chunk.shape[0] < 2:
                continue
            entropy = self._compute_svd_entropy_cov(chunk)
            entropies.append(entropy)
        
        return entropies

    def _compute_svd_entropy_cov(self, features):
        """공분산 행렬 기반 SVD 엔트로피 계산"""
        try:
            features = features.float()
            cov = torch.cov(features.T)
            S = torch.linalg.svdvals(cov)
            S = S[S > 0]
            if S.numel() == 0:
                return 0.0
            S_norm = S / S.sum()
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
            return entropy.item()
        except Exception as e:
            logger.error(f"Failed to compute SVD entropy (covariance): {e}")
            return 0.0

    def _calculate_entropy_trend_slope(self, valid_entropies: list) -> float or None:
        """
        엔트로피 리스트의 추세선 기울기를 계산합니다.
        """
        if len(valid_entropies) < 2:
            return None # 점이 2개 미만이면 기울기를 계산할 수 없음
        
        chunk_indices = np.arange(len(valid_entropies))
        # np.polyfit은 선형 회귀를 수행하고 계수(기울기, 절편)를 반환합니다.
        # 1차 다항식(직선)이므로 deg=1을 사용합니다.
        slope, _ = np.polyfit(chunk_indices, valid_entropies, 1)
        return float(slope)

    def _calculate_entropy_statistics(self, chunk_entropies: list) -> dict:
        """
        주어진 청크 엔트로피 리스트로부터 평균, 변동 계수(CV), 추세 기울기를 계산합니다.
        """
        valid_entropies = [e for e in chunk_entropies if e is not None and e > 0]
        
        # 기본값 설정
        avg_entropy = 0.0
        cv_entropy = 0.0
        slope_entropy = 0.0 # 초기값을 0.0으로 설정

        if valid_entropies: # valid_entropies가 비어있지 않은 경우에만 계산
            avg_entropy = np.mean(valid_entropies)
            std_entropy = np.std(valid_entropies)
            cv_entropy = (std_entropy / avg_entropy) if avg_entropy > 0 else 0.0
            
            # 추세 기울기 계산 함수 호출
            # _calculate_entropy_trend_slope가 None을 반환할 수 있으므로, 결과에 따라 처리
            calculated_slope = self._calculate_entropy_trend_slope(valid_entropies)
            if calculated_slope is not None:
                slope_entropy = calculated_slope
            # else: slope_entropy는 초기값 0.0을 유지

        return {
            "avg_svd_entropy": float(avg_entropy),
            "cv_svd_entropy": float(cv_entropy),
            "slope_svd_entropy": float(slope_entropy) # 항상 float으로 반환하도록 보장
        }