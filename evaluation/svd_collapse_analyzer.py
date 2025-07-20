import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SVDCollapseAnalyzer:
    """
    SVD 엔트로피 기반 청크-wise representation collapse 분석 클래스
    (공분산 행렬 기반 SVD 엔트로피 계산)
    """
    def __init__(self, config=None): # config를 선택적으로 받도록 수정
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
        
        # 평균 및 CV 통계 계산 추가
        stats = self._calculate_entropy_statistics(entropies)
        
        return {
            'total_generated_tokens': gen_features.shape[0],
            'chunk_svd_entropies': entropies,
            **stats  # 딕셔너리에 통계 결과 병합
        }

    def get_collapse_metrics_fixed_chunk(self, features, chunk_size=64): # 기본값을 64로 수정
        """
        고정된 크기의 청크로 분할하여 SVD 엔트로피를 계산합니다.
        """
        gen_features = self.get_generation_features(features)
        entropies = self._get_fixed_chunk_svd_entropies(gen_features, chunk_size)
        
        # 평균 및 CV 통계 계산 추가
        stats = self._calculate_entropy_statistics(entropies)
        
        return {
            'total_generated_tokens': gen_features.shape[0],
            'chunk_size': chunk_size,
            'fixed_chunk_svd_entropies': entropies,
            'num_valid_chunks': len(entropies),
            **stats  # 딕셔너리에 통계 결과 병합
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

    def _calculate_entropy_statistics(self, chunk_entropies: list) -> dict:
        """
        주어진 청크 엔트로피 리스트로부터 평균과 변동 계수(CV)를 계산합니다.
        """
        # None 값을 제외하고 유효한 엔트로피 값만 필터링
        valid_entropies = [e for e in chunk_entropies if e is not None and e > 0]
        
        if not valid_entropies:
            return {"avg_svd_entropy": None, "cv_svd_entropy": None}
            
        # 평균 계산
        avg_entropy = np.mean(valid_entropies)
        
        # 표준편차 계산
        std_entropy = np.std(valid_entropies)
        
        # 변동 계수(CV) 계산 (평균이 0인 경우 방지)
        cv_entropy = (std_entropy / avg_entropy) if avg_entropy > 0 else 0.0
        
        return {
            "avg_svd_entropy": float(avg_entropy),
            "cv_svd_entropy": float(cv_entropy)
        }