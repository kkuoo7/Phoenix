import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SVDCollapseAnalyzer:
    """
    SVD 엔트로피 기반 청크-wise representation collapse 분석 클래스
    (공분산 행렬 기반 SVD 엔트로피 계산)
    """
    def __init__(self, config):
        self.config = config

    def get_generation_features(self, features, input_len):
        """프롬프트 이후 생성 토큰에 해당하는 feature만 추출"""
        if features.shape[0] <= input_len:
            return torch.tensor([])
        return features[input_len:]

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

    def get_collapse_metrics(self, features, input_len, num_chunks=5):
        gen_features = self.get_generation_features(features, input_len)
        entropies = self.get_chunk_svd_entropies(gen_features, num_chunks=num_chunks)
        return {
            'total_generated_tokens': gen_features.shape[0],
            'chunk_svd_entropies': entropies
        }

    def _compute_svd_entropy_cov(self, features):
        """공분산 행렬 기반 SVD 엔트로피 계산"""
        try:
            features = features.float()
            # (N, D) → (D, D) 공분산 행렬
            cov = torch.cov(features.T)
            S = torch.linalg.svdvals(cov)
            S = S[S > 0]  # 0 이상만
            S_norm = S / S.sum()
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
            return entropy.item()
        except Exception as e:
            logger.error(f"Failed to compute SVD entropy (covariance): {e}")
            return 0.0 