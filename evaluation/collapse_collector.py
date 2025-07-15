import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def to_tensor_safe(f):
    if isinstance(f, torch.Tensor):
        return f.detach().clone()
    elif isinstance(f, list):
        return torch.stack([to_tensor_safe(x) for x in f])
    else:
        return torch.tensor(f, dtype=torch.float32)

class CollapseCollector:
    """
    Collapse 특성 수집기: 생성 토큰(모델 output) feature만 사용하여 representation collapse(SVD entropy) 측정.
    - 각 샘플의 생성 feature 시퀀스를 5등분하여 각 청크별 SVD entropy를 계산
    - GNC2/UNC3는 지원하지 않음
    """
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        self.features = []
        self.token_ids = []
    
    def collect_features_efficient(self, input_ids, attention_mask):
        """HASS 구조에 맞는 효율적인 특성 수집 (베이스 코드 스타일)"""
        try:
            logger.info(f"[DEBUG] Input IDs shape: {input_ids.shape}, device: {input_ids.device}")
            logger.info(f"[DEBUG] Attention mask shape: {attention_mask.shape}, device: {attention_mask.device}")
            logger.info(f"[DEBUG] Input IDs dtype: {input_ids.dtype}, Attention mask dtype: {attention_mask.dtype}")
            with torch.no_grad():
                # 모델 타입에 따라 다른 처리
                if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
                    # HASS EaModel 구조
                    logger.info("Using HASS EaModel structure")
                    outputs = self.model.base_model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False  # 메모리 절약
                    )
                else:
                    # 일반적인 transformers 모델 구조
                    logger.info("Using standard transformers model structure")
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                
                # Hidden states 추출
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # 마지막 레이어의 hidden states 사용
                    hidden_states = outputs.hidden_states[-1]
                    logger.info(f"[DEBUG] Hidden states shape: {hidden_states.shape}, device: {hidden_states.device}")
                else:
                    # outputs[0]이 hidden states인 경우
                    hidden_states = outputs[0]
                    logger.info(f"[DEBUG] Outputs[0] shape: {hidden_states.shape}, device: {hidden_states.device}")
                
                # 베이스 코드 스타일: 디바이스 불일치 무시, PyTorch가 자동 처리하도록 둠
                X, Y = self._extract_features_accurate(
                    hidden_states, input_ids, attention_mask
                )
                logger.info(f"[DEBUG] _extract_features_accurate returned X shape: {X.shape if X is not None else None}, Y shape: {Y.shape if Y is not None else None}")
                return X, Y
        except Exception as e:
            logger.error(f"Feature collection failed: {e}")
            logger.error(f"Exception details: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
    
    def _extract_features_accurate(self, hidden_states, input_ids, attention_mask):
        """HASS 구조에 맞는 정확한 토큰-특성 매칭"""
        # HASS: hidden_states는 이미 LM Head 이전의 특성
        Y = input_ids[:, 1:]  # 다음 토큰 (타겟)
        X = hidden_states[:, :-1]  # 현재 토큰의 hidden states
        logger.info(f"[DEBUG] _extract_features_accurate: Y shape before mask: {Y.shape}, X shape before mask: {X.shape}")
        # 유효한 토큰만 선택
        valid_mask = attention_mask[:, 1:].bool()
        # device 일치 - Y와 X 모두에 맞춤
        valid_mask_y = valid_mask.to(Y.device)
        valid_mask_x = valid_mask.to(X.device)
        Y, X = Y[valid_mask_y], X[valid_mask_x]
        logger.info(f"[DEBUG] _extract_features_accurate: Y shape after mask: {Y.shape}, X shape after mask: {X.shape}")
        return X, Y
    
    def add_batch_features(self, features, token_ids):
        """배치 특성 추가 (리스트/텐서 직접 관리)"""
        logger.info(f"[DEBUG] add_batch_features: features is None? {features is None}, token_ids is None? {token_ids is None}")
        if features is not None and token_ids is not None:
            logger.info(f"[DEBUG] add_batch_features: features shape: {features.shape}, token_ids shape: {token_ids.shape}")
            if isinstance(features, torch.Tensor):
                self.features.append(features.cpu())
            else:
                self.features.append(torch.tensor(features))
            if isinstance(token_ids, torch.Tensor):
                self.token_ids.append(token_ids.cpu())
            else:
                self.token_ids.append(torch.tensor(token_ids))
        logger.info(f"[DEBUG] add_batch_features: total features collected: {len(self.features)}, total token_ids collected: {len(self.token_ids)}")
    
    def get_all_features(self):
        """누적된 모든 특성 반환 (텐서)"""
        if not self.features or not self.token_ids:
            return torch.tensor([]), torch.tensor([])
        features = torch.cat(self.features, dim=0)
        token_ids = torch.cat(self.token_ids, dim=0)
        return features, token_ids
    
    def get_generation_features(self, input_len):
        """
        Returns only the features corresponding to generated tokens (not prompt/input).
        Args:
            input_len (int): Length of the prompt/input tokens
        Returns:
            Tensor: [num_generated, hidden_dim] features for generated tokens
        """
        features, _ = self.get_all_features()
        logger.info(f"[DEBUG] get_generation_features: all features shape: {features.shape}, input_len: {input_len}")
        if features.shape[0] <= input_len:
            logger.info(f"[DEBUG] get_generation_features: No generated tokens (features.shape[0]={features.shape[0]}, input_len={input_len})")
            return torch.tensor([])
        logger.info(f"[DEBUG] get_generation_features: Returning features[{input_len}:] shape: {features[input_len:].shape}")
        return features[input_len:]

    def get_chunk_svd_entropies(self, gen_features, num_chunks=5):
        """
        Split the generated feature sequence into num_chunks and compute SVD entropy for each chunk.
        Args:
            gen_features (Tensor): [num_generated, hidden_dim]
            num_chunks (int): Number of chunks (default 5)
        Returns:
            List[float]: SVD entropy for each chunk (length <= num_chunks)
        """
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
                entropies.append(self._compute_svd_entropy(chunk))
        return entropies

    def get_collapse_metrics(self, input_len, num_chunks=5):
        """
        Returns chunk-wise SVD entropy for generated tokens only.
        Args:
            input_len (int): Length of the prompt/input tokens
            num_chunks (int): Number of chunks (default 5)
        Returns:
            dict: {
                'total_generated_tokens': int,
                'chunk_svd_entropies': List[float]
            }
        """
        gen_features = self.get_generation_features(input_len)
        entropies = self.get_chunk_svd_entropies(gen_features, num_chunks=num_chunks)
        return {
            'total_generated_tokens': gen_features.shape[0],
            'chunk_svd_entropies': entropies
        }

    def _compute_svd_entropy(self, features):
        """SVD 엔트로피 계산"""
        try:
            features = features.float()  # float16 → float32 변환
            # 특성 행렬의 SVD 계산
            U, S, V = torch.svd(features)
            # 정규화된 특이값
            S_norm = S / S.sum()
            # 엔트로피 계산 (0이 아닌 값만)
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
            return entropy.item()
        except Exception as e:
            logger.error(f"Failed to compute SVD entropy: {e}")
            return 0.0
    
    def clear(self):
        """수집기 정리 (내부 리스트 초기화)"""
        self.features = []
        self.token_ids = [] 

    def _group_features_by_token(self, features, token_ids):
        """토큰별 특성 그룹화 (스페셜 토큰 제외)"""
        token_features = {}
        # 스페셜 토큰 ID 목록 생성
        special_token_ids = set()
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                special_token_ids.add(self.tokenizer.eos_token_id)
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                special_token_ids.add(self.tokenizer.pad_token_id)
            if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                special_token_ids.add(self.tokenizer.bos_token_id)
            if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
                special_token_ids.add(self.tokenizer.unk_token_id)
        for i, token_id in enumerate(token_ids):
            token_id = token_id.item() if hasattr(token_id, 'item') else int(token_id)
            if token_id in special_token_ids:
                continue  # 스페셜 토큰은 제외
            if token_id not in token_features:
                token_features[token_id] = []
            token_features[token_id].append(features[i])
        # 텐서로 변환
        for token_id in list(token_features.keys()):
            if len(token_features[token_id]) >= self.config.min_samples_per_token:
                feature_list = token_features[token_id]
                try:
                    tensor_features = [to_tensor_safe(f) for f in feature_list]
                    token_features[token_id] = torch.stack(tensor_features)
                except Exception as e:
                    logger.error(f"Failed to stack tensors for token {token_id}: {e}")
                    del token_features[token_id]
            else:
                del token_features[token_id]
        return token_features

    def _compute_gnc2_class_means(self, token_features):
        """GNC2: 논문 방식(CoV) - 클래스 평균 벡터의 hyperspherical uniformity"""
        try:
            # 클래스별 평균 벡터 계산
            class_means = []
            for token_id, features in token_features.items():
                if isinstance(features, list):
                    if len(features) == 0:
                        continue
                    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features])
                if features.shape[0] > 1:
                    class_mean = features.mean(dim=0)
                    class_means.append(class_mean)
            if len(class_means) < 2:
                return 0.0
            class_means = torch.stack(class_means)
            # 중심화 및 정규화
            global_mean = class_means.mean(dim=0, keepdim=True)
            centered_means = class_means - global_mean
            normalized_means = torch.nn.functional.normalize(centered_means, p=2, dim=1)
            # 모든 쌍 간의 로그 역 거리 계산
            pdist = torch.pdist(normalized_means, p=2)
            log_inv_distances = torch.log(1.0 / pdist)
            # 변동 계수(CoV) 계산
            mean_dist = log_inv_distances.mean()
            std_dist = log_inv_distances.std()
            return (std_dist / mean_dist).item()
        except Exception as e:
            logger.error(f"Failed to compute GNC2 (reference style): {e}")
            return 0.0

    def _compute_unc3_duality(self, token_features, classifier_weights):
        """UNC3: 논문 방식(CoV) - classifier weight와 클래스 평균의 duality"""
        try:
            # 클래스별 평균 벡터 계산 및 공통 클래스 ID 추출
            class_means_map = {}
            for token_id, features in token_features.items():
                if isinstance(features, list):
                    if len(features) == 0:
                        continue
                    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features])
                if features.shape[0] > 1:
                    class_means_map[token_id] = features.mean(dim=0)
            if len(class_means_map) < 2:
                return 0.0
            common_ids = sorted(list(class_means_map.keys()))
            class_means = torch.stack([class_means_map[cid] for cid in common_ids])
            weights = classifier_weights[common_ids, :]
            # 중심화 및 정규화
            global_mean = class_means.mean(dim=0, keepdim=True)
            normalized_means = torch.nn.functional.normalize(class_means - global_mean, p=2, dim=1)
            normalized_weights = torch.nn.functional.normalize(weights, p=2, dim=1)
            # 클래스별 코사인 유사도 계산
            similarities = (normalized_weights * normalized_means).sum(dim=1)
            # 변동 계수(CoV) 계산
            mean_sim = similarities.mean()
            std_sim = similarities.std()
            return (std_sim / mean_sim).item()
        except Exception as e:
            logger.error(f"Failed to compute UNC3 (reference style): {e}")
            return 0.0

    def get_gnc2_unc3_metrics(self):
        """
        (Pile 평가 등에서만 사용) GNC2/UNC3를 별도로 측정.
        Returns:
            dict: {
                'gnc2': float,
                'unc3': float
            }
        """
        try:
            all_features, all_tokens = self.get_all_features()
            if all_features.numel() == 0:
                logger.warning("No features collected for GNC2/UNC3 analysis (empty tensor)")
                return None
            token_features = self._group_features_by_token(all_features, all_tokens)
            # classifier weights 추출 (Huggingface 모델에서만)
            classifier_weights = None
            real_model = self.model
            if hasattr(self.model, 'model'):
                real_model = self.model.model
            if hasattr(real_model, 'get_output_embeddings'):
                classifier_weights = real_model.get_output_embeddings().weight.detach().cpu()
            elif hasattr(real_model, 'lm_head') and hasattr(real_model.lm_head, 'weight'):
                classifier_weights = real_model.lm_head.weight.detach().cpu()
            else:
                logger.warning("Cannot extract classifier weights from model.")
            return {
                'gnc2': self._compute_gnc2_class_means(token_features),
                'unc3': self._compute_unc3_duality(token_features, classifier_weights) if classifier_weights is not None else None
            }
        except Exception as e:
            logger.error(f"Failed to compute GNC2/UNC3 metrics: {e}")
            return None 