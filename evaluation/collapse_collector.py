import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from memory_manager import IncrementalMemoryManager

logger = logging.getLogger(__name__)

def to_tensor_safe(f):
    if isinstance(f, torch.Tensor):
        return f.detach().clone()
    elif isinstance(f, list):
        return torch.stack([to_tensor_safe(x) for x in f])
    else:
        return torch.tensor(f, dtype=torch.float32)

class CollapseCollector:
    """HASS 구조에 맞는 효율적인 collapse 특성 수집기"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.memory_manager = IncrementalMemoryManager(
            save_every=config.save_every,
            max_buffer_size=config.max_buffer_size
        )
        self.device = config.device
    
    def collect_features_efficient(self, input_ids, attention_mask):
        """HASS 구조에 맞는 효율적인 특성 수집 (베이스 코드 스타일)"""
        try:
            logger.info(f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}")
            logger.info(f"Attention mask shape: {attention_mask.shape}, device: {attention_mask.device}")
            logger.info(f"Input IDs dtype: {input_ids.dtype}, Attention mask dtype: {attention_mask.dtype}")
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
                    logger.info(f"Hidden states shape: {hidden_states.shape}, device: {hidden_states.device}")
                else:
                    # outputs[0]이 hidden states인 경우
                    hidden_states = outputs[0]
                    logger.info(f"Outputs[0] shape: {hidden_states.shape}, device: {hidden_states.device}")
                
                # 베이스 코드 스타일: 디바이스 불일치 무시, PyTorch가 자동 처리하도록 둠
                return self._extract_features_accurate(
                    hidden_states, input_ids, attention_mask
                )
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
        
        # 유효한 토큰만 선택
        valid_mask = attention_mask[:, 1:].bool()
        # device 일치 - Y와 X 모두에 맞춤
        valid_mask_y = valid_mask.to(Y.device)
        valid_mask_x = valid_mask.to(X.device)
        Y, X = Y[valid_mask_y], X[valid_mask_x]
        
        return X, Y
    
    def add_batch_features(self, features, token_ids):
        """배치 특성 추가"""
        if features is not None and token_ids is not None:
            self.memory_manager.add_features(features, token_ids)
    
    def get_collapse_metrics(self):
        """collapse 메트릭 계산"""
        try:
            all_features, all_tokens = self.memory_manager.get_all_features()
            # 리스트 타입이면 텐서로 변환
            if isinstance(all_features, list):
                if len(all_features) == 0:
                    logger.warning("No features collected for collapse analysis (empty list)")
                    return None
                all_features = torch.tensor(all_features)
            if isinstance(all_tokens, list):
                if len(all_tokens) == 0:
                    logger.warning("No tokens collected for collapse analysis (empty list)")
                    return None
                all_tokens = torch.tensor(all_tokens)
            if all_features.numel() == 0:
                logger.warning("No features collected for collapse analysis (empty tensor)")
                return None
            return self._compute_collapse_metrics(all_features, all_tokens)
        except Exception as e:
            logger.error(f"Collapse metrics computation failed: {e}")
            return None
    
    def _compute_collapse_metrics(self, features, token_ids):
        """collapse 메트릭 계산 (GNC2, UNC3, SVD 엔트로피)"""
        try:
            # 리스트 타입이면 텐서로 변환
            if isinstance(features, list):
                if len(features) == 0:
                    logger.warning("No features for collapse metrics (empty list)")
                    return None
                features = torch.tensor(features)
            if isinstance(token_ids, list):
                if len(token_ids) == 0:
                    logger.warning("No token_ids for collapse metrics (empty list)")
                    return None
                token_ids = torch.tensor(token_ids)
            # 토큰별 특성 그룹화
            token_features = self._group_features_by_token(features, token_ids)
            
            results = {
                "total_tokens": len(token_features),
                "token_metrics": {},
                "overall_metrics": {}
            }
            
            # 개별 토큰 메트릭 계산
            for token_id, features in token_features.items():
                # features가 list면 torch.tensor로 변환
                if isinstance(features, list):
                    if len(features) == 0:
                        continue
                    # 디버깅: list 요소 확인
                    tensor_features = []
                    for i, f in enumerate(features):
                        if isinstance(f, list):
                            logger.warning(f"Element {i} is still a list: {type(f)}")
                            f = torch.tensor(f, dtype=torch.float32)
                        elif isinstance(f, torch.Tensor):
                            f = f.detach().clone()
                        else:
                            f = torch.tensor(f, dtype=torch.float32)
                        tensor_features.append(f)
                    features = torch.stack(tensor_features)
                
                if features.shape[0] < self.config.min_samples_per_token:
                    continue
                results["token_metrics"][token_id] = {
                    'svd_entropy': self._compute_svd_entropy(features),
                    'num_samples': features.shape[0]
                }
            
            # 전체 메트릭 계산
            if token_features:
                all_features = torch.cat(list(token_features.values()), dim=0)
                results["overall_metrics"] = {
                    'svd_entropy': self._compute_svd_entropy(all_features),
                    'gnc2': self._compute_gnc2_class_means(token_features),
                    'unc3': self._compute_unc3_duality(token_features)
                }
            
            return results
        except Exception as e:
            logger.error(f"Failed to compute collapse metrics: {e}")
            return None
    
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
            # 추가적으로 bos, unk 등 필요시 확장 가능
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
    
    def _compute_gnc2_class_means(self, token_features):
        """GNC2: 클래스 평균 벡터 기반 collapse 측정"""
        try:
            if len(token_features) < 2:
                return 0.0
            
            # 각 토큰 클래스의 평균 벡터 계산
            class_means = []
            for token_id, features in token_features.items():
                # features가 list면 torch.tensor로 변환
                if isinstance(features, list):
                    if len(features) == 0:
                        continue
                    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features])
                
                if features.shape[0] >= self.config.min_samples_per_token:
                    class_mean = features.mean(dim=0)
                    class_means.append(class_mean)
            
            if len(class_means) < 2:
                return 0.0
            
            # 모든 클래스 평균을 하나의 텐서로 스택
            means_tensor = torch.stack(class_means)
            
            # GNC2 계산
            mu_bar = means_tensor.mean(dim=0, keepdim=True)
            centered = means_tensor - mu_bar
            normed = centered / (centered.norm(dim=1, keepdim=True) + 1e-8)
            
            n = normed.size(0)
            values = []
            for i in range(n):
                for j in range(n):
                    if i == j: 
                        continue
                    dist = (normed[i] - normed[j]).norm()
                    # 거리가 너무 작으면 건너뛰기
                    if dist < 1e-6:
                        continue
                    values.append(torch.log(1.0 / dist))
            
            if values:
                return torch.stack(values).mean().item()
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to compute GNC2: {e}")
            return 0.0
    
    def _compute_unc3_duality(self, token_features):
        """UNC3: classifier와 클래스 평균 벡터의 cosine similarity CoV"""
        try:
            if len(token_features) < 2:
                return 0.0
            
            # 각 토큰 클래스의 평균 벡터 계산
            class_means = []
            for token_id, features in token_features.items():
                # features가 list면 torch.tensor로 변환
                if isinstance(features, list):
                    if len(features) == 0:
                        continue
                    features = torch.stack([torch.tensor(f, dtype=torch.float32) for f in features])
                
                if features.shape[0] >= self.config.min_samples_per_token:
                    class_mean = features.mean(dim=0)
                    class_means.append(class_mean)
            
            if len(class_means) < 2:
                return 0.0
            
            # 모든 클래스 평균을 하나의 텐서로 스택
            means_tensor = torch.stack(class_means)
            
            # classifier는 means_tensor를 사용 (토큰 ID를 클래스로 취급)
            classifier = means_tensor
            
            # UNC3 계산
            classifier_norm = classifier / (classifier.norm(dim=1, keepdim=True) + 1e-8)
            mu_bar = means_tensor.mean(dim=0, keepdim=True)
            means_centered = means_tensor - mu_bar
            means_norm = means_centered / (means_centered.norm(dim=1, keepdim=True) + 1e-8)
            
            cos_sim = (classifier_norm * means_norm).sum(dim=1)
            
            # Coefficient of Variation (CoV) 계산
            mean_sim = cos_sim.mean()
            std_sim = cos_sim.std()
            
            if mean_sim < 1e-8:
                return 0.0
            
            return (std_sim / mean_sim).item()
            
        except Exception as e:
            logger.error(f"Failed to compute UNC3: {e}")
            return 0.0
    
    def clear(self):
        """수집기 정리"""
        self.memory_manager.clear_all() 