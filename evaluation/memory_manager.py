import torch
import gc
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IncrementalMemoryManager:
    """점진적 메모리 관리를 위한 클래스"""
    
    def __init__(self, save_every=1024, max_buffer_size=10000):
        self.buffer = []
        self.save_every = save_every
        self.max_buffer_size = max_buffer_size
        self.total_features = 0
        self.saved_features = []
    
    def add_features(self, features, token_ids):
        """점진적 특성 추가 (GPU 배치 처리)"""
        try:
            # GPU에서 배치로 처리 (CPU 이동 지연)
            self.buffer.append((
                features.detach(),  # GPU에 유지
                token_ids.detach()  # GPU에 유지
            ))
            self.total_features += features.shape[0]
            
            if len(self.buffer) >= self.save_every:
                self._save_and_clear()
        except Exception as e:
            logger.error(f"Failed to add features: {e}")
    
    def _save_and_clear(self):
        """중간 결과 저장 및 버퍼 정리 (배치 CPU 이동)"""
        if not self.buffer:
            return
        
        try:
            # GPU에서 배치로 통합
            combined_features = torch.cat([f[0] for f in self.buffer], dim=0)
            combined_tokens = torch.cat([f[1] for f in self.buffer], dim=0)
            
            # 한 번에 CPU로 이동 (배치 처리)
            combined_features = combined_features.cpu()
            combined_tokens = combined_tokens.cpu()
            
            # 중간 저장
            self.saved_features.append((combined_features, combined_tokens))
            
            # 메모리 정리
            del self.buffer
            self.buffer = []
            gc.collect()  # 가비지 컬렉션 강제 실행
            
            # 최대 버퍼 크기 제한
            if len(self.saved_features) > self.max_buffer_size:
                self._consolidate_saved_features()
                
        except Exception as e:
            logger.error(f"Failed to save and clear buffer: {e}")
    
    def _consolidate_saved_features(self):
        """저장된 특성 통합 (메모리 절약)"""
        if len(self.saved_features) <= self.max_buffer_size // 2:
            return
        
        try:
            # 절반씩 통합
            mid = len(self.saved_features) // 2
            first_half = self.saved_features[:mid]
            second_half = self.saved_features[mid:]
            
            # 첫 번째 절반 통합
            if first_half:
                features_list = [f[0] for f in first_half]
                tokens_list = [f[1] for f in first_half]
                consolidated_features = torch.cat(features_list, dim=0)
                consolidated_tokens = torch.cat(tokens_list, dim=0)
                
                self.saved_features = [
                    (consolidated_features, consolidated_tokens)
                ] + second_half
                
        except Exception as e:
            logger.error(f"Failed to consolidate saved features: {e}")
    
    def get_all_features(self):
        """모든 특성 반환 (분석용)"""
        try:
            if not self.saved_features and not self.buffer:
                return torch.tensor([]), torch.tensor([])
            
            # 저장된 특성들 통합
            all_features = []
            all_tokens = []
            
            for features, tokens in self.saved_features:
                all_features.append(features)
                all_tokens.append(tokens)
            
            if self.buffer:
                buffer_features = torch.cat([f[0] for f in self.buffer], dim=0)
                buffer_tokens = torch.cat([f[1] for f in self.buffer], dim=0)
                all_features.append(buffer_features)
                all_tokens.append(buffer_tokens)
            
            if all_features:
                return torch.cat(all_features, dim=0), torch.cat(
                    all_tokens, dim=0
                )
            else:
                return torch.tensor([]), torch.tensor([])
                
        except Exception as e:
            logger.error(f"Failed to get all features: {e}")
            return torch.tensor([]), torch.tensor([])
    
    def clear_all(self):
        """모든 데이터 정리"""
        del self.buffer
        del self.saved_features
        self.buffer = []
        self.saved_features = []
        self.total_features = 0
        gc.collect()
