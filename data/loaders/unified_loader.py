"""Unified dataset loader for HASS evaluation."""

import os
import json
from typing import Dict, List, Any, Optional, Union
import logging

from .helmet_loader import HelmetLoader
from .longproc_loader import LongProcLoader
from .eagle_loader import EagleLoader

logger = logging.getLogger(__name__)

class UnifiedLoader:
    """모든 데이터셋을 통합적으로 로드하는 로더"""
    
    def __init__(self):
        """초기화"""
        self.helmet_loader = HelmetLoader()
        self.longproc_loader = LongProcLoader()
        self.eagle_loader = EagleLoader()
        
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """사용 가능한 모든 데이터셋 목록 반환"""
        return {
            "helmet": self.helmet_loader.list_datasets(),
            "longproc": self.longproc_loader.list_datasets(),
            "eagle": self.eagle_loader.list_datasets()
        }
    
    def load_dataset(self, dataset_name: str, dataset_type: str = "auto", 
                    max_samples: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        데이터셋 로드
        
        Args:
            dataset_name: 데이터셋 이름
            dataset_type: 데이터셋 타입 ("helmet", "longproc", "eagle", "auto")
            max_samples: 최대 샘플 수
            **kwargs: 추가 인자들
            
        Returns:
            데이터셋 정보 딕셔너리
        """
        if dataset_type == "auto":
            # 자동으로 데이터셋 타입 결정
            dataset_type = self._detect_dataset_type(dataset_name)
        
        if dataset_type == "helmet":
            return self.helmet_loader.load_dataset(dataset_name, max_samples, **kwargs)
        elif dataset_type == "longproc":
            return self.longproc_loader.load_dataset(dataset_name, max_samples)
        elif dataset_type == "eagle":
            return self.eagle_loader.load_dataset(dataset_name, max_samples)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _detect_dataset_type(self, dataset_name: str) -> str:
        """데이터셋 타입 자동 감지"""
        helmet_datasets = self.helmet_loader.list_datasets()
        longproc_datasets = self.longproc_loader.list_datasets()
        eagle_datasets = self.eagle_loader.list_datasets()
        
        if dataset_name in helmet_datasets:
            return "helmet"
        elif dataset_name in longproc_datasets:
            return "longproc"
        elif dataset_name in eagle_datasets:
            return "eagle"
        else:
            raise ValueError(f"Dataset {dataset_name} not found in any loader")
    
    def get_dataset_info(self, dataset_name: str, dataset_type: str = "auto") -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        if dataset_type == "auto":
            dataset_type = self._detect_dataset_type(dataset_name)
        
        if dataset_type == "helmet":
            return self.helmet_loader.get_dataset_info(dataset_name)
        else:
            return {
                "dataset_name": dataset_name,
                "dataset_type": dataset_type,
                "description": f"{dataset_type} dataset"
            } 