"""LongProc dataset loader for HASS evaluation."""

import os
import json
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LongProcLoader:
    """LongProc 데이터셋을 HASS에서 사용하기 위한 로더"""
    
    def __init__(self, longproc_data_dir: str = None):
        """
        Args:
            longproc_data_dir: LongProc 데이터 디렉토리 경로
        """
        self.longproc_data_dir = longproc_data_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "longproc"
        )
        self.available_datasets = self._get_available_datasets()
        
    def _get_available_datasets(self) -> List[str]:
        """사용 가능한 LongProc 데이터셋 목록 반환"""
        datasets = []
        if os.path.exists(self.longproc_data_dir):
            for item in os.listdir(self.longproc_data_dir):
                item_path = os.path.join(self.longproc_data_dir, item)
                if os.path.isdir(item_path):
                    datasets.append(item)
        return datasets
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """LongProc 데이터셋 로드"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not available. Available: {self.available_datasets}")
        
        dataset_path = os.path.join(self.longproc_data_dir, dataset_name)
        test_file = os.path.join(dataset_path, "test.jsonl")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        # 데이터 로드
        data = []
        with open(test_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        if max_samples:
            data = data[:max_samples]
        
        logger.info(f"Loaded {len(data)} samples from {dataset_name}")
        
        return {
            "data": data,
            "dataset_name": dataset_name,
            "total_samples": len(data)
        }
    
    def list_datasets(self) -> List[str]:
        """사용 가능한 데이터셋 목록 반환"""
        return self.available_datasets 