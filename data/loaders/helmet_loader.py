"""HELMET dataset loader for HASS evaluation."""

import os
import sys
import json
from typing import Dict, List, Any, Optional
import logging

# HELMET 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
hass_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
helmet_dir = os.path.join(hass_dir, "helmet")

if helmet_dir not in sys.path:
    sys.path.insert(0, helmet_dir)

from data import load_data
from model_utils import load_LLM

logger = logging.getLogger(__name__)

class HelmetLoader:
    """HELMET 데이터셋을 HASS에서 사용하기 위한 로더"""
    
    def __init__(self, helmet_data_dir: str = None):
        """
        Args:
            helmet_data_dir: HELMET 데이터 디렉토리 경로
        """
        self.helmet_data_dir = helmet_data_dir or os.path.join(helmet_dir, "data")
        self.available_datasets = self._get_available_datasets()
        
    def _get_available_datasets(self) -> List[str]:
        """사용 가능한 HELMET 데이터셋 목록 반환"""
        datasets = []
        if os.path.exists(self.helmet_data_dir):
            for item in os.listdir(self.helmet_data_dir):
                item_path = os.path.join(self.helmet_data_dir, item)
                if os.path.isdir(item_path):
                    datasets.append(item)
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        dataset_path = os.path.join(self.helmet_data_dir, dataset_name)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset {dataset_name} not found in {self.helmet_data_dir}")
        
        # 데이터셋별 정보
        dataset_info = {
            "json_kv": {
                "type": "key-value retrieval",
                "description": "JSON key-value extraction",
                "test_file": "test.jsonl",
                "demo_file": "demo.jsonl"
            },
            "infbench": {
                "type": "long-context QA",
                "description": "InfiniteBench long-context tasks",
                "test_file": "test.jsonl", 
                "demo_file": "demo.jsonl"
            },
            "multi_lexsum": {
                "type": "summarization",
                "description": "Multi-lexsum legal document summarization",
                "test_file": "test.jsonl",
                "demo_file": "demo.jsonl"
            },
            "msmarco": {
                "type": "reranking",
                "description": "MS MARCO document reranking",
                "test_file": "test.jsonl",
                "demo_file": "demo.jsonl"
            },
            "ruler": {
                "type": "reasoning",
                "description": "RULER reasoning tasks",
                "test_file": "test.jsonl",
                "demo_file": "demo.jsonl"
            },
            "kilt": {
                "type": "knowledge-intensive",
                "description": "KILT knowledge-intensive tasks",
                "test_file": "test.jsonl",
                "demo_file": "demo.jsonl"
            },
            "alce": {
                "type": "citation",
                "description": "ALCE citation generation",
                "test_file": "test.jsonl",
                "demo_file": "prompts.json"
            }
        }
        
        return dataset_info.get(dataset_name, {
            "type": "unknown",
            "description": "Unknown dataset type",
            "test_file": "test.jsonl",
            "demo_file": "demo.jsonl"
        })
    
    def load_dataset(self, dataset_name: str, max_samples: Optional[int] = None, 
                    shots: int = 0, seed: int = 42) -> Dict[str, Any]:
        """
        HELMET 데이터셋 로드
        
        Args:
            dataset_name: 데이터셋 이름
            max_samples: 최대 샘플 수
            shots: few-shot 예제 수
            seed: 랜덤 시드
            
        Returns:
            데이터셋 정보 딕셔너리
        """
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not available. Available: {self.available_datasets}")
        
        dataset_info = self.get_dataset_info(dataset_name)
        test_file = os.path.join(self.helmet_data_dir, dataset_name, dataset_info["test_file"])
        demo_file = os.path.join(self.helmet_data_dir, dataset_name, dataset_info["demo_file"])
        
        # HELMET의 load_data 함수 사용
        try:
            # Mock args 객체 생성
            class MockArgs:
                def __init__(self):
                    self.max_test_samples = max_samples
                    self.shots = shots
                    self.seed = seed
                    self.popularity_threshold = None
            
            args = MockArgs()
            
            data = load_data(args, dataset_name, test_file, demo_file)
            
            logger.info(f"Loaded {len(data['data'])} samples from {dataset_name}")
            
            return {
                "data": data["data"],
                "prompt_template": data["prompt_template"],
                "user_template": data["user_template"], 
                "system_template": data["system_template"],
                "post_process": data.get("post_process"),
                "dataset_info": dataset_info
            }
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def list_datasets(self) -> List[str]:
        """사용 가능한 데이터셋 목록 반환"""
        return self.available_datasets
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """데이터셋 통계 정보 반환"""
        try:
            data = self.load_dataset(dataset_name, max_samples=None)
            stats = {
                "dataset_name": dataset_name,
                "total_samples": len(data["data"]),
                "dataset_info": data["dataset_info"]
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats for {dataset_name}: {e}")
            return {"error": str(e)} 