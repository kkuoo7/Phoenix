from dataclasses import dataclass
from typing import Optional

@dataclass
class CollapseConfig:
    """Collapse 평가를 위한 설정 클래스"""
    
    # 메모리 관리 설정
    save_every: int = 1024
    max_buffer_size: int = 10000
    device: str = "cpu"
    
    # 수집 설정
    min_samples_per_token: int = 2
    max_tokens_to_analyze: Optional[int] = None
    
    # 분석 설정
    compute_token_metrics: bool = True
    compute_overall_metrics: bool = True
    compute_class_metrics: bool = True  # GNC2, UNC3용
    
    # GNC2, UNC3 설정
    min_classes_for_gnc2: int = 2
    min_classes_for_unc3: int = 2
    
    # 에러 처리 설정
    continue_on_error: bool = True
    log_level: str = "INFO"
    
    # 파일 저장 설정
    save_intermediate_results: bool = True
    intermediate_save_dir: str = "collapse_intermediate"
