# HASS/evaluation/collapse_collector.py (수정된 최종 버전)

import torch
from collections import defaultdict

class CollapseCollector:
    """
    모델의 디코딩 과정에서 발생하는 마지막 히든 벡터(penultimate features)를
    상태별로 수집하고 관리하는 클래스입니다.
    
    이 클래스는 데이터 '저장소'의 역할만 수행하며, 모델을 직접 호출하지 않습니다.
    """
    def __init__(self):
        """
        초기화 시, 상태(str)를 키로, 텐서 리스트를 값으로 갖는
        defaultdict를 생성하여 데이터를 종류별로 저장합니다.
        """
        self.hidden_states = defaultdict(list)

    def add(self, state: str, hidden_state: torch.Tensor):
        """
        주어진 상태에 해당하는 히든 벡터를 리스트에 추가합니다.
        메모리 누수를 방지하고 GPU 의존성을 제거하기 위해 .detach().cpu()를 사용합니다.

        Args:
            state (str): 피처의 상태 ('accepted', 'rejected', 'bonus', 'baseline_accepted' 등).
            hidden_state (torch.Tensor): 수집된 (batch_size, 1, hidden_dim) 또는 
                                        (batch_size, hidden_dim) 형태의 히든 벡터.
        """
        # 배치 차원을 제거하고(squeeze) CPU로 이동하여 저장
        self.hidden_states[state].append(hidden_state.squeeze(0).detach().cpu())

    def get_hidden_states_by_state(self, state_type: str) -> list[torch.Tensor]:
        """지정된 상태의 모든 히든 벡터 리스트를 반환합니다."""
        return self.hidden_states.get(state_type, [])

    def get_collected_states(self) -> list[str]:
        """지금까지 수집된 모든 상태의 종류를 리스트로 반환합니다."""
        return list(self.hidden_states.keys())

    def clear(self):
        """수집된 모든 데이터를 초기화하여 다음 분석을 준비합니다."""
        self.hidden_states.clear()

    def __str__(self) -> str:
        """현재까지 수집된 데이터의 요약 정보를 문자열로 반환합니다."""
        summary = "CollapseCollector 현황:\n"
        if not self.hidden_states:
            return summary + "  - 수집된 데이터가 없습니다."
        
        for state, tensors in self.hidden_states.items():
            summary += f"  - 상태 '{state}': {len(tensors)}개 텐서 수집됨\n"
        return summary