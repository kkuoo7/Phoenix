import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

class CollapseAnalyzer:
    """Collapse 메트릭 분석을 위한 클래스"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_collapse_metrics(self, metrics_data):
        """collapse 메트릭 분석"""
        if not metrics_data:
            return None
        
        analysis = {
            "summary": self._compute_summary(metrics_data),
            "token_analysis": self._analyze_token_metrics(metrics_data),
            "overall_analysis": self._analyze_overall_metrics(metrics_data),
            "chunk_analysis": self._analyze_chunk_metrics(metrics_data)
        }
        
        return analysis
    
    def _compute_summary(self, metrics_data):
        """전체 요약 통계"""
        total_tokens = metrics_data.get("total_tokens", 0)
        token_metrics = metrics_data.get("token_metrics", {})
        
        if not token_metrics:
            return {"status": "no_data", "message": "No token metrics available"}
        
        # 통계 계산
        svd_entropies = [m['svd_entropy'] for m in token_metrics.values()]
        sample_counts = [m['num_samples'] for m in token_metrics.values()]
        
        summary = {
            "total_tokens_analyzed": total_tokens,
            "tokens_with_metrics": len(token_metrics),
            "avg_svd_entropy": np.mean(svd_entropies),
            "std_svd_entropy": np.std(svd_entropies),
            "avg_samples_per_token": np.mean(sample_counts),
            "min_samples": min(sample_counts),
            "max_samples": max(sample_counts)
        }
        
        return summary
    
    def _analyze_token_metrics(self, metrics_data):
        """토큰별 메트릭 분석"""
        token_metrics = metrics_data.get("token_metrics", {})
        
        if not token_metrics:
            return {}
        
        # 토큰별 분석
        token_analysis = {}
        for token_id, metrics in token_metrics.items():
            svd_entropy = metrics['svd_entropy']
            num_samples = metrics['num_samples']
            
            token_analysis[token_id] = {
                "svd_entropy": svd_entropy,
                "num_samples": num_samples
            }
        
        return token_analysis
    
    def _analyze_overall_metrics(self, metrics_data):
        """전체 메트릭 분석"""
        overall_metrics = metrics_data.get("overall_metrics", {})
        
        if not overall_metrics:
            return {}
        
        svd_entropy = overall_metrics.get('svd_entropy', 0)
        gnc2 = overall_metrics.get('gnc2', 0)
        unc3 = overall_metrics.get('unc3', 0)
        
        overall_analysis = {
            "overall_svd_entropy": svd_entropy,
            "overall_gnc2": gnc2,
            "overall_unc3": unc3
        }
        
        return overall_analysis
    
    def _analyze_chunk_metrics(self, metrics_data):
        """Chunk-wise 메트릭 분석"""
        chunk_metrics = metrics_data.get("chunk_metrics", {})
        
        if not chunk_metrics:
            return {}
        
        chunk_entropies = chunk_metrics.get("chunk_svd_entropies", [])
        num_chunks = chunk_metrics.get("num_chunks", 0)
        avg_svd_entropy = chunk_metrics.get("avg_svd_entropy", 0)
        std_svd_entropy = chunk_metrics.get("std_svd_entropy", 0)
        
        # Collapse 패턴 분석
        collapse_pattern = "unknown"
        if len(chunk_entropies) >= 2:
            # 첫 번째와 마지막 청크의 엔트로피 비교
            first_entropy = chunk_entropies[0]
            last_entropy = chunk_entropies[-1]
            entropy_decline = first_entropy - last_entropy
            
            if entropy_decline > 0.5:
                collapse_pattern = "strong_decline"
            elif entropy_decline > 0.1:
                collapse_pattern = "moderate_decline"
            elif entropy_decline > 0.01:
                collapse_pattern = "weak_decline"
            else:
                collapse_pattern = "no_decline"
        
        chunk_analysis = {
            "num_chunks": num_chunks,
            "avg_svd_entropy": avg_svd_entropy,
            "std_svd_entropy": std_svd_entropy,
            "chunk_entropies": chunk_entropies,
            "collapse_pattern": collapse_pattern
        }
        
        return chunk_analysis
    
    def save_analysis(self, analysis, output_file):
        """분석 결과 저장"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Analysis saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
    
    def generate_report(self, metrics_data, output_file):
        """완전한 분석 리포트 생성"""
        analysis = self.analyze_collapse_metrics(metrics_data)
        if analysis:
            self.save_analysis(analysis, output_file)
            return analysis
        return None 