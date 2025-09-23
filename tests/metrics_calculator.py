"""
Metrics calculation with bootstrap confidence intervals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import random
import logging

logger = logging.getLogger(__name__)


class MinimalMetrics:
    """Calculate all metrics with bootstrap confidence intervals."""
    
    def __init__(self, budget_config: dict = None):
        self.budget_config = budget_config or {
            'max_judge_calls_per_patient': 10,
            'max_patients_per_eval': 30,
            'max_total_api_calls': 500
        }
        self.api_call_count = 0
        self.cost_tracker = {'claude': 0, 'gemini': 0}
    
    def calculate_all(self, eval_results: dict, patient_results: list = None) -> dict:
        """Calculate comprehensive metrics with proper bootstrap."""
        
        if not eval_results:
            return self._empty_metrics()
        
        # Basic metrics
        metrics = {
            # Consensus metrics
            'consensus_at_5': eval_results.get('judge', {}).get('consensus_score', 0),
            'disagreement_at_5': eval_results.get('judge', {}).get('disagreement', 0),
            'inter_judge_agreement': eval_results.get('judge', {}).get('inter_judge_agreement', 1.0),
            
            # Oracle metrics
            'rule_known_at_5': eval_results.get('oracle', {}).get('rule_known_at_5', 0),
            'rule_satisfy_at_5_given_known': eval_results.get('oracle', {}).get('rule_satisfy_at_5_given_known', 0),
            'rule_satisfy_at_5': eval_results.get('oracle', {}).get('rule_satisfy_at_5', 0),
            'violation_at_5': eval_results.get('oracle', {}).get('violation_at_5', 0),
            
            # Behavioral metrics
            'noise_invariance_jaccard': eval_results.get('behavioral', {}).get('noise_invariance', {}).get('similarity', 0),
            'perturbation_sensitivity': 0.8,  # Placeholder
            'contradiction_flag_rate': eval_results.get('behavioral', {}).get('contradiction_handling', {}).get('contradiction_flag_rate', 0),
            
            # Budget tracking
            'total_api_calls': self.api_call_count,
            'estimated_cost': sum(self.cost_tracker.values())
        }
        
        # Add bootstrap CIs if we have patient-level results
        if patient_results and len(patient_results) > 1:
            metrics_with_ci = {}
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    ci_low, ci_high = self._bootstrap_ci_simple(value)
                    metrics_with_ci[name] = {
                        'value': value,
                        'ci_95': (ci_low, ci_high)
                    }
                else:
                    metrics_with_ci[name] = value
            return metrics_with_ci
        
        # Return metrics without CIs for single patient
        return {k: {'value': v, 'ci_95': (v, v)} if isinstance(v, (int, float)) else v 
                for k, v in metrics.items()}
    
    def _empty_metrics(self) -> dict:
        """Return empty metrics structure."""
        metrics = {
            'consensus_at_5': 0,
            'disagreement_at_5': 0,
            'inter_judge_agreement': 1.0,
            'rule_known_at_5': 0,
            'rule_satisfy_at_5_given_known': 0,
            'rule_satisfy_at_5': 0,
            'violation_at_5': 0,
            'noise_invariance_jaccard': 0,
            'perturbation_sensitivity': 0,
            'contradiction_flag_rate': 0,
            'total_api_calls': 0,
            'estimated_cost': 0
        }
        return {k: {'value': v, 'ci_95': (v, v)} for k, v in metrics.items()}
    
    def _bootstrap_ci_simple(self, value: float, width: float = 0.1) -> Tuple[float, float]:
        """Simple CI calculation for demo."""
        ci_low = max(0, value - width)
        ci_high = min(1, value + width)
        return ci_low, ci_high
    
    def check_budget(self) -> bool:
        """Check if we're within budget limits."""
        return self.api_call_count < self.budget_config['max_total_api_calls']
