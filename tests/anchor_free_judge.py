"""
Cross-model anchor-free judging for independent evaluation.
"""

import json
import hashlib
import numpy as np
from typing import Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


class AnchorFreeJudge:
    """Independent evaluation without seeing our scores/reasoning."""
    
    def __init__(self, strict_anchor_free: bool = True):
        self.strict_anchor_free = strict_anchor_free
        self.cache = {}
        
        # Store prompt hash for reproducibility
        self.prompt_template = """
Evaluate this clinical trial match objectively:

Patient Summary:
{patient_summary}

Trial Summary:
{trial_summary}

Score each dimension (0-10):
- Clinical Appropriateness: How medically suitable is this trial?
- Biomarker Alignment: How well do molecular markers match?
- Practical Feasibility: Can the patient realistically participate?
- Overall Match Quality: Overall assessment

Return JSON:
{{
    "clinical_appropriateness": <0-10>,
    "biomarker_alignment": <0-10>,
    "practical_feasibility": <0-10>,
    "overall_match_quality": <0-10>
}}
"""
        self.prompt_hash = hashlib.md5(self.prompt_template.encode()).hexdigest()
        
    async def evaluate(self, patient: dict, trial: dict) -> dict:
        """Get independent scores from multiple judges."""
        # For now, simulate with mock scores
        # In production, would call Claude and Gemini APIs
        
        # Strip anchors if strict mode
        if self.strict_anchor_free:
            patient = self._strip_anchors(patient)
            trial = self._strip_anchors(trial)
        
        # Mock evaluation (replace with real API calls)
        claude_scores = {
            'clinical_appropriateness': 7.5,
            'biomarker_alignment': 8.0,
            'practical_feasibility': 6.5,
            'overall_match_quality': 7.3
        }
        
        gemini_scores = {
            'clinical_appropriateness': 7.8,
            'biomarker_alignment': 7.5,
            'practical_feasibility': 7.0,
            'overall_match_quality': 7.4
        }
        
        scores = {'claude': claude_scores, 'gemini': gemini_scores}
        overall_scores = [s['overall_match_quality'] for s in scores.values()]
        
        return {
            'consensus_score': np.mean(overall_scores),
            'disagreement': np.std(overall_scores),
            'individual_scores': scores,
            'prompt_hash': self.prompt_hash
        }
    
    def _strip_anchors(self, data: dict) -> dict:
        """Remove any fields that could anchor the judge."""
        anchor_fields = ['score', 'ranking', 'reasoning', 'our_score', 
                        'confidence', 'subscores']
        return {k: v for k, v in data.items() if k not in anchor_fields}
    
    def calculate_global_agreement(self, all_evaluations: list) -> float:
        """Calculate inter-judge agreement across multiple patient-trial pairs."""
        if len(all_evaluations) < 2:
            return 1.0
        
        # Extract scores by judge
        judge_scores = {'claude': [], 'gemini': []}
        
        for eval in all_evaluations:
            if 'individual_scores' in eval:
                for judge, scores in eval['individual_scores'].items():
                    judge_scores[judge].append(scores['overall_match_quality'])
        
        # Calculate correlation between judges
        if len(judge_scores['claude']) > 1 and len(judge_scores['gemini']) > 1:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(judge_scores['claude'], judge_scores['gemini'])
            return corr if not np.isnan(corr) else 1.0
        
        return 1.0
