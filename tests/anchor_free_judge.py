"""
Cross-model anchor-free judging for independent evaluation.
"""

import json
import hashlib
import numpy as np
from typing import Dict, List, Optional
import logging
import os
import asyncio
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Import LLM clients
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Anthropic not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google Gemini not available")


class AnchorFreeJudge:
    """Independent evaluation without seeing our scores/reasoning."""
    
    def __init__(self, strict_anchor_free: bool = True):
        self.strict_anchor_free = strict_anchor_free
        self.cache = {}
        
        # Initialize LLM clients
        self.claude_client = None
        self.gemini_model = None
        
        # Initialize Claude
        claude_key = os.getenv('CLAUDE_API_KEY')
        if CLAUDE_AVAILABLE and claude_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=claude_key)
                logger.info("Claude client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude: {e}")
        
        # Initialize Gemini
        gemini_key = os.getenv('GEMINI_API_KEY')
        if GEMINI_AVAILABLE and gemini_key:
            try:
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        
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

Return ONLY a JSON object with these exact fields:
{{
    "clinical_appropriateness": <number 0-10>,
    "biomarker_alignment": <number 0-10>,
    "practical_feasibility": <number 0-10>,
    "overall_match_quality": <number 0-10>
}}
"""
        self.prompt_hash = hashlib.md5(self.prompt_template.encode()).hexdigest()
        
    async def evaluate(self, patient: dict, trial: dict) -> dict:
        """Get independent scores from multiple judges."""
        # Strip anchors if strict mode
        if self.strict_anchor_free:
            patient = self._strip_anchors(patient)
            trial = self._strip_anchors(trial)
        
        # Prepare prompt
        patient_summary = self._summarize_patient(patient)
        trial_summary = self._summarize_trial(trial)
        
        prompt = self.prompt_template.format(
            patient_summary=patient_summary,
            trial_summary=trial_summary
        )
        
        scores = {}
        
        # Get Claude evaluation
        if self.claude_client:
            try:
                claude_scores = await self._get_claude_scores(prompt)
                scores['claude'] = claude_scores
            except Exception as e:
                logger.error(f"Claude evaluation failed: {e}")
                scores['claude'] = self._default_scores()
        else:
            scores['claude'] = self._default_scores()
        
        # Get Gemini evaluation
        if self.gemini_model:
            try:
                gemini_scores = await self._get_gemini_scores(prompt)
                scores['gemini'] = gemini_scores
            except Exception as e:
                logger.error(f"Gemini evaluation failed: {e}")
                scores['gemini'] = self._default_scores()
        else:
            scores['gemini'] = self._default_scores()
        
        # Calculate consensus
        overall_scores = [s['overall_match_quality'] for s in scores.values()]
        
        return {
            'consensus_score': np.mean(overall_scores),
            'disagreement': np.std(overall_scores),
            'individual_scores': scores,
            'prompt_hash': self.prompt_hash
        }
    
    async def _get_claude_scores(self, prompt: str) -> dict:
        """Get scores from Claude."""
        try:
            response = await asyncio.to_thread(
                self.claude_client.messages.create,
                model="claude-3-haiku-20240307",  # Faster and cheaper
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            # Find JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    scores = json.loads(json_match.group())
                    # Validate and clean scores
                    return self._validate_scores(scores)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from Claude: {e}")
                    return self._default_scores()
            else:
                logger.warning(f"No JSON found in Claude response: {content[:200]}")
                return self._default_scores()
                
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._default_scores()
    
    async def _get_gemini_scores(self, prompt: str) -> dict:
        """Get scores from Gemini."""
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt
            )
            
            # Extract JSON from response
            content = response.text
            # Find JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    scores = json.loads(json_match.group())
                    # Validate and clean scores
                    return self._validate_scores(scores)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from Gemini: {e}")
                    return self._default_scores()
            else:
                logger.warning(f"No JSON found in Gemini response: {content[:200]}")
                return self._default_scores()
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._default_scores()
    
    def _default_scores(self) -> dict:
        """Return default scores when API fails."""
        return {
            'clinical_appropriateness': 5.0,
            'biomarker_alignment': 5.0,
            'practical_feasibility': 5.0,
            'overall_match_quality': 5.0
        }
    
    def _validate_scores(self, scores: dict) -> dict:
        """Validate and clean scores from LLM response."""
        required_fields = [
            'clinical_appropriateness',
            'biomarker_alignment',
            'practical_feasibility',
            'overall_match_quality'
        ]
        
        validated = {}
        for field in required_fields:
            if field in scores:
                try:
                    # Convert to float and clip to 0-10 range
                    value = float(scores[field])
                    validated[field] = max(0.0, min(10.0, value))
                except (ValueError, TypeError):
                    validated[field] = 5.0
            else:
                validated[field] = 5.0
        
        return validated
    
    def _summarize_patient(self, patient: dict) -> str:
        """Create patient summary for evaluation."""
        summary = f"""
Age: {patient.get('age', 'Unknown')}
Gender: {patient.get('gender', 'Unknown')}
Cancer Type: {patient.get('cancer_type', 'Unknown')}
Stage: {patient.get('stage', 'Unknown')}
ECOG Status: {patient.get('ecog_status', 'Unknown')}
Biomarkers Detected: {', '.join(patient.get('biomarkers_detected', [])) or 'None'}
Prior Treatments: {patient.get('prior_treatments', 'Unknown')}
Location: {patient.get('city', 'Unknown')}, {patient.get('state', 'Unknown')}
"""
        return summary.strip()
    
    def _summarize_trial(self, trial: dict) -> str:
        """Create trial summary for evaluation."""
        summary = f"""
Trial ID: {trial.get('nct_id', 'Unknown')}
Title: {trial.get('title', 'Unknown')}
Phase: {trial.get('phase', 'Unknown')}
Status: {trial.get('status', 'Unknown')}
Conditions: {trial.get('conditions', 'Unknown')}
Interventions: {trial.get('interventions', 'Unknown')}
Eligibility: Ages {trial.get('minimum_age', '?')} to {trial.get('maximum_age', '?')}, Gender: {trial.get('gender', 'All')}
"""
        return summary.strip()
    
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
