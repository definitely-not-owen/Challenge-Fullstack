"""
Hybrid LLM-Powered Clinical Trial Ranker.

This module implements sophisticated trial ranking using deterministic filters
and multiple LLM experts with judge consolidation.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import random

import pandas as pd

# LLM imports
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini not available")

from deterministic_filter import DeterministicFilter, FilteredTrial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    MOCK = "mock"


@dataclass
class ExpertScore:
    """Score from a single expert."""
    expert_name: str
    score: float
    reasoning: str
    confidence: float
    key_points: List[str]


@dataclass
class TrialScore:
    """Complete scoring for a trial."""
    trial_id: str
    total_score: float
    subscores: Dict[str, float]  # eligibility, biomarker, clinical, practical
    confidence: float
    reasoning: str
    key_matches: List[str]
    concerns: List[str]
    deterministic_filters: Dict[str, bool]
    expert_scores: List[ExpertScore]
    judge_consolidation: str


@dataclass 
class RankedTrial:
    """Trial with complete ranking information."""
    trial: Any
    score: TrialScore
    rank: int


class LLMClient:
    """Unified interface for different LLM providers."""
    
    def __init__(self, provider: LLMProvider, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == LLMProvider.OPENAI and OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(api_key=self.api_key or os.getenv('OPENAI_API_KEY'))
        elif self.provider == LLMProvider.GEMINI and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key or os.getenv('GEMINI_API_KEY'))
            self.client = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.client = None
    
    async def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """Generate response from LLM with exponential backoff for rate limiting."""
        max_retries = 5
        base_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                if self.provider == LLMProvider.OPENAI and OPENAI_AVAILABLE:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    response = await self.client.chat.completions.create(
                        model="gpt-5",
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "json_object"}
                    )
                    return response.choices[0].message.content
                    
                elif self.provider == LLMProvider.GEMINI and GEMINI_AVAILABLE:
                    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    response = await asyncio.to_thread(
                        self.client.generate_content,
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            response_mime_type="application/json"
                        )
                    )
                    return response.text
                else:
                    # Mock response for testing
                    return json.dumps({
                        "score": 75.0,
                        "reasoning": "Mock scoring for testing",
                        "confidence": 0.8,
                        "key_points": ["Test match"]
                    })
                    
            except Exception as e:
                # Check if it's a rate limit error
                is_rate_limit = False
                if hasattr(e, 'status_code') and e.status_code == 429:
                    is_rate_limit = True
                elif 'rate' in str(e).lower() or '429' in str(e):
                    is_rate_limit = True
                
                if is_rate_limit and attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                elif attempt == max_retries - 1:
                    logger.error(f"Max retries reached for LLM call: {e}")
                    # Return a default response instead of failing
                    return json.dumps({
                        "score": 50.0,
                        "reasoning": "Error in LLM evaluation - using default score",
                        "confidence": 0.1,
                        "key_points": ["Error occurred"]
                    })
                else:
                    # Non-rate limit error, retry with smaller delay
                    delay = 0.5 * (attempt + 1)
                    logger.warning(f"LLM error: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)


class ExpertPrompts:
    """Specialized prompts for different expert perspectives."""
    
    @staticmethod
    def medical_expert() -> str:
        """Prompt for medical/clinical expert perspective."""
        return """You are a senior oncologist evaluating clinical trial eligibility.
        
Focus on:
- Clinical stage appropriateness
- Performance status requirements
- Prior therapy considerations
- Safety concerns
- Treatment sequencing logic

Provide a score (0-100) based on medical appropriateness."""

    @staticmethod
    def biomarker_specialist() -> str:
        """Prompt for molecular/biomarker expert perspective."""
        return """You are a molecular oncologist specializing in precision medicine.
        
Focus on:
- Biomarker matching (detected vs required)
- Molecular exclusions
- Actionable mutations
- Companion diagnostics
- Exploratory vs mandatory biomarkers

Provide a score (0-100) based on molecular compatibility."""

    @staticmethod
    def patient_advocate() -> str:
        """Prompt for patient quality-of-life perspective."""
        return """You are a patient advocate evaluating trial feasibility and burden.
        
Focus on:
- Travel requirements
- Visit frequency
- Quality of life impact
- Support requirements
- Practical barriers

Provide a score (0-100) based on patient feasibility."""

    @staticmethod
    def consolidation_judge() -> str:
        """Prompt for meta-judge consolidation."""
        return """You are a clinical trial matching expert consolidating multiple expert opinions.
        
Given the scores and reasoning from:
1. Medical Expert
2. Biomarker Specialist  
3. Patient Advocate

Provide:
- Final consolidated score (0-100)
- Weighted reasoning considering all perspectives
- Key matches and concerns
- Confidence level in the match"""


class HybridTrialRanker:
    """
    Hybrid deterministic + LLM trial ranking system.
    
    Combines rule-based filtering with sophisticated LLM scoring
    using mixture-of-experts and judge consolidation.
    """
    
    def __init__(
        self,
        openai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        use_mixture_of_experts: bool = True
    ):
        """
        Initialize the hybrid ranker.
        
        Args:
            openai_key: OpenAI API key
            gemini_key: Google Gemini API key
            use_mixture_of_experts: Whether to use multiple expert perspectives
        """
        self.deterministic_filter = DeterministicFilter()
        self.use_mixture_of_experts = use_mixture_of_experts
        
        # Initialize LLM clients
        self.openai_client = None
        self.gemini_client = None
        
        if openai_key or os.getenv('OPENAI_API_KEY'):
            self.openai_client = LLMClient(LLMProvider.OPENAI, openai_key)
            logger.info("OpenAI client initialized")
        
        if gemini_key or os.getenv('GEMINI_API_KEY'):
            self.gemini_client = LLMClient(LLMProvider.GEMINI, gemini_key)
            logger.info("Gemini client initialized")
        
        if not self.openai_client and not self.gemini_client:
            logger.warning("No LLM clients available, using mock scoring")
            self.mock_client = LLMClient(LLMProvider.MOCK)
    
    async def rank_trials(
        self,
        patient: Dict[str, Any],
        trials: List[Any],
        max_trials: Optional[int] = None
    ) -> List[RankedTrial]:
        """
        Rank trials for a patient using hybrid approach.
        
        Args:
            patient: Patient data dictionary
            trials: List of trial objects
            max_trials: Maximum number of trials to return
            
        Returns:
            List of ranked trials with scores
        """
        logger.info(f"Starting hybrid ranking for {len(trials)} trials")
        
        # Step 1: Apply deterministic filters
        passed_trials, filter_results = self.deterministic_filter.apply_filters(patient, trials)
        
        if not passed_trials:
            logger.warning("No trials passed deterministic filters")
            return []
        
        logger.info(f"{len(passed_trials)} trials passed deterministic filters")
        
        # Step 2: Score trials with LLM
        scored_trials = []
        for trial in passed_trials:
            # Get filter decisions for this trial
            filter_decision = next(
                (fr for fr in filter_results if fr.trial == trial),
                None
            )
            
            score = await self._score_trial(patient, trial, filter_decision)
            scored_trials.append((trial, score))
        
        # Step 3: Sort by score
        scored_trials.sort(key=lambda x: x[1].total_score, reverse=True)
        
        # Step 4: Create ranked results
        ranked_results = []
        for rank, (trial, score) in enumerate(scored_trials[:max_trials], 1):
            ranked_results.append(RankedTrial(
                trial=trial,
                score=score,
                rank=rank
            ))
        
        logger.info(f"Ranking complete: {len(ranked_results)} trials ranked")
        return ranked_results
    
    async def _score_trial(
        self,
        patient: Dict[str, Any],
        trial: Any,
        filter_decision: Optional[FilteredTrial]
    ) -> TrialScore:
        """Score a single trial using LLM(s)."""
        
        # Get deterministic filter summary
        filter_summary = {}
        if filter_decision:
            filter_summary = filter_decision.get_filter_summary()
        
        if self.use_mixture_of_experts:
            # Use multiple experts
            expert_scores = await self._get_expert_scores(patient, trial)
            
            # Consolidate with judge
            final_score = await self._consolidate_scores(
                patient, trial, expert_scores, filter_summary
            )
        else:
            # Single LLM scoring
            final_score = await self._simple_score(patient, trial, filter_summary)
        
        return final_score
    
    async def _get_expert_scores(
        self,
        patient: Dict[str, Any],
        trial: Any
    ) -> List[ExpertScore]:
        """Get scores from multiple expert perspectives."""
        expert_configs = [
            ("Medical Expert", ExpertPrompts.medical_expert()),
            ("Biomarker Specialist", ExpertPrompts.biomarker_specialist()),
            ("Patient Advocate", ExpertPrompts.patient_advocate())
        ]
        
        # Run expert evaluations sequentially with delays to avoid rate limits
        expert_scores = []
        for i, (expert_name, system_prompt) in enumerate(expert_configs):
            if i > 0:
                # Small delay between expert calls to avoid rate limits
                await asyncio.sleep(0.5)
            
            score = await self._get_single_expert_score(
                patient, trial, expert_name, system_prompt
            )
            expert_scores.append(score)
        
        return expert_scores
    
    async def _get_single_expert_score(
        self,
        patient: Dict[str, Any],
        trial: Any,
        expert_name: str,
        system_prompt: str
    ) -> ExpertScore:
        """Get score from a single expert perspective."""
        
        # Create prompt
        prompt = self._create_scoring_prompt(patient, trial)
        
        # Choose LLM client (alternate between available clients)
        client = self._select_llm_client()
        
        try:
            response = await client.generate(prompt, system_prompt, temperature=0.7)
            result = json.loads(response)
            
            return ExpertScore(
                expert_name=expert_name,
                score=float(result.get('score', 50)),
                reasoning=result.get('reasoning', ''),
                confidence=float(result.get('confidence', 0.5)),
                key_points=result.get('key_points', [])
            )
        except Exception as e:
            logger.error(f"Error getting {expert_name} score: {e}")
            return ExpertScore(
                expert_name=expert_name,
                score=50.0,
                reasoning=f"Error in scoring: {str(e)}",
                confidence=0.0,
                key_points=[]
            )
    
    async def _consolidate_scores(
        self,
        patient: Dict[str, Any],
        trial: Any,
        expert_scores: List[ExpertScore],
        filter_summary: Dict[str, bool]
    ) -> TrialScore:
        """Consolidate multiple expert scores with judge."""
        
        # Prepare consolidation prompt
        consolidation_data = {
            "patient_summary": self._summarize_patient(patient),
            "trial_summary": self._summarize_trial(trial),
            "expert_scores": [
                {
                    "expert": score.expert_name,
                    "score": score.score,
                    "reasoning": score.reasoning,
                    "confidence": score.confidence
                }
                for score in expert_scores
            ],
            "deterministic_filters": filter_summary
        }
        
        prompt = f"""Consolidate these expert evaluations into a final score:

{json.dumps(consolidation_data, indent=2)}

Provide a JSON response with:
- total_score: weighted final score (0-100)
- subscores: dict with eligibility, biomarker, clinical, practical scores
- confidence: overall confidence (0-1)
- reasoning: consolidated reasoning
- key_matches: list of key positive factors
- concerns: list of concerns
- judge_consolidation: your consolidation reasoning"""
        
        # Use best available LLM for judge (prefer Gemini for consolidation)
        client = self.gemini_client or self.openai_client or self.mock_client
        
        try:
            response = await client.generate(
                prompt,
                ExpertPrompts.consolidation_judge(),
                temperature=0.5
            )
            result = json.loads(response)
            
            return TrialScore(
                trial_id=getattr(trial, 'nct_id', 'unknown'),
                total_score=float(result.get('total_score', 50)),
                subscores=result.get('subscores', {}),
                confidence=float(result.get('confidence', 0.5)),
                reasoning=result.get('reasoning', ''),
                key_matches=result.get('key_matches', []),
                concerns=result.get('concerns', []),
                deterministic_filters=filter_summary,
                expert_scores=expert_scores,
                judge_consolidation=result.get('judge_consolidation', '')
            )
        except Exception as e:
            logger.error(f"Error in score consolidation: {e}")
            # Fallback to average
            avg_score = sum(s.score for s in expert_scores) / len(expert_scores)
            return TrialScore(
                trial_id=getattr(trial, 'nct_id', 'unknown'),
                total_score=avg_score,
                subscores={},
                confidence=0.3,
                reasoning="Consolidation error - using average score",
                key_matches=[],
                concerns=["Consolidation failed"],
                deterministic_filters=filter_summary,
                expert_scores=expert_scores,
                judge_consolidation=str(e)
            )
    
    async def _simple_score(
        self,
        patient: Dict[str, Any],
        trial: Any,
        filter_summary: Dict[str, bool]
    ) -> TrialScore:
        """Simple single-LLM scoring without mixture of experts."""
        prompt = self._create_scoring_prompt(patient, trial)
        
        system_prompt = """You are a clinical trial matching expert.
Score this trial match from 0-100 considering:
- Eligibility (40 points)
- Biomarker alignment (30 points)  
- Clinical appropriateness (20 points)
- Practical factors (10 points)

Provide JSON with: total_score, subscores, confidence, reasoning, key_matches, concerns"""
        
        client = self._select_llm_client()
        
        try:
            response = await client.generate(prompt, system_prompt)
            result = json.loads(response)
            
            return TrialScore(
                trial_id=getattr(trial, 'nct_id', 'unknown'),
                total_score=float(result.get('total_score', 50)),
                subscores=result.get('subscores', {}),
                confidence=float(result.get('confidence', 0.5)),
                reasoning=result.get('reasoning', ''),
                key_matches=result.get('key_matches', []),
                concerns=result.get('concerns', []),
                deterministic_filters=filter_summary,
                expert_scores=[],
                judge_consolidation=""
            )
        except Exception as e:
            logger.error(f"Error in simple scoring: {e}")
            return TrialScore(
                trial_id=getattr(trial, 'nct_id', 'unknown'),
                total_score=50.0,
                subscores={},
                confidence=0.0,
                reasoning=f"Scoring error: {str(e)}",
                key_matches=[],
                concerns=["Scoring failed"],
                deterministic_filters=filter_summary,
                expert_scores=[],
                judge_consolidation=""
            )
    
    def _create_scoring_prompt(self, patient: Dict[str, Any], trial: Any) -> str:
        """Create prompt for scoring a trial."""
        return f"""Evaluate this clinical trial match:

PATIENT:
- Age: {patient.get('age')}
- Gender: {patient.get('gender')}
- Cancer: {patient.get('cancer_type')} Stage {patient.get('cancer_stage')}
- Biomarkers Detected: {patient.get('biomarkers_detected', 'None')}
- Biomarkers Ruled Out: {patient.get('biomarkers_ruled_out', 'None')}
- ECOG Status: {patient.get('ecog_status', 'Unknown')}
- Prior Treatments: {patient.get('previous_treatments', 'None')}
- Location: {patient.get('city')}, {patient.get('state')}

TRIAL:
- NCT ID: {getattr(trial, 'nct_id', 'Unknown')}
- Title: {getattr(trial, 'title', 'Unknown')}
- Phase: {getattr(trial, 'phase', 'Unknown')}
- Status: {getattr(trial, 'status', 'Unknown')}
- Summary: {getattr(trial, 'brief_summary', 'No summary')[:500]}
- Eligibility Age: {getattr(trial, 'min_age', 'N/A')}-{getattr(trial, 'max_age', 'N/A')}
- Gender: {getattr(trial, 'gender', 'All')}

Evaluate the match quality and provide a structured JSON score."""
    
    def _summarize_patient(self, patient: Dict[str, Any]) -> str:
        """Create concise patient summary."""
        return (
            f"{patient.get('age')}yo {patient.get('gender')} with "
            f"{patient.get('cancer_type')} stage {patient.get('cancer_stage')}, "
            f"biomarkers: {patient.get('biomarkers_detected', 'none')}"
        )
    
    def _summarize_trial(self, trial: Any) -> str:
        """Create concise trial summary."""
        return (
            f"{getattr(trial, 'nct_id', 'Unknown')}: "
            f"{getattr(trial, 'title', 'Unknown')[:100]}"
        )
    
    def _select_llm_client(self) -> LLMClient:
        """Select best available LLM client."""
        # Prefer OpenAI, then Gemini, then mock
        if self.openai_client:
            return self.openai_client
        elif self.gemini_client:
            return self.gemini_client
        else:
            return self.mock_client


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_ranker():
        """Test the hybrid ranker."""
        ranker = HybridTrialRanker()
        
        # Mock patient
        patient = {
            'age': 65,
            'gender': 'Female',
            'cancer_type': 'Breast Cancer',
            'cancer_stage': 'II',
            'biomarkers_detected': 'ER+, PR+',
            'biomarkers_ruled_out': 'HER2+',
            'state': 'California'
        }
        
        print(f"Hybrid Trial Ranker initialized")
        print(f"Test patient: {patient['age']}yo {patient['gender']} with {patient['cancer_type']}")
        print(f"Using mixture of experts: {ranker.use_mixture_of_experts}")
    
    asyncio.run(test_ranker())
