"""
Pydantic validators for structured output validation.

Ensures all LLM outputs follow strict contracts to prevent malformed responses.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator, ValidationError
import json
import logging

logger = logging.getLogger(__name__)


class ExpertScoreModel(BaseModel):
    """Validated expert score model."""
    expert_name: str = Field(..., min_length=1, max_length=100)
    score: float = Field(..., ge=0, le=100, description="Score between 0-100")
    subscore_category: str = Field(..., description="Category this expert scores")
    reasoning: str = Field(..., min_length=10, max_length=2000)
    confidence: float = Field(..., ge=0, le=1, description="Confidence between 0-1")
    key_points: List[str] = Field(default_factory=list, max_items=10)
    
    @validator('key_points')
    def validate_key_points(cls, v):
        """Ensure key points are non-empty strings."""
        return [point for point in v if point and isinstance(point, str)]


class SubscoresModel(BaseModel):
    """Validated subscores model."""
    eligibility: float = Field(..., ge=0, le=40, description="Eligibility score (0-40)")
    biomarker: float = Field(..., ge=0, le=30, description="Biomarker score (0-30)")
    clinical: float = Field(..., ge=0, le=20, description="Clinical score (0-20)")
    practical: float = Field(..., ge=0, le=10, description="Practical score (0-10)")
    
    @validator('*')
    def round_scores(cls, v):
        """Round scores to 1 decimal place."""
        return round(v, 1) if v is not None else 0.0
    
    def total(self) -> float:
        """Calculate total score."""
        return self.eligibility + self.biomarker + self.clinical + self.practical


class DeterministicFiltersModel(BaseModel):
    """Validated deterministic filter results."""
    age_ok: bool = Field(..., description="Age eligibility passed")
    gender_ok: bool = Field(..., description="Gender compatibility passed")
    trial_open: bool = Field(..., description="Trial is open/recruiting")
    geography_ok: bool = Field(..., description="Geographic proximity acceptable")
    biomarker_ok: bool = Field(..., description="No biomarker exclusions")


class TrialScoreModel(BaseModel):
    """Validated trial score model with all required fields."""
    trial_id: str = Field(..., min_length=1, max_length=50)
    total_score: float = Field(..., ge=0, le=100)
    subscores: SubscoresModel
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str = Field(..., min_length=20, max_length=5000)
    key_matches: List[str] = Field(default_factory=list, max_items=10)
    concerns: List[str] = Field(default_factory=list, max_items=10)
    deterministic_filters: Optional[DeterministicFiltersModel] = None
    expert_scores: List[ExpertScoreModel] = Field(default_factory=list)
    judge_consolidation: Optional[str] = Field(None, max_length=2000)
    
    @validator('total_score')
    def validate_total_score(cls, v, values):
        """Ensure total score aligns with subscores if present."""
        if 'subscores' in values and values['subscores']:
            expected_total = values['subscores'].total()
            # Allow some deviation due to rounding
            if abs(v - expected_total) > 2.0:
                logger.warning(f"Total score {v} deviates from subscore sum {expected_total}")
        return round(v, 1)
    
    @validator('key_matches', 'concerns')
    def validate_string_lists(cls, v):
        """Ensure lists contain non-empty strings."""
        return [item for item in v if item and isinstance(item, str)]


class LLMResponseValidator:
    """Validates and fixes LLM responses to ensure contract compliance."""
    
    @staticmethod
    def validate_expert_score(response: Dict[str, Any], expert_name: str, subscore_category: str) -> ExpertScoreModel:
        """
        Validate and fix expert score response.
        
        Args:
            response: Raw LLM response dict
            expert_name: Name of the expert
            subscore_category: Category this expert is responsible for
            
        Returns:
            Validated ExpertScoreModel
        """
        try:
            # Add required fields if missing
            response['expert_name'] = expert_name
            response['subscore_category'] = subscore_category
            
            # Fix common issues
            if 'score' not in response or response['score'] is None:
                response['score'] = 50.0  # Default middle score
            
            if 'confidence' not in response or response['confidence'] is None:
                response['confidence'] = 0.5
            
            if 'reasoning' not in response or not response['reasoning']:
                response['reasoning'] = "No detailed reasoning provided"
            
            if 'key_points' not in response:
                response['key_points'] = []
            
            # Validate with Pydantic
            return ExpertScoreModel(**response)
            
        except ValidationError as e:
            logger.error(f"Expert score validation failed: {e}")
            # Return safe default
            return ExpertScoreModel(
                expert_name=expert_name,
                subscore_category=subscore_category,
                score=50.0,
                reasoning="Validation error - using default score",
                confidence=0.0,
                key_points=[]
            )
    
    @staticmethod
    def validate_trial_score(response: Dict[str, Any], trial_id: str) -> TrialScoreModel:
        """
        Validate and fix complete trial score response.
        
        Args:
            response: Raw LLM response dict
            trial_id: Trial identifier
            
        Returns:
            Validated TrialScoreModel
        """
        try:
            # Ensure trial_id
            response['trial_id'] = trial_id
            
            # Fix subscores if missing or invalid
            if 'subscores' not in response or not isinstance(response['subscores'], dict):
                response['subscores'] = {
                    'eligibility': 20.0,
                    'biomarker': 15.0,
                    'clinical': 10.0,
                    'practical': 5.0
                }
            
            # Ensure all subscore fields exist
            subscore_defaults = {
                'eligibility': 20.0,
                'biomarker': 15.0,
                'clinical': 10.0,
                'practical': 5.0
            }
            for key, default in subscore_defaults.items():
                if key not in response['subscores']:
                    response['subscores'][key] = default
            
            # Fix total score if missing
            if 'total_score' not in response or response['total_score'] is None:
                subscores_obj = SubscoresModel(**response['subscores'])
                response['total_score'] = subscores_obj.total()
            
            # Fix confidence if missing
            if 'confidence' not in response or response['confidence'] is None:
                response['confidence'] = 0.5
            
            # Fix reasoning if missing
            if 'reasoning' not in response or not response['reasoning']:
                response['reasoning'] = "Automated scoring based on patient-trial compatibility"
            
            # Ensure lists exist
            if 'key_matches' not in response:
                response['key_matches'] = []
            if 'concerns' not in response:
                response['concerns'] = []
            
            # Validate with Pydantic
            return TrialScoreModel(**response)
            
        except ValidationError as e:
            logger.error(f"Trial score validation failed: {e}")
            # Return safe default
            return TrialScoreModel(
                trial_id=trial_id,
                total_score=50.0,
                subscores=SubscoresModel(
                    eligibility=20.0,
                    biomarker=15.0,
                    clinical=10.0,
                    practical=5.0
                ),
                confidence=0.0,
                reasoning="Validation error - using default scores",
                key_matches=[],
                concerns=["Score validation failed"]
            )
    
    @staticmethod
    def parse_llm_json(response_text: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Safely parse JSON from LLM response.
        
        Args:
            response_text: Raw LLM response text
            default: Default dict to return on parse failure
            
        Returns:
            Parsed JSON dict or default
        """
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return default or {}


class SubscoreAssigner:
    """
    Assigns subscores deterministically based on expert roles.
    
    This prevents hallucination by having each expert own specific subscores.
    """
    
    @staticmethod
    def assign_subscores_from_experts(expert_scores: List[ExpertScoreModel]) -> SubscoresModel:
        """
        Deterministically assign subscores from expert scores.
        
        Mapping:
        - Eligibility ← Medical Expert (40% of their score)
        - Biomarker ← Biomarker Specialist (30% of their score)
        - Practical ← Patient Advocate (10% of their score)
        - Clinical ← Weighted combination (20% split between Medical and Biomarker)
        
        Args:
            expert_scores: List of validated expert scores
            
        Returns:
            SubscoresModel with deterministic subscores
        """
        # Initialize with defaults
        subscores = {
            'eligibility': 20.0,  # Default 50% of max
            'biomarker': 15.0,    # Default 50% of max
            'clinical': 10.0,     # Default 50% of max
            'practical': 5.0      # Default 50% of max
        }
        
        # Find each expert's score
        medical_score = None
        biomarker_score = None
        advocate_score = None
        
        for expert in expert_scores:
            if 'medical' in expert.expert_name.lower():
                medical_score = expert.score
            elif 'biomarker' in expert.expert_name.lower():
                biomarker_score = expert.score
            elif 'advocate' in expert.expert_name.lower():
                advocate_score = expert.score
        
        # Assign subscores based on expert scores
        if medical_score is not None:
            # Medical expert owns eligibility (40 points max)
            subscores['eligibility'] = (medical_score / 100) * 40
        
        if biomarker_score is not None:
            # Biomarker specialist owns biomarker score (30 points max)
            subscores['biomarker'] = (biomarker_score / 100) * 30
        
        if advocate_score is not None:
            # Patient advocate owns practical score (10 points max)
            subscores['practical'] = (advocate_score / 100) * 10
        
        # Clinical score is weighted combination (20 points max)
        if medical_score is not None and biomarker_score is not None:
            # 60% medical, 40% biomarker for clinical appropriateness
            weighted = (medical_score * 0.6 + biomarker_score * 0.4)
            subscores['clinical'] = (weighted / 100) * 20
        elif medical_score is not None:
            subscores['clinical'] = (medical_score / 100) * 20
        elif biomarker_score is not None:
            subscores['clinical'] = (biomarker_score / 100) * 20
        
        return SubscoresModel(**subscores)


# Example usage
if __name__ == "__main__":
    # Test validation
    validator = LLMResponseValidator()
    
    # Test expert score validation
    raw_expert = {
        "score": 85,
        "reasoning": "Good match for patient",
        "confidence": 0.9
    }
    
    validated = validator.validate_expert_score(raw_expert, "Medical Expert", "eligibility")
    print(f"Validated expert score: {validated.model_dump()}")
    
    # Test subscore assignment
    expert_scores = [
        ExpertScoreModel(
            expert_name="Medical Expert",
            subscore_category="eligibility",
            score=80,
            reasoning="Good clinical match",
            confidence=0.8,
            key_points=["Stage appropriate"]
        ),
        ExpertScoreModel(
            expert_name="Biomarker Specialist",
            subscore_category="biomarker",
            score=90,
            reasoning="Excellent biomarker match",
            confidence=0.95,
            key_points=["ER+ match"]
        ),
        ExpertScoreModel(
            expert_name="Patient Advocate",
            subscore_category="practical",
            score=60,
            reasoning="Some travel required",
            confidence=0.7,
            key_points=["Distance concern"]
        )
    ]
    
    subscores = SubscoreAssigner.assign_subscores_from_experts(expert_scores)
    print(f"Assigned subscores: {subscores.model_dump()}")
    print(f"Total: {subscores.total()}")
