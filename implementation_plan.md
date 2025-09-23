# Clinical Trial Matching System - Implementation Plan

## Executive Summary

This document outlines the technical implementation plan for a clinical trial matching system that integrates BioMCP for trial fetching, LLM-powered ranking, and a comprehensive evaluation suite. The system will match patients from `patients.csv` to relevant clinical trials based on clinical criteria, biomarkers, and eligibility requirements.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Patient Data  │    │   BioMCP API    │    │   LLM Ranker    │
│   Processor     │    │   Integration   │    │   (OpenAI)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Trial Matching         │
                    │    Engine                 │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    Evaluation Suite       │
                    │    (LLM-as-Judge)         │
                    └───────────────────────────┘
```

## Component Specifications

### 1. Trial Fetcher (`src/trial_fetcher.py`)

**Objective**: Fetch clinical trials using BioMCP API with intelligent filtering

**Technical Implementation**:
```python
class BioMCPTrialFetcher:
    def __init__(self, api_key: str):
        self.client = BioMCPClient(api_key=api_key)
        self.cache = TrialCache()
    
    async def fetch_trials_for_cancer_type(self, cancer_type: str, 
                                         stage: str = None,
                                         biomarkers: List[str] = None) -> List[Trial]:
        """Fetch trials filtered by cancer type and clinical criteria"""
        
    async def fetch_trials_by_biomarkers(self, biomarkers: List[str]) -> List[Trial]:
        """Fetch trials matching specific biomarkers"""
        
    def normalize_trial_data(self, raw_trial: dict) -> Trial:
        """Normalize trial data to standard format"""
```

**Key Features**:
- **Smart Filtering**: Filter by cancer type, stage, biomarkers, recruitment status
- **Caching**: Cache trial data to reduce API calls
- **Rate Limiting**: Respect API limits with exponential backoff
- **Data Normalization**: Standardize trial data format

**BioMCP Integration Strategy**:
- Use BioMCP's clinical trials search API
- Leverage gene/variant tools for biomarker matching
- Implement fallback to ClinicalTrials.gov if BioMCP fails

### 2. Patient Data Processor (`src/patient_processor.py`)

**Objective**: Process and normalize patient data for matching

**Technical Implementation**:
```python
class PatientProcessor:
    def __init__(self):
        self.biomarker_normalizer = BiomarkerNormalizer()
        self.stage_mapper = CancerStageMapper()
    
    def load_patients(self, csv_path: str) -> List[Patient]:
        """Load and validate patient data from CSV"""
        
    def normalize_patient_data(self, patient: dict) -> Patient:
        """Normalize patient data for matching"""
        
    def extract_biomarkers(self, patient: dict) -> List[Biomarker]:
        """Extract and normalize biomarkers"""
        
    def map_cancer_stage(self, cancer_type: str, stage: str) -> CancerStage:
        """Map cancer stage to standard format"""
```

**Key Features**:
- **Data Validation**: Validate patient data completeness and format
- **Biomarker Normalization**: Use BioMCP to normalize biomarker names
- **Stage Mapping**: Map cancer stages to standard terminology
- **Demographic Processing**: Normalize age, gender, race data

### 3. Hybrid LLM Ranker (`src/ranker.py`)

**Objective**: Hybrid deterministic + LLM approach for intelligent trial ranking

**Technical Implementation**:
```python
class HybridTrialRanker:
    def __init__(self, llm_config: dict):
        self.deterministic_filter = DeterministicFilter()
        self.llm_ranker = LLMRanker(llm_config)
        self.mixture_of_experts = MixtureOfExperts()
    
    async def rank_trials(self, patient: Patient, trials: List[Trial]) -> List[RankedTrial]:
        """Hybrid ranking: deterministic filters + LLM scoring"""
        # Step 1: Apply deterministic filters
        filtered_trials = self.deterministic_filter.apply(patient, trials)
        
        # Step 2: LLM scoring with multiple experts
        scored_trials = await self.mixture_of_experts.score(patient, filtered_trials)
        
        # Step 3: Sort and return
        return sorted(scored_trials, key=lambda x: x.total_score, reverse=True)
```

**Hybrid Scoring System** (100 points):
1. **Eligibility Match** (40 pts)
   - Deterministic: Age, gender, trial status (20 pts)
   - LLM-assisted: Performance status, nuanced criteria (20 pts)

2. **Biomarker Alignment** (30 pts)
   - Deterministic: Hard exclusions (10 pts)
   - LLM-assisted: Exploratory vs mandatory markers (20 pts)

3. **Clinical Appropriateness** (20 pts)
   - LLM-assisted: Phase vs stage, prior therapy logic

4. **Practical Factors** (10 pts)
   - Deterministic: Geography, trial open/closed (5 pts)
   - LLM-assisted: Feasibility nuances (5 pts)

**Mixture-of-Experts Strategy**:
- **Medical Expert**: Eligibility nuance
- **Biomarker Specialist**: Molecular matching
- **Patient Advocate**: Quality of life/practicality
- **LLM Judge**: Meta-model consolidation (Gemini 2.5 Pro)

**Latest Frontier Models**:
- GPT-5 / Claude 4.1 Sonnet / Gemini 2.5 Flash for scoring
- Parallel API calls for speed
- Result consolidation via judge model

### 4. Evaluation Suite (`tests/eval.py`)

**Objective**: Comprehensive evaluation system using multiple approaches

**Technical Implementation**:
```python
class EvaluationSuite:
    def __init__(self):
        self.llm_judge = LLMJudge()
        self.biomarker_validator = BiomarkerValidator()
        self.clinical_validator = ClinicalValidator()
    
    async def evaluate_matching_quality(self, patient: Patient, 
                                      ranked_trials: List[RankedTrial]) -> EvaluationResult:
        """Comprehensive evaluation of matching quality"""
        
    def llm_as_judge_evaluation(self, patient: Patient, 
                               ranked_trials: List[RankedTrial]) -> float:
        """Use LLM to evaluate match quality"""
        
    def biomarker_matching_validation(self, patient: Patient, 
                                    trials: List[Trial]) -> Dict[str, float]:
        """Validate biomarker matching accuracy"""
        
    def clinical_logic_validation(self, patient: Patient, 
                                trials: List[Trial]) -> Dict[str, bool]:
        """Validate clinical appropriateness"""
```

**Evaluation Approaches**:

1. **LLM-as-Judge** (Primary):
   - Use GPT-5 to evaluate match quality
   - Generate synthetic ground truth rankings
   - Compare against expert clinical reasoning

2. **Biomarker Validation**:
   - Validate biomarker compatibility
   - Check for contraindications
   - Score molecular matching accuracy

3. **Clinical Logic Testing**:
   - Test stage-appropriate trial selection
   - Validate age/gender eligibility
   - Check treatment intent alignment

4. **Synthetic Data Generation**:
   - Generate edge cases for testing
   - Create challenging scenarios
   - Test system robustness

**Metrics**:
- **Precision@K**: Top-K trial relevance
- **NDCG**: Ranking quality
- **Clinical Relevance Score**: Expert-validated scoring
- **Biomarker Match Rate**: Molecular compatibility

### 5. Main Interface (`src/match.py`)

**Objective**: Command-line interface for the matching system

**Technical Implementation**:
```python
async def main():
    parser = argparse.ArgumentParser(description='Clinical Trial Matcher')
    parser.add_argument('--patient_id', required=True, help='Patient ID to match')
    parser.add_argument('--max_trials', type=int, default=10, help='Max trials to return')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    
    # Initialize components
    trial_fetcher = BioMCPTrialFetcher(api_key=os.getenv('BIOMCP_API_KEY'))
    patient_processor = PatientProcessor()
    ranker = LLMTrialRanker(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Load patient data
    patients = patient_processor.load_patients('patients.csv')
    patient = next(p for p in patients if p.id == args.patient_id)
    
    # Fetch and rank trials
    trials = await trial_fetcher.fetch_trials_for_patient(patient)
    ranked_trials = await ranker.rank_trials(patient, trials)
    
    # Output results
    print(f"Top {args.max_trials} trials for patient {patient.id}:")
    for i, trial in enumerate(ranked_trials[:args.max_trials]):
        print(f"{i+1}. {trial.nct_id} - Score: {trial.score:.3f}")
        print(f"   {trial.title}")
        print(f"   Match reasons: {trial.match_reasons}")
```

## Data Models

### Patient Model
```python
@dataclass
class Patient:
    id: str
    age: int
    gender: str
    race: str
    cancer_type: str
    cancer_stage: str
    cancer_grade: str
    biomarkers: List[Biomarker]
    ecog_status: int
    treatment_intent: str
    comorbidities: List[str]
    family_history: List[str]
```

### Trial Model
```python
@dataclass
class Trial:
    nct_id: str
    title: str
    description: str
    conditions: List[str]
    eligibility_criteria: EligibilityCriteria
    locations: List[Location]
    status: str
    phase: str
    biomarkers: List[str]
    age_range: Tuple[int, int]
    gender_eligibility: str
```

### RankedTrial Model
```python
@dataclass
class RankedTrial:
    trial: Trial
    total_score: float
    subscores: Dict[str, float]  # eligibility, biomarker, clinical, practical
    confidence: float
    reasoning: str
    key_matches: List[str]
    concerns: List[str]
    deterministic_filters_applied: Dict[str, bool]  # age_ok, gender_ok, trial_open, geography_ok
    expert_scores: Dict[str, float]  # medical_expert, biomarker_specialist, patient_advocate
    judge_consolidation: str  # Meta-model reasoning
```

## Implementation Timeline

### Phase 1: Foundation (45 minutes)
- [ ] Set up project structure
- [ ] Implement BioMCP integration
- [ ] Create basic trial fetcher
- [ ] Implement patient data processor

### Phase 2: Core Matching (60 minutes)
- [ ] Implement LLM ranker
- [ ] Create scoring system
- [ ] Build main interface
- [ ] Basic testing

### Phase 3: Evaluation (45 minutes)
- [ ] Implement evaluation suite
- [ ] Create LLM-as-judge system
- [ ] Add biomarker validation
- [ ] Performance testing

### Phase 4: Polish (30 minutes)
- [ ] Error handling and logging
- [ ] Documentation
- [ ] Final testing
- [ ] Performance optimization

## Technical Dependencies

```python
# requirements.txt
biomcp>=0.1.0
openai>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
pydantic>=2.0.0
asyncio
aiohttp>=3.8.0
python-dotenv>=1.0.0
argparse
dataclasses
typing
```

## Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export BIOMCP_API_KEY="your_biomcp_key"
export OPENAI_API_KEY="your_openai_key"

# Run the system
python src/match.py --patient_id P002
python tests/eval.py
```

## Key Technical Decisions

1. **Hybrid Approach**: Deterministic filters + LLM scoring for safety and quality
2. **Latest Frontier Models**: GPT-5, Claude 4.1 Sonnet, Gemini 2.5 Flash
3. **Mixture-of-Experts**: Multiple expert prompts with judge consolidation
4. **Deterministic Pre-filtering**: Remove obvious mismatches before LLM
5. **Transparency**: Log all deterministic filter decisions
6. **Parallel Processing**: Concurrent API calls to multiple models
7. **Caching Strategy**: Cache both deterministic and LLM results
8. **Error Handling**: Graceful degradation with fallback scoring

## Success Metrics

- **Functionality**: System runs without errors
- **Accuracy**: High-quality trial matches (validated by LLM judge)
- **Performance**: <30 seconds for patient matching
- **Evaluation**: Comprehensive test suite with clear metrics
- **Code Quality**: Clean, documented, maintainable code

## Risk Mitigation

1. **BioMCP API Issues**: Implement fallback to ClinicalTrials.gov
2. **LLM Rate Limits**: Implement retry logic with exponential backoff
3. **Data Quality**: Validate and normalize all input data
4. **Performance**: Cache results and optimize API calls
5. **Evaluation Bias**: Use multiple evaluation approaches

This implementation plan provides a robust, scalable solution that meets all requirements while staying within the 3-hour time constraint.
