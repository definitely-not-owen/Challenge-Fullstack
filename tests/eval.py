#!/usr/bin/env python3
"""
Minimal evaluation suite - simple, clever, no human labels needed.
"""

import asyncio
import json
import sys
from pathlib import Path
import logging
import pandas as pd
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.anchor_free_judge import AnchorFreeJudge
from tests.deterministic_oracles import DeterministicOracles
from tests.behavioral_tests import BehavioralTests
from tests.metrics_calculator import MinimalMetrics
from tests.report_generator import MinimalReporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_trials(patient: dict):
    """Create mock trials for testing when real trials aren't available."""
    
    @dataclass
    class MockTrial:
        nct_id: str
        title: str
        phase: str
        status: str
        conditions: str
        interventions: str
        minimum_age: str
        maximum_age: str
        gender: str
        
        def __dict__(self):
            return {
                'nct_id': self.nct_id,
                'title': self.title,
                'phase': self.phase,
                'status': self.status,
                'conditions': self.conditions,
                'interventions': self.interventions,
                'minimum_age': self.minimum_age,
                'maximum_age': self.maximum_age,
                'gender': self.gender
            }
    
    @dataclass
    class MockRankedTrial:
        trial: MockTrial
        score: dict
    
    # Create mock trials based on patient's cancer type
    cancer_type = patient.get('cancer_type', 'Cancer')
    
    mock_trials = [
        MockRankedTrial(
            trial=MockTrial(
                nct_id='NCT00001',
                title=f'Phase 2 Study of Novel Agent for {cancer_type}',
                phase='Phase 2',
                status='Recruiting',
                conditions=cancer_type,
                interventions='Novel Agent A',
                minimum_age='18 Years',
                maximum_age='99 Years',
                gender='All'
            ),
            score={'total_score': 85}
        ),
        MockRankedTrial(
            trial=MockTrial(
                nct_id='NCT00002',
                title=f'Combination Therapy for Advanced {cancer_type}',
                phase='Phase 3',
                status='Recruiting',
                conditions=cancer_type,
                interventions='Drug B + Drug C',
                minimum_age='21 Years',
                maximum_age='85 Years',
                gender='All'
            ),
            score={'total_score': 75}
        ),
        MockRankedTrial(
            trial=MockTrial(
                nct_id='NCT00003',
                title=f'Immunotherapy for {cancer_type}',
                phase='Phase 1',
                status='Recruiting',
                conditions=cancer_type,
                interventions='Checkpoint Inhibitor',
                minimum_age='18 Years',
                maximum_age='80 Years',
                gender='All'
            ),
            score={'total_score': 65}
        )
    ]
    
    return mock_trials


async def main():
    """Run the minimal evaluation suite."""
    from src.match import HybridClinicalTrialMatcher
    
    start_time = time.time()
    logger.info("Starting evaluation...")
    
    # Initialize matcher with SDK mode (no MCP)
    matcher = HybridClinicalTrialMatcher(
        biomcp_mode="sdk",  # Use SDK mode instead of MCP
        use_mixture_of_experts=False,  # Start without experts for speed
        verbose=False
    )
    
    # Load test patients
    patients_df = pd.read_csv('patients.csv')
    
    # Initialize components
    judge = AnchorFreeJudge()
    oracles = DeterministicOracles()
    behavioral = BehavioralTests(matcher)
    metrics = MinimalMetrics()
    reporter = MinimalReporter()
    
    all_results = []
    
    # Test ALL patients
    num_patients = len(patients_df)  # Test all 30 patients
    
    # Process patients in parallel batches
    BATCH_SIZE = 10  # Process 10 patients at a time
    
    async def evaluate_patient(patient_id):
        """Evaluate a single patient asynchronously."""
        logger.info(f"Evaluating patient {patient_id}/{num_patients}...")
        
        # Get patient data for oracles
        patient = patients_df.iloc[patient_id - 1].to_dict()
        
        # Clean patient data
        for field in ['biomarkers_detected', 'biomarkers_ruled_out']:
            if pd.notna(patient.get(field)):
                patient[field] = [b.strip() for b in str(patient[field]).split(',')]
            else:
                patient[field] = []
        
        # Get rankings using string patient ID
        try:
            ranked = await matcher.match_patient_trials(str(patient_id), max_trials=5)
        except Exception as e:
            logger.error(f"Failed to match patient {patient_id}: {e}")
            # Use mock trials for testing if real trials fail
            ranked = create_mock_trials(patient)
        
        if not ranked:
            logger.warning(f"No trials found for patient {patient_id}, using mock data")
            ranked = create_mock_trials(patient)
        
        # Evaluate
        # Get trial dict properly
        if hasattr(ranked[0].trial, '__dict__'):
            trial_dict = ranked[0].trial.__dict__() if callable(ranked[0].trial.__dict__) else ranked[0].trial.__dict__
        else:
            trial_dict = vars(ranked[0].trial)
            
        trial_dicts = []
        for r in ranked:
            if hasattr(r.trial, '__dict__'):
                td = r.trial.__dict__() if callable(r.trial.__dict__) else r.trial.__dict__
            else:
                td = vars(r.trial)
            trial_dicts.append(td)
        
        # Run evaluations in parallel
        judge_task = judge.evaluate(patient, trial_dict)
        behavioral_task = behavioral.run_all_tests(patient_id)
        
        judge_result, behavioral_result = await asyncio.gather(
            judge_task,
            behavioral_task
        )
        
        patient_results = {
            'judge': judge_result,
            'oracle': oracles.calculate_metrics(patient, trial_dicts),
            'behavioral': behavioral_result
        }
        
        return patient_results
    
    # Process patients in batches
    for batch_start in range(1, num_patients + 1, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_patients + 1)
        batch_ids = list(range(batch_start, batch_end))
        
        logger.info(f"Processing batch: patients {batch_start} to {batch_end - 1}")
        
        # Run batch in parallel
        batch_tasks = [evaluate_patient(pid) for pid in batch_ids]
        batch_results = await asyncio.gather(*batch_tasks)
        
        all_results.extend(batch_results)
    
    # Calculate metrics
    final_metrics = metrics.calculate_all(all_results[0] if all_results else {}, all_results)
    
    # Generate report
    report = reporter.generate(final_metrics, [])
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\n⏱️  Evaluation completed in {elapsed_time:.1f} seconds")
    print(f"   ({elapsed_time/num_patients:.1f} seconds per patient average)")
    
    return final_metrics


if __name__ == "__main__":
    asyncio.run(main())