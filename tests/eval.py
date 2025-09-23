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

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.anchor_free_judge import AnchorFreeJudge
from tests.deterministic_oracles import DeterministicOracles
from tests.behavioral_tests import BehavioralTests
from tests.metrics_calculator import MinimalMetrics
from tests.report_generator import MinimalReporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run the minimal evaluation suite."""
    from src.match import HybridClinicalTrialMatcher
    
    logger.info("Starting evaluation...")
    
    # Initialize matcher
    matcher = HybridClinicalTrialMatcher(
        biomcp_mode="auto",
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
    
    # Test first 3 patients
    for patient_id in [1, 2, 3]:
        logger.info(f"Evaluating patient {patient_id}/3...")
        
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
            logger.error(f"Failed to match patient: {e}")
            continue
        
        if not ranked:
            logger.warning("No trials found")
            continue
        
        # Evaluate
        patient_results = {
            'judge': await judge.evaluate(patient, ranked[0].trial.__dict__),
            'oracle': oracles.calculate_metrics(patient, [r.trial.__dict__ for r in ranked]),
            'behavioral': await behavioral.run_all_tests(patient_id)
        }
        
        all_results.append(patient_results)
    
    # Calculate metrics
    final_metrics = metrics.calculate_all(all_results[0] if all_results else {}, all_results)
    
    # Generate report
    report = reporter.generate(final_metrics, [])
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    return final_metrics


if __name__ == "__main__":
    asyncio.run(main())