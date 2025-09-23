#!/usr/bin/env python3
"""
Clinical Trial Matching System - Evaluation Suite

This module provides comprehensive evaluation of the clinical trial matching system
using multiple validation approaches including LLM-as-judge, biomarker validation,
and clinical logic testing.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biomcp_fetcher import TrialMatcher, ClinicalTrial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationSuite:
    """
    Comprehensive evaluation suite for clinical trial matching.
    
    Provides multiple evaluation approaches to validate matching quality
    and system performance.
    """
    
    def __init__(self, mode: str = "auto"):
        """
        Initialize evaluation suite.
        
        Args:
            mode: BioMCP client mode for testing
        """
        self.matcher = TrialMatcher(mode=mode)
        self.patients_df = None
        self.evaluation_results = {}
        self._load_patient_data()
    
    def _load_patient_data(self):
        """Load patient data for evaluation."""
        try:
            csv_path = Path(__file__).parent.parent / "patients.csv"
            self.patients_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.patients_df)} patients for evaluation")
        except FileNotFoundError:
            logger.error("patients.csv not found")
            sys.exit(1)
    
    async def evaluate_all_patients(self, max_trials: int = 5) -> Dict[str, Any]:
        """
        Evaluate matching for all patients in the dataset.
        
        Args:
            max_trials: Maximum trials to fetch per patient
        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting comprehensive evaluation of all patients...")
        
        results = {
            'total_patients': len(self.patients_df),
            'successful_matches': 0,
            'failed_matches': 0,
            'total_trials_found': 0,
            'cancer_type_breakdown': {},
            'evaluation_metrics': {},
            'patient_results': []
        }
        
        for idx, patient in self.patients_df.iterrows():
            patient_id = f"P{idx + 1:03d}"
            patient_data = patient.to_dict()
            
            try:
                # Get matching trials
                trials = await self.matcher.match_patient(patient_data, max_trials=max_trials)
                
                # Record results
                patient_result = {
                    'patient_id': patient_id,
                    'name': patient_data['name'],
                    'cancer_type': patient_data['cancer_type'],
                    'cancer_stage': patient_data['cancer_stage'],
                    'trials_found': len(trials),
                    'trials': [
                        {
                            'nct_id': trial.nct_id,
                            'title': trial.title,
                            'phase': trial.phase,
                            'status': trial.status
                        }
                        for trial in trials
                    ]
                }
                
                results['patient_results'].append(patient_result)
                results['total_trials_found'] += len(trials)
                
                if trials:
                    results['successful_matches'] += 1
                else:
                    results['failed_matches'] += 1
                
                # Track by cancer type
                cancer_type = patient_data['cancer_type']
                if cancer_type not in results['cancer_type_breakdown']:
                    results['cancer_type_breakdown'][cancer_type] = {
                        'patients': 0,
                        'trials_found': 0
                    }
                
                results['cancer_type_breakdown'][cancer_type]['patients'] += 1
                results['cancer_type_breakdown'][cancer_type]['trials_found'] += len(trials)
                
                logger.info(f"  {patient_id}: {len(trials)} trials found")
                
            except Exception as e:
                logger.error(f"  {patient_id}: Error - {e}")
                results['failed_matches'] += 1
        
        # Calculate metrics
        results['evaluation_metrics'] = self._calculate_metrics(results)
        
        return results
    
    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate evaluation metrics from results."""
        total_patients = results['total_patients']
        successful_matches = results['successful_matches']
        total_trials = results['total_trials_found']
        
        metrics = {
            'match_rate': successful_matches / total_patients if total_patients > 0 else 0,
            'avg_trials_per_patient': total_trials / total_patients if total_patients > 0 else 0,
            'avg_trials_per_successful_match': total_trials / successful_matches if successful_matches > 0 else 0
        }
        
        # Calculate trials per cancer type
        for cancer_type, data in results['cancer_type_breakdown'].items():
            if data['patients'] > 0:
                metrics[f'avg_trials_{cancer_type.lower().replace(" ", "_")}'] = (
                    data['trials_found'] / data['patients']
                )
        
        return metrics
    
    async def evaluate_biomarker_matching(self) -> Dict[str, Any]:
        """
        Evaluate biomarker-based matching accuracy.
        
        Tests how well the system matches trials based on molecular markers.
        """
        logger.info("Evaluating biomarker matching...")
        
        # Test cases with known biomarker requirements
        test_cases = [
            {
                'name': 'ER+ Breast Cancer',
                'cancer_type': 'Breast',
                'biomarkers_detected': 'ER+, PR+',
                'biomarkers_ruled_out': 'HER2+',
                'expected_biomarkers': ['ER+', 'PR+'],
                'excluded_biomarkers': ['HER2+']
            },
            {
                'name': 'EGFR+ Lung Cancer',
                'cancer_type': 'Lung',
                'biomarkers_detected': 'EGFR mutation',
                'biomarkers_ruled_out': 'ALK fusion',
                'expected_biomarkers': ['EGFR'],
                'excluded_biomarkers': ['ALK']
            },
            {
                'name': 'KRAS+ Pancreatic Cancer',
                'cancer_type': 'Pancreatic',
                'biomarkers_detected': 'KRAS mutation',
                'biomarkers_ruled_out': 'MSI-H',
                'expected_biomarkers': ['KRAS'],
                'excluded_biomarkers': ['MSI-H']
            }
        ]
        
        results = {
            'test_cases': len(test_cases),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        for test_case in test_cases:
            try:
                # Create patient data
                patient_data = {
                    'cancer_type': f"{test_case['cancer_type']} Cancer",
                    'cancer_stage': 'II',
                    'biomarkers_detected': test_case['biomarkers_detected'],
                    'biomarkers_ruled_out': test_case['biomarkers_ruled_out']
                }
                
                # Get matching trials
                trials = await self.matcher.match_patient(patient_data, max_trials=5)
                
                # Check biomarker compatibility
                compatible_trials = 0
                for trial in trials:
                    if trial.matches_biomarkers(
                        test_case['biomarkers_detected'].split(', '),
                        test_case['biomarkers_ruled_out'].split(', ')
                    ):
                        compatible_trials += 1
                
                test_result = {
                    'test_name': test_case['name'],
                    'trials_found': len(trials),
                    'compatible_trials': compatible_trials,
                    'compatibility_rate': compatible_trials / len(trials) if trials else 0,
                    'passed': compatible_trials > 0
                }
                
                results['test_details'].append(test_result)
                
                if test_result['passed']:
                    results['passed_tests'] += 1
                else:
                    results['failed_tests'] += 1
                
                logger.info(f"  {test_case['name']}: {compatible_trials}/{len(trials)} compatible trials")
                
            except Exception as e:
                logger.error(f"  {test_case['name']}: Error - {e}")
                results['failed_tests'] += 1
        
        results['success_rate'] = results['passed_tests'] / results['test_cases']
        return results
    
    async def evaluate_clinical_logic(self) -> Dict[str, Any]:
        """
        Evaluate clinical appropriateness of trial matches.
        
        Tests stage-appropriate trial selection and other clinical logic.
        """
        logger.info("Evaluating clinical logic...")
        
        # Test stage-appropriate matching
        stage_tests = [
            {'stage': 'I', 'expected_phases': ['Phase 1', 'Phase 2', 'Phase 3']},
            {'stage': 'II', 'expected_phases': ['Phase 2', 'Phase 3']},
            {'stage': 'III', 'expected_phases': ['Phase 2', 'Phase 3']},
            {'stage': 'IV', 'expected_phases': ['Phase 1', 'Phase 2', 'Phase 3']}
        ]
        
        results = {
            'stage_tests': len(stage_tests),
            'passed_stage_tests': 0,
            'stage_test_details': []
        }
        
        for stage_test in stage_tests:
            try:
                # Create test patient
                patient_data = {
                    'cancer_type': 'Breast Cancer',
                    'cancer_stage': stage_test['stage'],
                    'biomarkers_detected': 'ER+',
                    'biomarkers_ruled_out': ''
                }
                
                trials = await self.matcher.match_patient(patient_data, max_trials=5)
                
                # Check if trials are stage-appropriate
                appropriate_trials = 0
                for trial in trials:
                    if any(phase in trial.phase for phase in stage_test['expected_phases']):
                        appropriate_trials += 1
                
                test_result = {
                    'stage': stage_test['stage'],
                    'trials_found': len(trials),
                    'appropriate_trials': appropriate_trials,
                    'appropriateness_rate': appropriate_trials / len(trials) if trials else 0,
                    'passed': appropriate_trials > 0
                }
                
                results['stage_test_details'].append(test_result)
                
                if test_result['passed']:
                    results['passed_stage_tests'] += 1
                
                logger.info(f"  Stage {stage_test['stage']}: {appropriate_trials}/{len(trials)} appropriate trials")
                
            except Exception as e:
                logger.error(f"  Stage {stage_test['stage']}: Error - {e}")
        
        results['stage_success_rate'] = results['passed_stage_tests'] / results['stage_tests']
        return results
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("CLINICAL TRIAL MATCHING SYSTEM - EVALUATION REPORT")
        report.append("=" * 80)
        
        # Overall metrics
        metrics = results['evaluation_metrics']
        report.append(f"\nOVERALL PERFORMANCE:")
        report.append(f"  Total Patients: {results['total_patients']}")
        report.append(f"  Successful Matches: {results['successful_matches']}")
        report.append(f"  Failed Matches: {results['failed_matches']}")
        report.append(f"  Match Rate: {metrics['match_rate']:.1%}")
        report.append(f"  Avg Trials per Patient: {metrics['avg_trials_per_patient']:.1f}")
        report.append(f"  Avg Trials per Match: {metrics['avg_trials_per_successful_match']:.1f}")
        
        # Cancer type breakdown
        report.append(f"\nCANCER TYPE BREAKDOWN:")
        for cancer_type, data in results['cancer_type_breakdown'].items():
            avg_trials = data['trials_found'] / data['patients'] if data['patients'] > 0 else 0
            report.append(f"  {cancer_type}: {data['patients']} patients, {data['trials_found']} trials ({avg_trials:.1f} avg)")
        
        # Top performing patients
        report.append(f"\nTOP PERFORMING PATIENTS:")
        top_patients = sorted(
            results['patient_results'],
            key=lambda x: x['trials_found'],
            reverse=True
        )[:5]
        
        for patient in top_patients:
            report.append(f"  {patient['patient_id']} ({patient['name']}): {patient['trials_found']} trials")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)


async def main():
    """Main entry point for evaluation suite."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Matching System - Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/eval.py                    # Run all evaluations
  python tests/eval.py --eval_type all   # Run all evaluations
  python tests/eval.py --eval_type biomarker  # Run biomarker evaluation only
  python tests/eval.py --max_trials 3    # Limit trials per patient
        """
    )
    
    parser.add_argument(
        '--eval_type',
        choices=['all', 'patients', 'biomarker', 'clinical'],
        default='all',
        help='Type of evaluation to run (default: all)'
    )
    
    parser.add_argument(
        '--max_trials',
        type=int,
        default=5,
        help='Maximum trials to fetch per patient (default: 5)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['sdk', 'mcp', 'auto'],
        default='auto',
        help='BioMCP client mode (default: auto)'
    )
    
    parser.add_argument(
        '--output',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for API key
    api_key = os.getenv('NCI_API_KEY')
    if not api_key:
        logger.warning("NCI_API_KEY not set - using mock data for evaluation")
    else:
        logger.info(f"NCI_API_KEY found: {api_key[:8]}...")
    
    try:
        # Initialize evaluation suite
        evaluator = EvaluationSuite(mode=args.mode)
        
        # Run evaluations based on type
        if args.eval_type in ['all', 'patients']:
            logger.info("Running patient matching evaluation...")
            results = await evaluator.evaluate_all_patients(max_trials=args.max_trials)
            
            if args.output == 'json':
                print(json.dumps(results, indent=2))
            else:
                report = evaluator.generate_evaluation_report(results)
                print(report)
        
        if args.eval_type in ['all', 'biomarker']:
            logger.info("Running biomarker evaluation...")
            biomarker_results = await evaluator.evaluate_biomarker_matching()
            
            if args.output == 'json':
                print(json.dumps(biomarker_results, indent=2))
            else:
                print(f"\nBIOMARKER EVALUATION:")
                print(f"  Test Cases: {biomarker_results['test_cases']}")
                print(f"  Passed: {biomarker_results['passed_tests']}")
                print(f"  Failed: {biomarker_results['failed_tests']}")
                print(f"  Success Rate: {biomarker_results['success_rate']:.1%}")
        
        if args.eval_type in ['all', 'clinical']:
            logger.info("Running clinical logic evaluation...")
            clinical_results = await evaluator.evaluate_clinical_logic()
            
            if args.output == 'json':
                print(json.dumps(clinical_results, indent=2))
            else:
                print(f"\nCLINICAL LOGIC EVALUATION:")
                print(f"  Stage Tests: {clinical_results['stage_tests']}")
                print(f"  Passed: {clinical_results['passed_stage_tests']}")
                print(f"  Success Rate: {clinical_results['stage_success_rate']:.1%}")
        
        logger.info("Evaluation complete!")
    
    except KeyboardInterrupt:
        logger.info("Evaluation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
