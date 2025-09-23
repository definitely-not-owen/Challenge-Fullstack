#!/usr/bin/env python3
"""
Clinical Trial Matching System - Main Interface

This is the primary entry point for the clinical trial matching system.
It provides a command-line interface for matching patients with clinical trials.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from biomcp_fetcher import TrialMatcher, ClinicalTrial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClinicalTrialMatcher:
    """
    Main clinical trial matching system.
    
    Coordinates trial fetching, LLM ranking, and result presentation.
    """
    
    def __init__(self, mode: str = "auto"):
        """
        Initialize the clinical trial matcher.
        
        Args:
            mode: BioMCP client mode ("sdk", "mcp", or "auto")
        """
        self.matcher = TrialMatcher(mode=mode)
        self.patients_df = None
        self._load_patient_data()
    
    def _load_patient_data(self):
        """Load patient data from CSV file."""
        try:
            csv_path = Path(__file__).parent.parent / "patients.csv"
            self.patients_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.patients_df)} patients from CSV")
        except FileNotFoundError:
            logger.error("patients.csv not found. Please ensure it's in the project root.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading patient data: {e}")
            sys.exit(1)
    
    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """
        Get patient data by ID.
        
        Args:
            patient_id: Patient identifier (e.g., "P001", "P002")
        
        Returns:
            Dictionary containing patient information
        """
        # Handle different ID formats
        if patient_id.startswith('P'):
            # Direct ID match
            patient_row = self.patients_df[self.patients_df.index.astype(str) == patient_id[1:]]
        else:
            # Try direct index
            try:
                idx = int(patient_id) - 1
                patient_row = self.patients_df.iloc[[idx]]
            except (ValueError, IndexError):
                patient_row = self.patients_df[self.patients_df.index.astype(str) == patient_id]
        
        if patient_row.empty:
            raise ValueError(f"Patient {patient_id} not found")
        
        return patient_row.iloc[0].to_dict()
    
    async def match_patient_trials(
        self,
        patient_id: str,
        max_trials: int = 10,
        cancer_type_filter: str = None
    ) -> List[ClinicalTrial]:
        """
        Find matching clinical trials for a patient.
        
        Args:
            patient_id: Patient identifier
            max_trials: Maximum number of trials to return
            cancer_type_filter: Optional cancer type filter
        
        Returns:
            List of matching clinical trials
        """
        # Get patient data
        patient = self.get_patient(patient_id)
        
        # Apply cancer type filter if specified
        if cancer_type_filter and cancer_type_filter.lower() not in patient['cancer_type'].lower():
            logger.warning(f"Patient {patient_id} has {patient['cancer_type']}, not {cancer_type_filter}")
            return []
        
        logger.info(f"Matching trials for patient {patient_id}: {patient['name']}")
        logger.info(f"  Cancer: {patient['cancer_type']} Stage {patient['cancer_stage']}")
        logger.info(f"  Biomarkers: {patient.get('biomarkers_detected', 'None')}")
        
        # Find matching trials
        trials = await self.matcher.match_patient(patient, max_trials=max_trials)
        
        logger.info(f"Found {len(trials)} matching trials")
        return trials
    
    def format_trial_output(self, trials: List[ClinicalTrial], patient_id: str) -> str:
        """
        Format trial results for display.
        
        Args:
            trials: List of clinical trials
            patient_id: Patient identifier
        
        Returns:
            Formatted string output
        """
        if not trials:
            return f"\nNo matching trials found for patient {patient_id}.\n"
        
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"CLINICAL TRIAL MATCHES FOR PATIENT {patient_id}")
        output.append(f"{'='*80}")
        
        for i, trial in enumerate(trials, 1):
            output.append(f"\n{i}. {trial.nct_id}")
            output.append(f"   Title: {trial.title}")
            output.append(f"   Phase: {trial.phase} | Status: {trial.status}")
            
            if trial.min_age or trial.max_age:
                age_range = f"Ages {trial.min_age or 'N/A'}-{trial.max_age or 'N/A'}"
                output.append(f"   Eligibility: {age_range}, {trial.gender}")
            
            if trial.locations:
                loc = trial.locations[0]
                location = f"{loc.get('city', '')}, {loc.get('state', '')}"
                if location.strip(', '):
                    output.append(f"   Location: {location}")
            
            if trial.interventions:
                interventions = ", ".join(trial.interventions[:3])
                output.append(f"   Interventions: {interventions}")
            
            if trial.sponsor:
                output.append(f"   Sponsor: {trial.sponsor}")
        
        output.append(f"\n{'='*80}")
        output.append(f"Total: {len(trials)} matching trials")
        output.append(f"{'='*80}\n")
        
        return "\n".join(output)


async def main():
    """Main entry point for the clinical trial matcher."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Matching System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/match.py --patient_id P001
  python src/match.py --patient_id P002 --max_trials 5
  python src/match.py --patient_id P003 --cancer_type "Breast"
        """
    )
    
    parser.add_argument(
        '--patient_id',
        required=True,
        help='Patient ID to match (e.g., P001, P002, or 1, 2)'
    )
    
    parser.add_argument(
        '--max_trials',
        type=int,
        default=10,
        help='Maximum number of trials to return (default: 10)'
    )
    
    parser.add_argument(
        '--cancer_type',
        help='Filter by cancer type (e.g., "Breast", "Lung")'
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
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for API key
    api_key = os.getenv('NCI_API_KEY')
    if not api_key:
        logger.warning("NCI_API_KEY not set - using mock data")
    else:
        logger.info(f"NCI_API_KEY found: {api_key[:8]}...")
    
    try:
        # Initialize matcher
        matcher = ClinicalTrialMatcher(mode=args.mode)
        
        # Find matching trials
        trials = await matcher.match_patient_trials(
            patient_id=args.patient_id,
            max_trials=args.max_trials,
            cancer_type_filter=args.cancer_type
        )
        
        # Format and display results
        if args.output == 'json':
            # JSON output for programmatic use
            trials_data = []
            for trial in trials:
                trials_data.append({
                    'nct_id': trial.nct_id,
                    'title': trial.title,
                    'phase': trial.phase,
                    'status': trial.status,
                    'min_age': trial.min_age,
                    'max_age': trial.max_age,
                    'gender': trial.gender,
                    'locations': trial.locations,
                    'interventions': trial.interventions,
                    'sponsor': trial.sponsor
                })
            
            print(json.dumps({
                'patient_id': args.patient_id,
                'total_trials': len(trials),
                'trials': trials_data
            }, indent=2))
        else:
            # Text output for human reading
            output = matcher.format_trial_output(trials, args.patient_id)
            print(output)
    
    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
