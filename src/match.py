#!/usr/bin/env python3
"""
Clinical Trial Matching System - Main Interface with Hybrid Ranking

This is the primary entry point for the clinical trial matching system.
It integrates BioMCP fetching with hybrid deterministic + LLM ranking.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from biomcp_fetcher import BioMCPClient, ClinicalTrial
from llm_ranker import HybridTrialRanker, RankedTrial
from enhanced_filters import GenderNormalizer, BiomarkerMatcher, GeographicCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridClinicalTrialMatcher:
    """
    Main clinical trial matching system with hybrid ranking.
    
    Integrates BioMCP trial fetching with deterministic filtering
    and mixture-of-experts LLM ranking.
    """
    
    def __init__(
        self,
        biomcp_mode: str = "auto",
        use_mixture_of_experts: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the hybrid clinical trial matcher.
        
        Args:
            biomcp_mode: BioMCP client mode ("sdk", "mcp", or "auto")
            use_mixture_of_experts: Whether to use multiple LLM experts
            verbose: Enable verbose logging
        """
        # Initialize components
        self.biomcp_client = BioMCPClient(mode=biomcp_mode)
        self.hybrid_ranker = HybridTrialRanker(
            use_mixture_of_experts=use_mixture_of_experts
        )
        self.patients_df = None
        self.verbose = verbose
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        self._load_patient_data()
        self._check_api_keys()
    
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
    
    def _check_api_keys(self):
        """Check for required API keys and provide status."""
        keys_status = {
            'NCI_API_KEY': os.getenv('NCI_API_KEY'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY')
        }
        
        logger.info("API Key Status:")
        for key_name, key_value in keys_status.items():
            if key_value:
                logger.info(f"  ✓ {key_name}: Configured ({key_value[:8]}...)")
            else:
                logger.warning(f"  ✗ {key_name}: Not configured")
        
        if not any(keys_status.values()):
            logger.warning("No API keys configured - will use mock data")
    
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
        cancer_type_filter: Optional[str] = None
    ) -> List[RankedTrial]:
        """
        Find and rank matching clinical trials for a patient using hybrid approach.
        
        Args:
            patient_id: Patient identifier
            max_trials: Maximum number of trials to return
            cancer_type_filter: Optional cancer type filter
        
        Returns:
            List of ranked trials with scores
        """
        start_time = datetime.now()
        
        # Get patient data
        patient = self.get_patient(patient_id)
        
        # Apply cancer type filter if specified
        if cancer_type_filter and cancer_type_filter.lower() not in patient['cancer_type'].lower():
            logger.warning(f"Patient {patient_id} has {patient['cancer_type']}, not {cancer_type_filter}")
            return []
        
        # Log patient info
        logger.info("=" * 60)
        logger.info(f"MATCHING TRIALS FOR PATIENT {patient_id}")
        logger.info("=" * 60)
        logger.info(f"Name: {patient['name']}")
        logger.info(f"Demographics: {patient['age']}yo {patient['gender']}")
        logger.info(f"Cancer: {patient['cancer_type']} Stage {patient['cancer_stage']}")
        logger.info(f"Biomarkers Detected: {patient.get('biomarkers_detected', 'None')}")
        logger.info(f"Biomarkers Ruled Out: {patient.get('biomarkers_ruled_out', 'None')}")
        logger.info(f"Location: {patient.get('city', '')}, {patient.get('state', '')}")
        
        # Step 1: Fetch trials from BioMCP
        logger.info("\n📡 Fetching trials from BioMCP...")
        async with self.biomcp_client as client:
            # Build search terms
            search_terms = []
            if patient.get('cancer_stage'):
                search_terms.append(f"stage {patient['cancer_stage']}")
            
            # Add normalized biomarkers
            biomarkers = patient.get('biomarkers_detected', '')
            if pd.notna(biomarkers) and biomarkers:
                normalized_markers = BiomarkerMatcher.extract_biomarkers(str(biomarkers))
                search_terms.extend(normalized_markers)
            
            # Search for trials
            trials = await client.search_trials(
                condition=patient['cancer_type'],
                additional_terms=search_terms,
                max_results=max_trials * 3  # Get extra for filtering
            )
        
        if not trials:
            logger.warning("No trials found from BioMCP")
            return []
        
        logger.info(f"  ✓ Retrieved {len(trials)} trials")
        
        # Step 2: Apply hybrid ranking
        logger.info("\n🤖 Applying hybrid ranking...")
        logger.info(f"  • Deterministic filtering enabled")
        logger.info(f"  • Mixture of experts: {self.hybrid_ranker.use_mixture_of_experts}")
        
        ranked_trials = await self.hybrid_ranker.rank_trials(
            patient=patient,
            trials=trials,
            max_trials=max_trials
        )
        
        # Calculate timing
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Ranking complete in {elapsed:.1f} seconds")
        logger.info(f"  • {len(ranked_trials)} trials ranked")
        
        return ranked_trials
    
    def format_trial_output(self, ranked_trials: List[RankedTrial], patient_id: str) -> str:
        """
        Format ranked trial results for display with scores and reasoning.
        
        Args:
            ranked_trials: List of ranked trials with scores
            patient_id: Patient identifier
        
        Returns:
            Formatted string output
        """
        if not ranked_trials:
            return f"\nNo matching trials found for patient {patient_id}.\n"
        
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"RANKED CLINICAL TRIALS FOR PATIENT {patient_id}")
        output.append(f"{'='*80}")
        
        for ranked in ranked_trials:
            trial = ranked.trial
            score = ranked.score
            
            # Header with rank and score
            output.append(f"\n#{ranked.rank}. {trial.nct_id} - SCORE: {score.total_score:.1f}/100")
            output.append(f"   {'─'*70}")
            
            # Trial details
            output.append(f"   Title: {trial.title}")
            output.append(f"   Phase: {trial.phase} | Status: {trial.status}")
            
            # Score breakdown
            if score.subscores:
                output.append(f"\n   📊 Score Breakdown:")
                output.append(f"      • Eligibility: {score.subscores.get('eligibility', 0):.1f}/40")
                output.append(f"      • Biomarker: {score.subscores.get('biomarker', 0):.1f}/30")
                output.append(f"      • Clinical: {score.subscores.get('clinical', 0):.1f}/20")
                output.append(f"      • Practical: {score.subscores.get('practical', 0):.1f}/10")
                output.append(f"      • Confidence: {score.confidence:.1%}")
            
            # Deterministic filters
            if score.deterministic_filters:
                output.append(f"\n   🔍 Filter Results:")
                for filter_name, passed in score.deterministic_filters.items():
                    status = "✓" if passed else "✗"
                    output.append(f"      {status} {filter_name.replace('_', ' ').title()}")
            
            # Key matches and concerns
            if score.key_matches:
                output.append(f"\n   ✅ Key Matches:")
                for match in score.key_matches[:3]:
                    output.append(f"      • {match}")
            
            if score.concerns:
                output.append(f"\n   ⚠️  Concerns:")
                for concern in score.concerns[:3]:
                    output.append(f"      • {concern}")
            
            # Expert scores if available
            if score.expert_scores and self.verbose:
                output.append(f"\n   👥 Expert Opinions:")
                for expert in score.expert_scores:
                    output.append(f"      • {expert.expert_name}: {expert.score:.1f}/100")
            
            # Brief reasoning
            if score.reasoning:
                reasoning_brief = score.reasoning[:200] + "..." if len(score.reasoning) > 200 else score.reasoning
                output.append(f"\n   💭 Reasoning: {reasoning_brief}")
            
            # Trial location
            if trial.locations:
                loc = trial.locations[0]
                location = f"{loc.get('city', '')}, {loc.get('state', '')}"
                if location.strip(', '):
                    output.append(f"\n   📍 Location: {location}")
        
        output.append(f"\n{'='*80}")
        output.append(f"Summary: {len(ranked_trials)} trials ranked by hybrid scoring")
        output.append(f"{'='*80}\n")
        
        return "\n".join(output)


async def main():
    """Main entry point for the hybrid clinical trial matcher."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Matching System with Hybrid Ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/match.py --patient_id 1
  python src/match.py --patient_id 2 --max_trials 5
  python src/match.py --patient_id 3 --cancer_type "Breast" --verbose
  python src/match.py --patient_id 4 --no-experts  # Disable mixture of experts
        """
    )
    
    parser.add_argument(
        '--patient_id',
        required=True,
        help='Patient ID to match (use numeric: 1, 2, 3, etc.)'
    )
    
    parser.add_argument(
        '--max_trials',
        type=int,
        default=5,
        help='Maximum number of trials to return (default: 5)'
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
        '--no-experts',
        action='store_true',
        help='Disable mixture of experts (use single LLM)'
    )
    
    parser.add_argument(
        '--output',
        choices=['text', 'json', 'detailed'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging and show expert scores'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize hybrid matcher
        matcher = HybridClinicalTrialMatcher(
            biomcp_mode=args.mode,
            use_mixture_of_experts=not args.no_experts,
            verbose=args.verbose
        )
        
        # Find and rank matching trials
        ranked_trials = await matcher.match_patient_trials(
            patient_id=args.patient_id,
            max_trials=args.max_trials,
            cancer_type_filter=args.cancer_type
        )
        
        # Format and display results
        if args.output == 'json':
            # JSON output for programmatic use
            trials_data = []
            for ranked in ranked_trials:
                trial = ranked.trial
                score = ranked.score
                
                trial_data = {
                    'rank': ranked.rank,
                    'nct_id': trial.nct_id,
                    'title': trial.title,
                    'phase': trial.phase,
                    'status': trial.status,
                    'total_score': score.total_score,
                    'subscores': score.subscores,
                    'confidence': score.confidence,
                    'key_matches': score.key_matches,
                    'concerns': score.concerns,
                    'deterministic_filters': score.deterministic_filters
                }
                
                if args.verbose and score.expert_scores:
                    trial_data['expert_scores'] = [
                        {
                            'expert': expert.expert_name,
                            'score': expert.score,
                            'confidence': expert.confidence
                        }
                        for expert in score.expert_scores
                    ]
                
                trials_data.append(trial_data)
            
            print(json.dumps({
                'patient_id': args.patient_id,
                'total_trials': len(ranked_trials),
                'mixture_of_experts': not args.no_experts,
                'trials': trials_data
            }, indent=2))
            
        elif args.output == 'detailed':
            # Detailed output with full reasoning
            output = matcher.format_trial_output(ranked_trials, args.patient_id)
            print(output)
            
            # Add detailed reasoning for each trial
            for ranked in ranked_trials:
                print(f"\nDETAILED REASONING FOR {ranked.trial.nct_id}:")
                print("=" * 60)
                print(ranked.score.reasoning)
                if ranked.score.judge_consolidation:
                    print(f"\nJUDGE CONSOLIDATION:")
                    print(ranked.score.judge_consolidation)
                print("=" * 60)
        else:
            # Standard text output
            output = matcher.format_trial_output(ranked_trials, args.patient_id)
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
