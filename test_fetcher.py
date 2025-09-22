"""
Quick test script to verify the trial fetcher is working.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trial_fetcher import BioMCPTrialFetcher


async def test_fetcher():
    """Test the trial fetcher with different cancer types from our patient data."""
    
    print("Testing Clinical Trial Fetcher\n" + "="*50)
    
    # Test cases based on our patient data
    test_cases = [
        {
            "cancer_type": "Breast Cancer",
            "stage": "II",
            "biomarkers": ["ER+", "PR+"],
            "description": "Breast Cancer Stage II with ER+/PR+"
        },
        {
            "cancer_type": "Lung Cancer",
            "stage": "I",
            "biomarkers": ["PD-L1"],
            "description": "Lung Cancer Stage I with PD-L1"
        },
        {
            "cancer_type": "Pancreatic Cancer",
            "stage": "III",
            "biomarkers": ["KRAS"],
            "description": "Pancreatic Cancer Stage III with KRAS mutation"
        }
    ]
    
    async with BioMCPTrialFetcher() as fetcher:
        for test_case in test_cases:
            print(f"\nTest: {test_case['description']}")
            print("-" * 40)
            
            try:
                trials = await fetcher.fetch_trials_for_cancer(
                    cancer_type=test_case["cancer_type"],
                    stage=test_case["stage"],
                    biomarkers=test_case["biomarkers"],
                    max_trials=5
                )
                
                if trials:
                    print(f"✓ Found {len(trials)} trials")
                    
                    # Show first trial details
                    if trials:
                        trial = trials[0]
                        print(f"\nFirst trial:")
                        print(f"  NCT ID: {trial.nct_id}")
                        print(f"  Title: {trial.title[:100]}...")
                        print(f"  Status: {trial.status.value}")
                        print(f"  Phase: {trial.phase}")
                        print(f"  Eligibility: Ages {trial.eligibility.min_age or 'N/A'}-{trial.eligibility.max_age or 'N/A'}")
                else:
                    print("✗ No trials found")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
    
    print("\n" + "="*50)
    print("Testing complete!")


if __name__ == "__main__":
    asyncio.run(test_fetcher())
