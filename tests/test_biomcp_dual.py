#!/usr/bin/env python3
"""
Test script for dual-mode BioMCP client (SDK and MCP styles).

This demonstrates how to use BioMCP in different modes based on
what's available in your environment.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from biomcp_fetcher import BioMCPClient, TrialMatcher


async def test_sdk_mode():
    """Test SDK-style HTTP API access."""
    print("\n" + "=" * 60)
    print("Testing SDK Mode (HTTP API)")
    print("=" * 60)
    
    async with BioMCPClient(mode="sdk") as client:
        # Test 1: Breast Cancer
        print("\n1. Searching for Breast Cancer trials...")
        trials = await client.search_trials(
            condition="Breast Cancer",
            additional_terms=["ER+", "PR+", "HER2-"],
            max_results=3
        )
        
        if trials:
            print(f"   ✓ Found {len(trials)} trials")
            for trial in trials:
                print(f"   - {trial.nct_id}: {trial.title[:50]}...")
        else:
            print("   ✗ No trials found")
        
        # Test 2: Lung Cancer with EGFR
        print("\n2. Searching for Lung Cancer with EGFR mutation...")
        trials = await client.search_trials(
            condition="Non-Small Cell Lung Cancer",
            additional_terms=["EGFR mutation", "stage IV"],
            max_results=3
        )
        
        if trials:
            print(f"   ✓ Found {len(trials)} trials")
            for trial in trials:
                print(f"   - {trial.nct_id}: {trial.title[:50]}...")
        else:
            print("   ✗ No trials found")


async def test_mcp_mode():
    """Test MCP protocol access."""
    print("\n" + "=" * 60)
    print("Testing MCP Mode (Model Context Protocol)")
    print("=" * 60)
    
    async with BioMCPClient(mode="mcp") as client:
        # Test 1: Pancreatic Cancer
        print("\n1. Searching for Pancreatic Cancer trials...")
        trials = await client.search_trials(
            condition="Pancreatic Cancer",
            additional_terms=["KRAS mutation", "metastatic"],
            max_results=3
        )
        
        if trials:
            print(f"   ✓ Found {len(trials)} trials")
            for trial in trials:
                print(f"   - {trial.nct_id}: {trial.title[:50]}...")
        else:
            print("   ✗ No trials found")
        
        # Test 2: Ovarian Cancer
        print("\n2. Searching for Ovarian Cancer with BRCA...")
        trials = await client.search_trials(
            condition="Ovarian Cancer",
            additional_terms=["BRCA mutation", "platinum-sensitive"],
            max_results=3
        )
        
        if trials:
            print(f"   ✓ Found {len(trials)} trials")
            for trial in trials:
                print(f"   - {trial.nct_id}: {trial.title[:50]}...")
        else:
            print("   ✗ No trials found")


async def test_auto_mode():
    """Test auto-detection mode."""
    print("\n" + "=" * 60)
    print("Testing Auto Mode (Best Available)")
    print("=" * 60)
    
    # This will automatically choose the best available mode
    async with BioMCPClient(mode="auto") as client:
        print(f"\n✓ Auto-selected mode: {client.mode}")
        
        print("\n1. Searching for Colorectal Cancer trials...")
        trials = await client.search_trials(
            condition="Colorectal Cancer",
            additional_terms=["MSI-H", "immunotherapy"],
            max_results=3
        )
        
        if trials:
            print(f"   ✓ Found {len(trials)} trials")
            for trial in trials:
                print(f"   - {trial.nct_id}: {trial.title[:50]}...")
        else:
            print("   ✗ No trials found")


async def test_patient_matching():
    """Test patient-centric trial matching."""
    print("\n" + "=" * 60)
    print("Testing Patient Matching")
    print("=" * 60)
    
    # Load a sample patient from our CSV
    import pandas as pd
    
    try:
        patients_df = pd.read_csv('patients.csv')
        patient = patients_df.iloc[0].to_dict()  # First patient
        
        print(f"\nPatient: {patient['name']}")
        print(f"  Cancer: {patient['cancer_type']} Stage {patient['cancer_stage']}")
        print(f"  Biomarkers: {patient.get('biomarkers_detected', 'None')}")
        
        # Match trials for this patient
        matcher = TrialMatcher(mode="auto")
        trials = await matcher.match_patient(patient, max_trials=5)
        
        print(f"\n✓ Found {len(trials)} matching trials:")
        for i, trial in enumerate(trials, 1):
            print(f"\n  {i}. {trial.nct_id}")
            print(f"     Title: {trial.title[:60]}...")
            print(f"     Phase: {trial.phase}, Status: {trial.status}")
            if trial.locations:
                loc = trial.locations[0]
                print(f"     Location: {loc.get('city', '')}, {loc.get('state', '')}")
    
    except FileNotFoundError:
        print("  ✗ patients.csv not found")
    except Exception as e:
        print(f"  ✗ Error: {e}")


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" BioMCP Dual-Mode Testing ".center(70, "="))
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('NCI_API_KEY')
    if api_key:
        print(f"\n✓ NCI_API_KEY found: {api_key[:8]}...")
    else:
        print("\n⚠ NCI_API_KEY not set - using mock data")
    
    # Run tests based on what's available
    try:
        # Test SDK mode (always available with mock fallback)
        await test_sdk_mode()
    except Exception as e:
        print(f"\n✗ SDK mode error: {e}")
    
    try:
        # Test MCP mode if available
        await test_mcp_mode()
    except Exception as e:
        print(f"\n✗ MCP mode error: {e}")
    
    try:
        # Test auto mode (recommended)
        await test_auto_mode()
    except Exception as e:
        print(f"\n✗ Auto mode error: {e}")
    
    try:
        # Test patient matching
        await test_patient_matching()
    except Exception as e:
        print(f"\n✗ Patient matching error: {e}")
    
    print("\n" + "=" * 70)
    print(" Testing Complete ".center(70, "="))
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
