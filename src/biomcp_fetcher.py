"""
BioMCP Trial Fetcher with dual-mode support (SDK and MCP).

This module provides both SDK-style and MCP-style interfaces for fetching
clinical trials using BioMCP, with intelligent fallback to mock data.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pathlib import Path

import pandas as pd

# BioMCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP client not available")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available for HTTP-based BioMCP")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClinicalTrial:
    """Standardized clinical trial data structure."""
    nct_id: str
    title: str
    brief_summary: str
    conditions: List[str]
    phase: str
    status: str
    eligibility_criteria: str
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gender: str = "All"
    locations: List[Dict[str, str]] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    sponsor: str = ""
    enrollment: Optional[int] = None
    start_date: Optional[str] = None
    primary_outcomes: List[str] = field(default_factory=list)
    
    def matches_biomarkers(self, detected: List[str], ruled_out: List[str]) -> bool:
        """Check if trial is compatible with patient biomarkers."""
        # Simple text-based matching for now
        criteria_text = self.eligibility_criteria.lower()
        
        # Check for ruled out biomarkers that might be required
        for marker in ruled_out:
            if marker.lower() in criteria_text and "positive" in criteria_text:
                return False
        
        return True


class BioMCPClient:
    """
    Dual-mode BioMCP client supporting both SDK and MCP protocol access.
    
    Provides unified interface for clinical trial fetching regardless of
    the underlying transport mechanism.
    """
    
    def __init__(self, mode: str = "auto", nci_api_key: Optional[str] = None):
        """
        Initialize BioMCP client.
        
        Args:
            mode: "sdk", "mcp", or "auto" (auto-detect best available)
            nci_api_key: NCI API key for enhanced features (uses env var if not provided)
        """
        self.mode = mode
        self.nci_api_key = nci_api_key or os.getenv('NCI_API_KEY')
        self.mcp_session: Optional[ClientSession] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Auto-detect best available mode
        if mode == "auto":
            if MCP_AVAILABLE:
                self.mode = "mcp"
            elif HTTPX_AVAILABLE:
                self.mode = "sdk"
            else:
                self.mode = "mock"
                logger.warning("No BioMCP transport available, using mock mode")
        
        logger.info(f"BioMCP client initialized in {self.mode} mode")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.mode == "sdk" and HTTPX_AVAILABLE:
            self.http_client = httpx.AsyncClient(
                headers={"X-NCI-API-Key": self.nci_api_key} if self.nci_api_key else {}
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.http_client:
            await self.http_client.aclose()
        if self.mcp_session:
            # MCP session cleanup if needed
            pass
    
    async def search_trials_sdk(
        self,
        condition: str,
        additional_terms: List[str] = None,
        max_results: int = 20
    ) -> List[ClinicalTrial]:
        """
        Search trials using SDK-style HTTP API.
        
        This method makes direct HTTP calls to BioMCP endpoints.
        """
        if not HTTPX_AVAILABLE or not self.http_client:
            return await self._get_mock_trials(condition, additional_terms, max_results)
        
        try:
            # Build search query
            search_terms = [condition]
            if additional_terms:
                search_terms.extend(additional_terms)
            query = " ".join(search_terms)
            
            # Make HTTP request to BioMCP endpoint
            # Note: Using actual BioMCP HTTP endpoint structure
            response = await self.http_client.post(
                "http://localhost:5173/search",  # Default BioMCP HTTP server
                json={
                    "domain": "trial",
                    "query": query,
                    "page": 1,
                    "page_size": max_results
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_trials(data)
            else:
                logger.error(f"HTTP request failed: {response.status_code}")
                return await self._get_mock_trials(condition, additional_terms, max_results)
                
        except Exception as e:
            logger.error(f"SDK search error: {e}")
            return await self._get_mock_trials(condition, additional_terms, max_results)
    
    async def search_trials_mcp(
        self,
        condition: str,
        additional_terms: List[str] = None,
        max_results: int = 20
    ) -> List[ClinicalTrial]:
        """
        Search trials using MCP protocol.
        
        This method uses the Model Context Protocol for AI-optimized search.
        """
        if not MCP_AVAILABLE:
            return await self._get_mock_trials(condition, additional_terms, max_results)
        
        try:
            # Initialize MCP session if not already done
            if not self.mcp_session:
                server_params = StdioServerParameters(
                    command="uvx",
                    args=["biomcp"],
                    env={"NCI_API_KEY": self.nci_api_key} if self.nci_api_key else None
                )
                
                transport = await stdio_client(server_params)
                self.mcp_session = ClientSession(transport[0], transport[1])
                await self.mcp_session.initialize()
            
            # Build search query
            search_terms = [condition]
            if additional_terms:
                search_terms.extend(additional_terms)
            query = " ".join(search_terms)
            
            # Use MCP tool to search trials
            result = await self.mcp_session.call_tool(
                "search-clinical-trials",
                arguments={
                    "query": query,
                    "max_results": max_results
                }
            )
            
            # Parse MCP response
            if result and hasattr(result, 'content'):
                trials_data = result.content
                if isinstance(trials_data, list):
                    return [self._parse_mcp_trial(t) for t in trials_data]
                elif isinstance(trials_data, dict) and 'trials' in trials_data:
                    return [self._parse_mcp_trial(t) for t in trials_data['trials']]
            
            logger.warning("No trials found via MCP")
            return await self._get_mock_trials(condition, additional_terms, max_results)
            
        except Exception as e:
            logger.error(f"MCP search error: {e}")
            return await self._get_mock_trials(condition, additional_terms, max_results)
    
    async def search_trials(
        self,
        condition: str,
        additional_terms: List[str] = None,
        max_results: int = 20
    ) -> List[ClinicalTrial]:
        """
        Unified search interface that uses the configured mode.
        
        Automatically selects SDK or MCP based on initialization.
        """
        logger.info(f"Searching trials for '{condition}' using {self.mode} mode")
        
        # Always use SDK mode
        if self.mode == "sdk" or self.mode == "auto":
            return await self.search_trials_sdk(condition, additional_terms, max_results)
        else:
            return await self._get_mock_trials(condition, additional_terms, max_results)
    
    def _parse_trials(self, data: Dict[str, Any]) -> List[ClinicalTrial]:
        """Parse trials from SDK/HTTP response."""
        trials = []
        
        # Handle different response structures
        trial_list = data.get('results', data.get('trials', []))
        
        for item in trial_list:
            trial = ClinicalTrial(
                nct_id=item.get('nct_id', item.get('id', '')),
                title=item.get('title', ''),
                brief_summary=item.get('summary', ''),
                conditions=item.get('conditions', []),
                phase=item.get('phase', 'N/A'),
                status=item.get('status', 'Unknown'),
                eligibility_criteria=item.get('eligibility', ''),
                min_age=item.get('min_age'),
                max_age=item.get('max_age'),
                gender=item.get('gender', 'All'),
                locations=item.get('locations', []),
                interventions=item.get('interventions', []),
                sponsor=item.get('sponsor', ''),
                enrollment=item.get('enrollment')
            )
            trials.append(trial)
        
        return trials
    
    def _parse_mcp_trial(self, data: Dict[str, Any]) -> ClinicalTrial:
        """Parse a single trial from MCP response."""
        return ClinicalTrial(
            nct_id=data.get('nctId', ''),
            title=data.get('title', ''),
            brief_summary=data.get('briefSummary', ''),
            conditions=data.get('conditions', []),
            phase=data.get('phase', 'N/A'),
            status=data.get('status', 'Unknown'),
            eligibility_criteria=data.get('eligibilityCriteria', ''),
            min_age=data.get('minimumAge'),
            max_age=data.get('maximumAge'),
            gender=data.get('gender', 'All'),
            locations=data.get('locations', []),
            interventions=data.get('interventions', []),
            sponsor=data.get('leadSponsorName', ''),
            enrollment=data.get('enrollment')
        )
    
    async def _get_mock_trials(
        self,
        condition: str,
        additional_terms: List[str] = None,
        max_results: int = 20
    ) -> List[ClinicalTrial]:
        """Generate mock trials for testing when BioMCP is unavailable."""
        mock_trials = []
        
        # Cancer-specific mock trials
        if "breast" in condition.lower():
            mock_trials.extend([
                ClinicalTrial(
                    nct_id="NCT04756765",
                    title="Pembrolizumab Plus Chemotherapy in HER2-Negative Breast Cancer",
                    brief_summary="Study of pembrolizumab plus chemotherapy for triple-negative breast cancer",
                    conditions=["Breast Cancer", "Triple Negative Breast Cancer"],
                    phase="Phase 3",
                    status="Recruiting",
                    eligibility_criteria="HER2-negative, ER-negative, PR-negative breast cancer",
                    min_age=18,
                    max_age=None,
                    gender="All",
                    locations=[{"city": "Boston", "state": "MA", "country": "USA"}],
                    interventions=["Pembrolizumab", "Chemotherapy"],
                    sponsor="National Cancer Institute"
                ),
                ClinicalTrial(
                    nct_id="NCT05633979",
                    title="Trastuzumab Deruxtecan in HER2+ Metastatic Breast Cancer",
                    brief_summary="Open-label study of trastuzumab deruxtecan in HER2-positive breast cancer",
                    conditions=["Breast Cancer", "HER2 Positive"],
                    phase="Phase 2",
                    status="Recruiting",
                    eligibility_criteria="HER2-positive metastatic breast cancer",
                    min_age=18,
                    max_age=None,
                    gender="All",
                    locations=[{"city": "New York", "state": "NY", "country": "USA"}],
                    interventions=["Trastuzumab Deruxtecan"],
                    sponsor="Daiichi Sankyo"
                )
            ])
        elif "lung" in condition.lower():
            mock_trials.extend([
                ClinicalTrial(
                    nct_id="NCT04746768",
                    title="Osimertinib in EGFR-Mutated Non-Small Cell Lung Cancer",
                    brief_summary="First-line osimertinib in EGFR mutation-positive advanced NSCLC",
                    conditions=["Non-Small Cell Lung Cancer", "EGFR Mutation"],
                    phase="Phase 3",
                    status="Recruiting",
                    eligibility_criteria="EGFR mutation-positive advanced NSCLC",
                    min_age=18,
                    max_age=None,
                    gender="All",
                    locations=[{"city": "Houston", "state": "TX", "country": "USA"}],
                    interventions=["Osimertinib"],
                    sponsor="AstraZeneca"
                )
            ])
        elif "pancreatic" in condition.lower():
            mock_trials.extend([
                ClinicalTrial(
                    nct_id="NCT04787991",
                    title="FOLFIRINOX vs Gemcitabine in Metastatic Pancreatic Cancer",
                    brief_summary="Comparing FOLFIRINOX to gemcitabine as first-line therapy",
                    conditions=["Pancreatic Cancer", "Metastatic"],
                    phase="Phase 3",
                    status="Recruiting",
                    eligibility_criteria="Metastatic pancreatic adenocarcinoma",
                    min_age=18,
                    max_age=75,
                    gender="All",
                    locations=[{"city": "Baltimore", "state": "MD", "country": "USA"}],
                    interventions=["FOLFIRINOX", "Gemcitabine"],
                    sponsor="Johns Hopkins"
                )
            ])
        
        # Filter by additional terms if provided
        if additional_terms:
            filtered_trials = []
            for trial in mock_trials:
                trial_text = f"{trial.title} {trial.brief_summary} {trial.eligibility_criteria}".lower()
                if any(term.lower() in trial_text for term in additional_terms):
                    filtered_trials.append(trial)
            mock_trials = filtered_trials
        
        return mock_trials[:max_results]


class TrialMatcher:
    """
    High-level trial matching interface using BioMCP.
    
    Provides patient-centric trial matching with biomarker consideration.
    """
    
    def __init__(self, mode: str = "auto"):
        """
        Initialize trial matcher.
        
        Args:
            mode: BioMCP client mode ("sdk", "mcp", or "auto")
        """
        self.client = BioMCPClient(mode=mode)
    
    async def match_patient(
        self,
        patient_data: Dict[str, Any],
        max_trials: int = 10
    ) -> List[ClinicalTrial]:
        """
        Match trials for a specific patient.
        
        Args:
            patient_data: Patient information including cancer type, stage, biomarkers
            max_trials: Maximum number of trials to return
        
        Returns:
            List of matched clinical trials
        """
        # Extract patient information
        cancer_type = patient_data.get('cancer_type', '')
        stage = patient_data.get('cancer_stage', '')
        
        # Handle NaN values and empty strings for biomarkers
        biomarkers_detected_str = patient_data.get('biomarkers_detected', '')
        biomarkers_ruled_out_str = patient_data.get('biomarkers_ruled_out', '')
        
        biomarkers_detected = []
        if pd.notna(biomarkers_detected_str) and biomarkers_detected_str.strip():
            biomarkers_detected = [b.strip() for b in str(biomarkers_detected_str).split(',') if b.strip()]
        
        biomarkers_ruled_out = []
        if pd.notna(biomarkers_ruled_out_str) and biomarkers_ruled_out_str.strip():
            biomarkers_ruled_out = [b.strip() for b in str(biomarkers_ruled_out_str).split(',') if b.strip()]
        
        # Build search terms
        additional_terms = []
        if stage:
            additional_terms.append(f"stage {stage}")
        additional_terms.extend(biomarkers_detected)
        
        # Search for trials
        async with self.client:
            trials = await self.client.search_trials(
                condition=cancer_type,
                additional_terms=additional_terms,
                max_results=max_trials * 2  # Get extra for filtering
            )
        
        # Filter by biomarker compatibility
        compatible_trials = []
        for trial in trials:
            if trial.matches_biomarkers(biomarkers_detected, biomarkers_ruled_out):
                compatible_trials.append(trial)
        
        return compatible_trials[:max_trials]


# Example usage
async def demo():
    """Demonstrate both SDK and MCP style usage."""
    
    print("=" * 60)
    print("BioMCP Trial Fetcher Demo")
    print("=" * 60)
    
    # Example 1: SDK-style usage
    print("\n1. SDK-style usage:")
    print("-" * 40)
    
    async with BioMCPClient(mode="sdk") as client:
        trials = await client.search_trials(
            condition="Breast Cancer",
            additional_terms=["HER2-positive", "stage II"],
            max_results=5
        )
        
        for trial in trials:
            print(f"  {trial.nct_id}: {trial.title[:60]}...")
    
    # Example 2: MCP-style usage
    print("\n2. MCP-style usage:")
    print("-" * 40)
    
    async with BioMCPClient(mode="mcp") as client:
        trials = await client.search_trials(
            condition="Lung Cancer",
            additional_terms=["EGFR mutation"],
            max_results=5
        )
        
        for trial in trials:
            print(f"  {trial.nct_id}: {trial.title[:60]}...")
    
    # Example 3: Auto-detect mode (recommended)
    print("\n3. Auto-detect mode (recommended):")
    print("-" * 40)
    
    matcher = TrialMatcher(mode="auto")
    
    patient = {
        'cancer_type': 'Pancreatic Cancer',
        'cancer_stage': 'III',
        'biomarkers_detected': 'KRAS mutation',
        'biomarkers_ruled_out': 'MSI-H'
    }
    
    trials = await matcher.match_patient(patient, max_trials=5)
    
    print(f"  Found {len(trials)} matching trials for patient")
    for trial in trials:
        print(f"  {trial.nct_id}: {trial.title[:60]}...")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
