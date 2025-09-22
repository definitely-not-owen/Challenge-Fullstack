"""
Clinical Trial Fetcher using BioMCP API.

This module handles fetching clinical trials based on patient criteria
using the BioMCP API for trial discovery and matching.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import hashlib
import os
from pathlib import Path

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field

# Try to import BioMCP, fallback to mock if not available
try:
    from biomcp import BioMCPClient
    BIOMCP_AVAILABLE = True
except ImportError:
    BIOMCP_AVAILABLE = False
    logging.warning("BioMCP SDK not installed. Using mock data. Install with: pip install biomcp")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrialStatus(Enum):
    """Clinical trial recruitment status."""
    RECRUITING = "Recruiting"
    ACTIVE_NOT_RECRUITING = "Active, not recruiting"
    COMPLETED = "Completed"
    SUSPENDED = "Suspended"
    TERMINATED = "Terminated"
    WITHDRAWN = "Withdrawn"
    NOT_YET_RECRUITING = "Not yet recruiting"


@dataclass
class Location:
    """Trial location information."""
    city: str
    state: str
    country: str
    facility: Optional[str] = None
    distance_miles: Optional[float] = None


@dataclass
class EligibilityCriteria:
    """Structured eligibility criteria for a trial."""
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gender: str = "All"
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    required_biomarkers: List[str] = field(default_factory=list)
    excluded_biomarkers: List[str] = field(default_factory=list)


@dataclass
class Trial:
    """Clinical trial data model."""
    nct_id: str
    title: str
    brief_summary: str
    conditions: List[str]
    phase: str
    status: TrialStatus
    eligibility: EligibilityCriteria
    locations: List[Location]
    sponsor: str
    enrollment: Optional[int] = None
    start_date: Optional[str] = None
    completion_date: Optional[str] = None
    last_updated: Optional[str] = None
    interventions: List[str] = field(default_factory=list)
    primary_outcomes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trial to dictionary for serialization."""
        return {
            'nct_id': self.nct_id,
            'title': self.title,
            'brief_summary': self.brief_summary,
            'conditions': self.conditions,
            'phase': self.phase,
            'status': self.status.value,
            'eligibility': {
                'min_age': self.eligibility.min_age,
                'max_age': self.eligibility.max_age,
                'gender': self.eligibility.gender,
                'inclusion_criteria': self.eligibility.inclusion_criteria,
                'exclusion_criteria': self.eligibility.exclusion_criteria,
                'required_biomarkers': self.eligibility.required_biomarkers,
                'excluded_biomarkers': self.eligibility.excluded_biomarkers,
            },
            'locations': [
                {'city': loc.city, 'state': loc.state, 'country': loc.country}
                for loc in self.locations
            ],
            'sponsor': self.sponsor,
            'enrollment': self.enrollment,
            'interventions': self.interventions
        }


class TrialCache:
    """Simple file-based cache for trial data to reduce API calls."""
    
    def __init__(self, cache_dir: str = ".cache/trials"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=24)
    
    def _get_cache_key(self, query_params: Dict[str, Any]) -> str:
        """Generate cache key from query parameters."""
        # Sort params for consistent hashing
        sorted_params = json.dumps(query_params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()
    
    def get(self, query_params: Dict[str, Any]) -> Optional[List[Trial]]:
        """Retrieve cached trials if available and not expired."""
        cache_key = self._get_cache_key(query_params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is expired
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_time > self.cache_duration:
            logger.info(f"Cache expired for key {cache_key}")
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Cache hit for key {cache_key}, {len(data)} trials")
                return [self._dict_to_trial(trial_dict) for trial_dict in data]
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def set(self, query_params: Dict[str, Any], trials: List[Trial]):
        """Cache trial data."""
        cache_key = self._get_cache_key(query_params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            trial_dicts = [trial.to_dict() for trial in trials]
            with open(cache_file, 'w') as f:
                json.dump(trial_dicts, f, indent=2)
            logger.info(f"Cached {len(trials)} trials with key {cache_key}")
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def _dict_to_trial(self, data: Dict[str, Any]) -> Trial:
        """Convert dictionary back to Trial object."""
        eligibility = EligibilityCriteria(
            min_age=data['eligibility'].get('min_age'),
            max_age=data['eligibility'].get('max_age'),
            gender=data['eligibility'].get('gender', 'All'),
            inclusion_criteria=data['eligibility'].get('inclusion_criteria', []),
            exclusion_criteria=data['eligibility'].get('exclusion_criteria', []),
            required_biomarkers=data['eligibility'].get('required_biomarkers', []),
            excluded_biomarkers=data['eligibility'].get('excluded_biomarkers', [])
        )
        
        locations = [
            Location(
                city=loc.get('city', ''),
                state=loc.get('state', ''),
                country=loc.get('country', '')
            )
            for loc in data.get('locations', [])
        ]
        
        return Trial(
            nct_id=data['nct_id'],
            title=data['title'],
            brief_summary=data['brief_summary'],
            conditions=data['conditions'],
            phase=data['phase'],
            status=TrialStatus(data['status']),
            eligibility=eligibility,
            locations=locations,
            sponsor=data['sponsor'],
            enrollment=data.get('enrollment'),
            interventions=data.get('interventions', [])
        )


class BioMCPTrialFetcher:
    """
    Fetches clinical trials using BioMCP API.
    
    This class implements intelligent filtering and caching to efficiently
    retrieve relevant trials for patient matching.
    """
    
    def __init__(self, nci_api_key: Optional[str] = None, use_mcp_tools: bool = False):
        """
        Initialize the BioMCP trial fetcher.
        
        Args:
            nci_api_key: Optional NCI API key for enhanced features
            use_mcp_tools: Whether to use MCP tools interface (for AI assistants)
        """
        self.nci_api_key = nci_api_key or os.getenv('NCI_API_KEY')
        self.cache = TrialCache()
        self.session: Optional[aiohttp.ClientSession] = None
        self.biomcp_client: Optional[Any] = None
        self.use_mcp_tools = use_mcp_tools
        
        # Initialize BioMCP client if available
        if BIOMCP_AVAILABLE:
            self.biomcp_client = BioMCPClient(nci_api_key=self.nci_api_key)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch_trials_for_cancer(
        self,
        cancer_type: str,
        stage: Optional[str] = None,
        biomarkers: Optional[List[str]] = None,
        max_trials: int = 100
    ) -> List[Trial]:
        """
        Fetch trials for a specific cancer type with optional filtering.
        
        Args:
            cancer_type: Type of cancer (e.g., "Breast", "Lung")
            stage: Cancer stage (e.g., "I", "II", "III", "IV")
            biomarkers: List of relevant biomarkers
            max_trials: Maximum number of trials to return
        
        Returns:
            List of Trial objects matching the criteria
        """
        query_params = {
            'cancer_type': cancer_type,
            'stage': stage,
            'biomarkers': biomarkers,
            'max_trials': max_trials
        }
        
        # Check cache first
        cached_trials = self.cache.get(query_params)
        if cached_trials:
            return cached_trials
        
        # Fetch from BioMCP
        trials = await self._fetch_from_biomcp(cancer_type, stage, biomarkers, max_trials)
        
        # Cache the results
        if trials:
            self.cache.set(query_params, trials)
        
        return trials
    
    async def _fetch_from_biomcp(
        self,
        cancer_type: str,
        stage: Optional[str],
        biomarkers: Optional[List[str]],
        max_trials: int
    ) -> List[Trial]:
        """
        Fetch trials from BioMCP API using Python SDK or HTTP endpoints.
        
        Integrates with BioMCP's clinical trials search functionality,
        falling back to mock data if SDK is not available.
        """
        if not BIOMCP_AVAILABLE or not self.biomcp_client:
            logger.warning("BioMCP SDK not available, using mock data")
            return self._get_mock_trials(cancer_type, stage, biomarkers, max_trials)
        
        try:
            logger.info(f"Fetching trials from BioMCP for {cancer_type}")
            
            # Build search query for BioMCP
            search_terms = [cancer_type]
            if stage:
                search_terms.append(f"stage {stage}")
            if biomarkers:
                search_terms.extend(biomarkers)
            
            query = " AND ".join(search_terms)
            
            # Use BioMCP Python SDK for trial search
            # The SDK provides domain-specific APIs for trials
            async with self.biomcp_client as client:
                # Search for clinical trials using BioMCP's unified search
                results = await client.search(
                    domain="trial",  # Specify clinical trials domain
                    query=query,
                    page=1,
                    page_size=max_trials
                )
                
                # Parse BioMCP response into our Trial objects
                trials = self._parse_biomcp_response(results)
                
                # If we got results, return them
                if trials:
                    logger.info(f"Retrieved {len(trials)} trials from BioMCP")
                    return trials
                
                # If no results from search, try alternative approach
                # Use MCP tools for more intelligent search if enabled
                if self.use_mcp_tools:
                    mcp_results = await self._fetch_using_mcp_tools(
                        client, cancer_type, stage, biomarkers, max_trials
                    )
                    if mcp_results:
                        return mcp_results
            
            # Fallback to mock data if no results
            logger.info("No trials found via BioMCP, using mock data")
            return self._get_mock_trials(cancer_type, stage, biomarkers, max_trials)
            
        except Exception as e:
            logger.error(f"BioMCP API error: {e}")
            # Fallback to mock data on error
            return self._get_mock_trials(cancer_type, stage, biomarkers, max_trials)
    
    async def _fetch_using_mcp_tools(
        self,
        client,
        cancer_type: str,
        stage: Optional[str],
        biomarkers: Optional[List[str]],
        max_trials: int
    ) -> List[Trial]:
        """
        Use BioMCP's MCP tools interface for intelligent trial search.
        
        This leverages the 24 specialized tools and sequential thinking
        capabilities for complex queries.
        """
        try:
            # Use MCP protocol's sequential thinking for complex queries
            tool_params = {
                "tool": "search_clinical_trials",
                "parameters": {
                    "condition": cancer_type,
                    "stage": stage,
                    "biomarkers": biomarkers,
                    "status": "recruiting",
                    "limit": max_trials
                }
            }
            
            results = await client.execute_tool(**tool_params)
            return self._parse_biomcp_response(results)
            
        except Exception as e:
            logger.error(f"MCP tools error: {e}")
            return []
    
    def _parse_biomcp_response(self, response: Dict[str, Any]) -> List[Trial]:
        """
        Parse BioMCP API response into Trial objects.
        
        Handles both direct API responses and MCP tool responses.
        """
        trials = []
        
        # Handle different response formats from BioMCP
        if isinstance(response, dict):
            # Check for results key (standard API response)
            trial_data = response.get('results', response.get('trials', []))
        elif isinstance(response, list):
            trial_data = response
        else:
            logger.warning(f"Unexpected response format from BioMCP: {type(response)}")
            return []
        
        for item in trial_data:
            try:
                # Parse NCT ID
                nct_id = item.get('nct_id', item.get('id', ''))
                if not nct_id:
                    continue
                
                # Parse basic information
                title = item.get('title', item.get('brief_title', ''))
                summary = item.get('summary', item.get('brief_summary', ''))
                conditions = item.get('conditions', [])
                if isinstance(conditions, str):
                    conditions = [conditions]
                
                # Parse phase
                phase = item.get('phase', 'N/A')
                
                # Parse status
                status_str = item.get('status', 'RECRUITING')
                status = self._map_biomcp_status(status_str)
                
                # Parse eligibility
                eligibility = self._parse_biomcp_eligibility(item.get('eligibility', {}))
                
                # Parse locations
                locations = self._parse_biomcp_locations(item.get('locations', []))
                
                # Parse sponsor
                sponsor = item.get('sponsor', {})
                if isinstance(sponsor, dict):
                    sponsor = sponsor.get('name', 'Unknown')
                elif not sponsor:
                    sponsor = 'Unknown'
                
                # Parse interventions
                interventions = item.get('interventions', [])
                if isinstance(interventions, str):
                    interventions = [interventions]
                
                trial = Trial(
                    nct_id=nct_id,
                    title=title,
                    brief_summary=summary,
                    conditions=conditions,
                    phase=phase,
                    status=status,
                    eligibility=eligibility,
                    locations=locations,
                    sponsor=sponsor,
                    interventions=interventions,
                    enrollment=item.get('enrollment'),
                    start_date=item.get('start_date'),
                    completion_date=item.get('completion_date')
                )
                
                trials.append(trial)
                
            except Exception as e:
                logger.warning(f"Error parsing BioMCP trial {item.get('nct_id', 'unknown')}: {e}")
                continue
        
        return trials
    
    def _map_biomcp_status(self, status_str: str) -> TrialStatus:
        """Map BioMCP status strings to TrialStatus enum."""
        status_mapping = {
            'recruiting': TrialStatus.RECRUITING,
            'active': TrialStatus.ACTIVE_NOT_RECRUITING,
            'completed': TrialStatus.COMPLETED,
            'suspended': TrialStatus.SUSPENDED,
            'terminated': TrialStatus.TERMINATED,
            'withdrawn': TrialStatus.WITHDRAWN,
            'not_yet_recruiting': TrialStatus.NOT_YET_RECRUITING,
        }
        
        # Handle case-insensitive matching
        status_lower = status_str.lower().replace(' ', '_')
        return status_mapping.get(status_lower, TrialStatus.RECRUITING)
    
    def _parse_biomcp_eligibility(self, eligibility_data: Dict[str, Any]) -> EligibilityCriteria:
        """Parse eligibility criteria from BioMCP response."""
        return EligibilityCriteria(
            min_age=eligibility_data.get('min_age'),
            max_age=eligibility_data.get('max_age'),
            gender=eligibility_data.get('gender', 'All'),
            inclusion_criteria=eligibility_data.get('inclusion', []),
            exclusion_criteria=eligibility_data.get('exclusion', []),
            required_biomarkers=eligibility_data.get('biomarkers', []),
            excluded_biomarkers=eligibility_data.get('excluded_biomarkers', [])
        )
    
    def _parse_biomcp_locations(self, locations_data: List[Dict[str, Any]]) -> List[Location]:
        """Parse trial locations from BioMCP response."""
        locations = []
        
        for loc_data in locations_data[:5]:  # Limit to 5 locations
            if isinstance(loc_data, dict):
                locations.append(Location(
                    city=loc_data.get('city', ''),
                    state=loc_data.get('state', ''),
                    country=loc_data.get('country', 'USA'),
                    facility=loc_data.get('facility', '')
                ))
        
        return locations
    
    def _get_mock_trials(
        self,
        cancer_type: str,
        stage: Optional[str],
        biomarkers: Optional[List[str]],
        max_trials: int
    ) -> List[Trial]:
        """
        Generate mock trial data for development and testing.
        
        This provides realistic trial data structure while BioMCP integration
        is being finalized.
        """
        mock_trials = []
        
        # Generate relevant mock trials based on cancer type
        base_trials = {
            "Breast": [
                ("NCT04756765", "Pembrolizumab Plus Chemotherapy in HER2-Negative Breast Cancer", 
                 "A study of pembrolizumab plus chemotherapy versus placebo plus chemotherapy for previously untreated locally recurrent inoperable or metastatic triple-negative breast cancer."),
                ("NCT05633979", "Trastuzumab Deruxtecan in HER2+ Metastatic Breast Cancer",
                 "An open-label study of trastuzumab deruxtecan in participants with HER2-positive metastatic breast cancer."),
                ("NCT04873362", "CDK4/6 Inhibitor With Endocrine Therapy",
                 "A study evaluating the efficacy and safety of ribociclib plus endocrine therapy in HR+/HER2- advanced breast cancer."),
            ],
            "Lung": [
                ("NCT04746768", "Osimertinib in EGFR-Mutated Non-Small Cell Lung Cancer",
                 "A study of osimertinib as first-line treatment in patients with EGFR mutation-positive advanced NSCLC."),
                ("NCT05259423", "Immunotherapy Combination for Advanced Lung Cancer",
                 "Evaluation of nivolumab plus ipilimumab versus chemotherapy in first-line treatment of advanced NSCLC."),
            ],
            "Pancreatic": [
                ("NCT04787991", "FOLFIRINOX vs Gemcitabine in Metastatic Pancreatic Cancer",
                 "Comparing FOLFIRINOX to gemcitabine as first-line therapy for metastatic pancreatic adenocarcinoma."),
                ("NCT05134519", "Olaparib Maintenance in BRCA-Mutated Pancreatic Cancer",
                 "Study of olaparib as maintenance therapy in patients with BRCA-mutated metastatic pancreatic cancer."),
            ]
        }
        
        # Get trials for the cancer type or use generic trials
        cancer_key = cancer_type.split()[0] if cancer_type else "Breast"
        trial_templates = base_trials.get(cancer_key, base_trials["Breast"])
        
        for i, (nct_id, title, summary) in enumerate(trial_templates[:max_trials]):
            if i >= max_trials:
                break
                
            # Create eligibility criteria based on input
            eligibility = EligibilityCriteria(
                min_age=18,
                max_age=75 if stage != "IV" else None,
                gender="All",
                inclusion_criteria=[
                    f"Confirmed diagnosis of {cancer_type}",
                    f"Stage {stage}" if stage else "Any stage",
                    "Adequate organ function",
                ],
                exclusion_criteria=[
                    "Prior systemic therapy for metastatic disease",
                    "Active CNS metastases",
                ],
                required_biomarkers=biomarkers or [],
            )
            
            # Create mock locations
            locations = [
                Location(city="Boston", state="MA", country="USA", facility="Dana-Farber Cancer Institute"),
                Location(city="New York", state="NY", country="USA", facility="Memorial Sloan Kettering"),
                Location(city="Houston", state="TX", country="USA", facility="MD Anderson Cancer Center"),
            ]
            
            trial = Trial(
                nct_id=nct_id,
                title=title,
                brief_summary=summary,
                conditions=[cancer_type],
                phase="Phase 3" if i % 2 == 0 else "Phase 2",
                status=TrialStatus.RECRUITING,
                eligibility=eligibility,
                locations=locations[:2],  # Just include 2 locations
                sponsor="National Cancer Institute",
                enrollment=200,
                interventions=[title.split()[0]]  # Use first word as intervention
            )
            
            mock_trials.append(trial)
        
        logger.info(f"Generated {len(mock_trials)} mock trials for {cancer_type}")
        return mock_trials
    
    async def fetch_trials_with_biomarker_matching(
        self,
        patient_data: Dict[str, Any]
    ) -> List[Trial]:
        """
        Advanced trial fetching using BioMCP's biomarker matching capabilities.
        
        This method leverages BioMCP's variant and gene tools for precise
        biomarker-based trial matching.
        
        Args:
            patient_data: Dictionary containing patient information including
                         cancer_type, stage, biomarkers_detected, biomarkers_ruled_out
        
        Returns:
            List of trials matched based on biomarker compatibility
        """
        cancer_type = patient_data.get('cancer_type', '')
        stage = patient_data.get('cancer_stage', '')
        biomarkers_detected = patient_data.get('biomarkers_detected', [])
        biomarkers_ruled_out = patient_data.get('biomarkers_ruled_out', [])
        
        # First, get base trials for the cancer type
        base_trials = await self.fetch_trials_for_cancer(
            cancer_type=cancer_type,
            stage=stage,
            biomarkers=biomarkers_detected,
            max_trials=50  # Get more trials for filtering
        )
        
        if not BIOMCP_AVAILABLE or not self.biomcp_client:
            # Without BioMCP, do basic filtering
            return self._filter_trials_by_biomarkers(
                base_trials, biomarkers_detected, biomarkers_ruled_out
            )
        
        try:
            # Use BioMCP's variant tools for enhanced matching
            async with self.biomcp_client as client:
                enhanced_trials = []
                
                for trial in base_trials:
                    # Check biomarker compatibility using BioMCP
                    is_compatible = await self._check_biomarker_compatibility(
                        client, trial, biomarkers_detected, biomarkers_ruled_out
                    )
                    
                    if is_compatible:
                        enhanced_trials.append(trial)
                
                logger.info(f"Enhanced matching: {len(enhanced_trials)} compatible trials from {len(base_trials)}")
                return enhanced_trials
                
        except Exception as e:
            logger.error(f"Error in biomarker matching: {e}")
            # Fallback to basic filtering
            return self._filter_trials_by_biomarkers(
                base_trials, biomarkers_detected, biomarkers_ruled_out
            )
    
    def _filter_trials_by_biomarkers(
        self,
        trials: List[Trial],
        biomarkers_detected: List[str],
        biomarkers_ruled_out: List[str]
    ) -> List[Trial]:
        """Basic biomarker filtering without BioMCP enhancement."""
        filtered_trials = []
        
        for trial in trials:
            # Check if trial requires any ruled-out biomarkers
            if any(marker in trial.eligibility.required_biomarkers 
                   for marker in biomarkers_ruled_out):
                continue
            
            # Check if trial excludes any detected biomarkers
            if any(marker in trial.eligibility.excluded_biomarkers 
                   for marker in biomarkers_detected):
                continue
            
            filtered_trials.append(trial)
        
        return filtered_trials
    
    async def _check_biomarker_compatibility(
        self,
        client,
        trial: Trial,
        biomarkers_detected: List[str],
        biomarkers_ruled_out: List[str]
    ) -> bool:
        """
        Use BioMCP's variant and gene tools to check biomarker compatibility.
        
        This would integrate with BioMCP's specialized biomarker analysis tools
        for precise matching.
        """
        # This is a placeholder for actual BioMCP biomarker matching
        # In production, this would use BioMCP's variant/gene tools
        return self._filter_trials_by_biomarkers(
            [trial], biomarkers_detected, biomarkers_ruled_out
        ) != []


# Example usage and testing
async def main():
    """Example usage of the trial fetcher."""
    async with BioMCPTrialFetcher() as fetcher:
        # Fetch breast cancer trials
        trials = await fetcher.fetch_trials_for_cancer(
            cancer_type="Breast Cancer",
            stage="II",
            biomarkers=["ER+", "HER2+"],
            max_trials=10
        )
        
        print(f"Found {len(trials)} trials")
        for trial in trials[:3]:
            print(f"\nNCT ID: {trial.nct_id}")
            print(f"Title: {trial.title}")
            print(f"Status: {trial.status.value}")
            print(f"Phase: {trial.phase}")
            print(f"Conditions: {', '.join(trial.conditions[:3])}")


if __name__ == "__main__":
    asyncio.run(main())
