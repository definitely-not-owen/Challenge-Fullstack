"""
Deterministic Filter Engine for Clinical Trial Pre-screening.

This module implements rule-based filters to remove obvious mismatches
before expensive LLM scoring, improving safety and efficiency.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from enum import Enum
import re

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterResult(Enum):
    """Result of a filter check."""
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class FilterDecision:
    """Records a single filter decision for transparency."""
    filter_name: str
    result: FilterResult
    reason: str
    details: Dict[str, Any] = None


@dataclass
class FilteredTrial:
    """Trial with filter decisions attached."""
    trial: Any  # ClinicalTrial object
    passed: bool
    decisions: List[FilterDecision]
    
    def get_filter_summary(self) -> Dict[str, bool]:
        """Get summary of all filter decisions."""
        return {
            decision.filter_name: decision.result == FilterResult.PASS
            for decision in self.decisions
        }


class DeterministicFilter:
    """
    Applies deterministic rules to filter trials before LLM scoring.
    
    Filters include:
    - Age eligibility
    - Gender compatibility
    - Trial status (active/recruiting)
    - Geographic proximity
    - Biomarker hard exclusions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize filter with configuration.
        
        Args:
            config: Filter configuration including thresholds
        """
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default filter configuration."""
        return {
            'max_distance_miles': 500,  # Geographic threshold
            'allow_gender_all': True,    # Accept trials open to all genders
            'excluded_statuses': [
                'Withdrawn', 'Terminated', 'Suspended', 'Completed'
            ],
            'strict_biomarker_matching': False,  # If True, requires exact match
        }
    
    def apply_filters(
        self,
        patient: Dict[str, Any],
        trials: List[Any]
    ) -> Tuple[List[Any], List[FilteredTrial]]:
        """
        Apply all filters to a list of trials.
        
        Args:
            patient: Patient data dictionary
            trials: List of trial objects
            
        Returns:
            Tuple of (passed_trials, all_filtered_trials_with_decisions)
        """
        filtered_results = []
        passed_trials = []
        
        for trial in trials:
            filtered_trial = self._filter_single_trial(patient, trial)
            filtered_results.append(filtered_trial)
            
            if filtered_trial.passed:
                passed_trials.append(trial)
                logger.debug(f"Trial {trial.nct_id} passed all filters")
            else:
                failed_filters = [
                    d.filter_name for d in filtered_trial.decisions 
                    if d.result == FilterResult.FAIL
                ]
                logger.debug(f"Trial {trial.nct_id} failed filters: {failed_filters}")
        
        logger.info(
            f"Filtered {len(trials)} trials: {len(passed_trials)} passed, "
            f"{len(trials) - len(passed_trials)} failed"
        )
        
        return passed_trials, filtered_results
    
    def _filter_single_trial(
        self,
        patient: Dict[str, Any],
        trial: Any
    ) -> FilteredTrial:
        """Apply all filters to a single trial."""
        decisions = []
        
        # Age filter
        age_decision = self._check_age_eligibility(patient, trial)
        decisions.append(age_decision)
        
        # Gender filter
        gender_decision = self._check_gender_compatibility(patient, trial)
        decisions.append(gender_decision)
        
        # Trial status filter
        status_decision = self._check_trial_status(trial)
        decisions.append(status_decision)
        
        # Geographic filter
        geo_decision = self._check_geographic_proximity(patient, trial)
        decisions.append(geo_decision)
        
        # Biomarker exclusion filter
        biomarker_decision = self._check_biomarker_exclusions(patient, trial)
        decisions.append(biomarker_decision)
        
        # Determine overall pass/fail
        passed = all(
            d.result != FilterResult.FAIL 
            for d in decisions
        )
        
        return FilteredTrial(
            trial=trial,
            passed=passed,
            decisions=decisions
        )
    
    def _check_age_eligibility(
        self,
        patient: Dict[str, Any],
        trial: Any
    ) -> FilterDecision:
        """Check if patient age meets trial requirements."""
        patient_age = patient.get('age')
        
        if patient_age is None:
            return FilterDecision(
                filter_name="age_eligibility",
                result=FilterResult.UNKNOWN,
                reason="Patient age not specified"
            )
        
        # Handle NaN values
        if pd.isna(patient_age):
            return FilterDecision(
                filter_name="age_eligibility",
                result=FilterResult.UNKNOWN,
                reason="Patient age is NaN"
            )
        
        patient_age = float(patient_age)
        
        # Get trial age limits
        min_age = getattr(trial, 'min_age', None) or 0
        max_age = getattr(trial, 'max_age', None) or 120
        
        # Check eligibility
        if patient_age < min_age:
            return FilterDecision(
                filter_name="age_eligibility",
                result=FilterResult.FAIL,
                reason=f"Patient age {patient_age} below minimum {min_age}",
                details={'patient_age': patient_age, 'min_age': min_age}
            )
        
        if patient_age > max_age:
            return FilterDecision(
                filter_name="age_eligibility",
                result=FilterResult.FAIL,
                reason=f"Patient age {patient_age} above maximum {max_age}",
                details={'patient_age': patient_age, 'max_age': max_age}
            )
        
        return FilterDecision(
            filter_name="age_eligibility",
            result=FilterResult.PASS,
            reason=f"Age {patient_age} within range {min_age}-{max_age}",
            details={'patient_age': patient_age, 'min_age': min_age, 'max_age': max_age}
        )
    
    def _check_gender_compatibility(
        self,
        patient: Dict[str, Any],
        trial: Any
    ) -> FilterDecision:
        """Check if patient gender is eligible for trial."""
        patient_gender = patient.get('gender', '').lower()
        trial_gender = getattr(trial, 'gender', 'All').lower()
        
        if not patient_gender:
            return FilterDecision(
                filter_name="gender_compatibility",
                result=FilterResult.UNKNOWN,
                reason="Patient gender not specified"
            )
        
        # Normalize gender values
        patient_gender = patient_gender.replace('female', 'f').replace('male', 'm')[0]
        
        # Check compatibility
        if trial_gender in ['all', 'both']:
            return FilterDecision(
                filter_name="gender_compatibility",
                result=FilterResult.PASS,
                reason="Trial accepts all genders"
            )
        
        trial_gender_normalized = trial_gender.replace('female', 'f').replace('male', 'm')[0] if trial_gender else ''
        
        if patient_gender == trial_gender_normalized:
            return FilterDecision(
                filter_name="gender_compatibility",
                result=FilterResult.PASS,
                reason=f"Gender match: {patient_gender}"
            )
        
        return FilterDecision(
            filter_name="gender_compatibility",
            result=FilterResult.FAIL,
            reason=f"Gender mismatch: patient is {patient_gender}, trial requires {trial_gender}",
            details={'patient_gender': patient_gender, 'trial_gender': trial_gender}
        )
    
    def _check_trial_status(self, trial: Any) -> FilterDecision:
        """Check if trial is actively recruiting."""
        trial_status = getattr(trial, 'status', '')
        
        if not trial_status:
            return FilterDecision(
                filter_name="trial_status",
                result=FilterResult.UNKNOWN,
                reason="Trial status not specified"
            )
        
        # Normalize status string
        if hasattr(trial_status, 'value'):
            status_str = trial_status.value
        else:
            status_str = str(trial_status)
        
        # Check against excluded statuses
        if any(excluded in status_str for excluded in self.config['excluded_statuses']):
            return FilterDecision(
                filter_name="trial_status",
                result=FilterResult.FAIL,
                reason=f"Trial status '{status_str}' is not recruiting",
                details={'status': status_str}
            )
        
        # Check for active recruitment
        if 'recruiting' in status_str.lower():
            return FilterDecision(
                filter_name="trial_status",
                result=FilterResult.PASS,
                reason=f"Trial is actively recruiting: {status_str}"
            )
        
        # Unknown or ambiguous status - let it pass for LLM to evaluate
        return FilterDecision(
            filter_name="trial_status",
            result=FilterResult.PASS,
            reason=f"Trial status '{status_str}' may be eligible",
            details={'status': status_str}
        )
    
    def _check_geographic_proximity(
        self,
        patient: Dict[str, Any],
        trial: Any
    ) -> FilterDecision:
        """Check if trial locations are within acceptable distance."""
        # For demo purposes, we'll implement a simple check
        # In production, this would calculate actual distances
        
        patient_state = patient.get('state', '')
        trial_locations = getattr(trial, 'locations', [])
        
        if not patient_state or not trial_locations:
            return FilterDecision(
                filter_name="geographic_proximity",
                result=FilterResult.UNKNOWN,
                reason="Location data incomplete"
            )
        
        # Check if any trial location is in same state (simple proximity)
        for location in trial_locations:
            if isinstance(location, dict):
                trial_state = location.get('state', '')
            else:
                trial_state = getattr(location, 'state', '')
            
            if trial_state and trial_state.lower() == patient_state.lower():
                return FilterDecision(
                    filter_name="geographic_proximity",
                    result=FilterResult.PASS,
                    reason=f"Trial available in patient's state: {patient_state}"
                )
        
        # For demo, allow trials in any location but flag distance
        return FilterDecision(
            filter_name="geographic_proximity",
            result=FilterResult.PASS,
            reason="Trial location may require travel",
            details={'patient_state': patient_state}
        )
    
    def _check_biomarker_exclusions(
        self,
        patient: Dict[str, Any],
        trial: Any
    ) -> FilterDecision:
        """Check for hard biomarker exclusions."""
        # Get patient biomarkers
        biomarkers_detected = patient.get('biomarkers_detected', '')
        biomarkers_ruled_out = patient.get('biomarkers_ruled_out', '')
        
        # Handle NaN and empty values
        if pd.isna(biomarkers_detected):
            biomarkers_detected = ''
        if pd.isna(biomarkers_ruled_out):
            biomarkers_ruled_out = ''
        
        biomarkers_detected = str(biomarkers_detected).upper()
        biomarkers_ruled_out = str(biomarkers_ruled_out).upper()
        
        # Get trial eligibility text
        eligibility_text = ''
        if hasattr(trial, 'eligibility_criteria'):
            eligibility_text = str(trial.eligibility_criteria).upper()
        elif hasattr(trial, 'eligibility'):
            if hasattr(trial.eligibility, 'inclusion_criteria'):
                eligibility_text = ' '.join(trial.eligibility.inclusion_criteria).upper()
        
        if not eligibility_text:
            return FilterDecision(
                filter_name="biomarker_exclusions",
                result=FilterResult.UNKNOWN,
                reason="No eligibility criteria to check"
            )
        
        # Check for hard exclusions
        exclusion_patterns = {
            'HER2+': ['HER2-POSITIVE', 'HER2+', 'ERBB2'],
            'HER2-': ['HER2-NEGATIVE', 'HER2-'],
            'ER+': ['ER-POSITIVE', 'ER+', 'ESTROGEN RECEPTOR POSITIVE'],
            'ER-': ['ER-NEGATIVE', 'ER-', 'ESTROGEN RECEPTOR NEGATIVE'],
            'EGFR': ['EGFR MUTATION', 'EGFR+'],
            'KRAS': ['KRAS MUTATION', 'KRAS+'],
            'BRCA': ['BRCA MUTATION', 'BRCA1', 'BRCA2'],
        }
        
        # Check if ruled-out biomarkers are required by trial
        if biomarkers_ruled_out:
            for biomarker in biomarkers_ruled_out.split(','):
                biomarker = biomarker.strip()
                if biomarker in exclusion_patterns:
                    for pattern in exclusion_patterns[biomarker]:
                        if pattern in eligibility_text and 'REQUIRED' in eligibility_text:
                            return FilterDecision(
                                filter_name="biomarker_exclusions",
                                result=FilterResult.FAIL,
                                reason=f"Trial requires {biomarker} but patient is negative",
                                details={'required': biomarker, 'patient_status': 'negative'}
                            )
        
        # Check if detected biomarkers are excluded by trial
        if biomarkers_detected:
            for biomarker in biomarkers_detected.split(','):
                biomarker = biomarker.strip()
                # Check for explicit exclusions in eligibility
                if f"NO {biomarker}" in eligibility_text or f"EXCLUDE {biomarker}" in eligibility_text:
                    return FilterDecision(
                        filter_name="biomarker_exclusions",
                        result=FilterResult.FAIL,
                        reason=f"Trial excludes {biomarker} positive patients",
                        details={'excluded': biomarker, 'patient_status': 'positive'}
                    )
        
        return FilterDecision(
            filter_name="biomarker_exclusions",
            result=FilterResult.PASS,
            reason="No hard biomarker exclusions detected"
        )
    
    def get_filter_statistics(
        self,
        filtered_results: List[FilteredTrial]
    ) -> Dict[str, Any]:
        """Generate statistics about filter performance."""
        total_trials = len(filtered_results)
        passed_trials = sum(1 for ft in filtered_results if ft.passed)
        
        # Count failures by filter
        filter_failures = {}
        for ft in filtered_results:
            for decision in ft.decisions:
                if decision.result == FilterResult.FAIL:
                    filter_failures[decision.filter_name] = filter_failures.get(decision.filter_name, 0) + 1
        
        return {
            'total_trials': total_trials,
            'passed_trials': passed_trials,
            'failed_trials': total_trials - passed_trials,
            'pass_rate': passed_trials / total_trials if total_trials > 0 else 0,
            'filter_failures': filter_failures,
            'most_common_failure': max(filter_failures.items(), key=lambda x: x[1])[0] if filter_failures else None
        }


# Example usage
if __name__ == "__main__":
    # Test the filter
    filter_engine = DeterministicFilter()
    
    # Mock patient
    test_patient = {
        'age': 65,
        'gender': 'Female',
        'state': 'California',
        'biomarkers_detected': 'ER+, PR+',
        'biomarkers_ruled_out': 'HER2+'
    }
    
    print("Deterministic Filter Engine initialized")
    print(f"Test patient: {test_patient}")
    print(f"Configuration: {filter_engine.config}")
