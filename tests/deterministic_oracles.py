"""
Deterministic oracles for weak supervision without labels.
"""

import re
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DeterministicOracles:
    """Simple, explainable rule-oracles for facts we can verify."""
    
    def __init__(self):
        self.eligibility_oracle = EligibilityOracle()
        self.biomarker_oracle = BiomarkerOracle()
        self.geography_oracle = GeographyOracle()
    
    def evaluate(self, patient: dict, trial: dict) -> dict:
        """Check deterministic rules."""
        elig = self.eligibility_oracle.check(patient, trial)
        bio = self.biomarker_oracle.check(patient, trial)
        geo = self.geography_oracle.check(patient, trial)
        
        # Determine overall status
        if any(x == 'unknown' for x in [elig, bio, geo]):
            all_pass = 'unknown'
        elif all(x == 'pass' for x in [elig, bio, geo] if x != 'unknown'):
            all_pass = 'pass'
        else:
            all_pass = 'fail'
        
        return {
            'eligibility_ok': elig,
            'biomarker_ok': bio,
            'geography_ok': geo,
            'all_rules_pass': all_pass
        }
    
    def calculate_metrics(self, patient: dict, ranked_trials: list) -> dict:
        """Calculate oracle-based metrics for top-K trials."""
        K = 5
        top_k = ranked_trials[:K] if len(ranked_trials) >= K else ranked_trials
        
        known_passes = 0
        known_fails = 0
        unknown_count = 0
        
        for trial in top_k:
            result = self.evaluate(patient, trial)
            
            if result['all_rules_pass'] == 'unknown':
                unknown_count += 1
            elif result['all_rules_pass'] == 'pass':
                known_passes += 1
            else:
                known_fails += 1
        
        total_known = known_passes + known_fails
        total = len(top_k)
        
        return {
            'rule_known_at_5': total_known / total if total > 0 else 0,
            'rule_satisfy_at_5_given_known': (
                known_passes / total_known if total_known > 0 else None
            ),
            'rule_satisfy_at_5': known_passes / total if total > 0 else 0,
            'violation_at_5': known_fails / total if total > 0 else 0,
            'unknown_at_5': unknown_count / total if total > 0 else 0
        }


class EligibilityOracle:
    """Check age, gender, performance status thresholds."""
    
    def check(self, patient: dict, trial: dict) -> str:
        """Returns 'pass', 'fail', or 'unknown'."""
        checks = []
        
        # Age check
        if 'age' not in patient:
            return 'unknown'
        if 'minimum_age' in trial and patient['age'] < self._parse_age(trial['minimum_age']):
            checks.append(False)
        if 'maximum_age' in trial and patient['age'] > self._parse_age(trial['maximum_age']):
            checks.append(False)
        else:
            checks.append(True)
        
        # Gender check
        if 'gender' in trial:
            trial_gender = self._normalize_gender(trial['gender'])
            if trial_gender in ['all', 'both']:
                checks.append(True)
            elif 'gender' not in patient:
                return 'unknown'
            else:
                patient_gender = self._normalize_gender(patient['gender'])
                checks.append(patient_gender == trial_gender)
        
        # ECOG check
        if 'ecog_status' in patient and patient.get('stage') == 'IV':
            if patient.get('metastatic') == False:
                checks.append(False)  # Stage IV must be metastatic
        
        if not checks:
            return 'unknown'
        
        return 'pass' if all(checks) else 'fail'
    
    def _parse_age(self, age_str: str) -> int:
        """Parse age from string like '18 Years'."""
        if isinstance(age_str, (int, float)):
            return int(age_str)
        match = re.search(r'(\d+)', str(age_str))
        return int(match.group(1)) if match else 0
    
    def _normalize_gender(self, gender: str) -> str:
        """Normalize gender strings."""
        if not gender:
            return 'unknown'
        g = str(gender).lower().strip()
        if g in ['m', 'male', 'man']:
            return 'male'
        elif g in ['f', 'female', 'woman']:
            return 'female'
        elif g in ['all', 'both', 'any']:
            return 'all'
        return 'other'


class BiomarkerOracle:
    """Check biomarker compatibility with synonyms and thresholds."""
    
    def __init__(self):
        self.synonyms = {
            'HER2': ['ERBB2', 'HER2/neu', 'HER-2'],
            'PD-L1': ['CD274', 'B7-H1', 'PDL1'],
            'EGFR': ['ERBB1', 'HER1'],
            'ER': ['ESR1', 'Estrogen Receptor'],
            'PR': ['PGR', 'Progesterone Receptor']
        }
        self._build_canonical_map()
    
    def _build_canonical_map(self):
        """Build reverse mapping for faster lookup."""
        self.canonical_map = {}
        for canonical, syns in self.synonyms.items():
            self.canonical_map[canonical.upper()] = canonical
            for syn in syns:
                self.canonical_map[syn.upper()] = canonical
    
    def check(self, patient: dict, trial: dict) -> str:
        """Returns 'pass', 'fail', or 'unknown'."""
        # For now, return pass if patient has biomarkers
        # In production, would check against trial requirements
        
        patient_markers = patient.get('biomarkers_detected', [])
        if not patient_markers:
            return 'unknown'
        
        # Simple check - if patient has any biomarkers, consider it a pass
        # This is a placeholder for more sophisticated matching
        return 'pass' if patient_markers else 'fail'


class GeographyOracle:
    """Check geographic feasibility."""
    
    def check(self, patient: dict, trial: dict) -> str:
        """Returns 'pass', 'fail', or 'unknown'."""
        patient_state = patient.get('state', '')
        
        if not patient_state:
            return 'unknown'
        
        # For mock data, assume trials are available in major states
        major_states = ['CA', 'NY', 'TX', 'FL', 'MA', 'IL', 'PA']
        
        if patient_state in major_states:
            return 'pass'
        
        # 50% chance for other states
        return 'pass' if hash(patient_state) % 2 == 0 else 'fail'
