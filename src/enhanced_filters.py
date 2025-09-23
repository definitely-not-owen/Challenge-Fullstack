"""
Enhanced deterministic filters with improved normalization and matching.

Implements smarter gender normalization, expanded biomarker patterns,
and distance-based geographic filtering.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


class GenderNormalizer:
    """Robust gender normalization with controlled mapping."""
    
    GENDER_MAPPINGS = {
        # Male variations
        'male': 'male',
        'm': 'male',
        'man': 'male',
        'boy': 'male',
        'masculine': 'male',
        
        # Female variations
        'female': 'female',
        'f': 'female',
        'woman': 'female',
        'girl': 'female',
        'feminine': 'female',
        
        # Other/All variations
        'all': 'all',
        'both': 'all',
        'any': 'all',
        'other': 'other',
        'non-binary': 'other',
        'nonbinary': 'other',
        'nb': 'other',
        'unknown': 'unknown',
        'not specified': 'unknown'
    }
    
    @classmethod
    def normalize(cls, gender_str: str) -> str:
        """
        Normalize gender string to standard values.
        
        Returns: 'male', 'female', 'all', 'other', or 'unknown'
        """
        if not gender_str:
            return 'unknown'
        
        # Clean and lowercase
        cleaned = str(gender_str).strip().lower()
        
        # Direct mapping
        if cleaned in cls.GENDER_MAPPINGS:
            return cls.GENDER_MAPPINGS[cleaned]
        
        # Check for contains
        for key, value in cls.GENDER_MAPPINGS.items():
            if key in cleaned:
                return value
        
        return 'unknown'
    
    @classmethod
    def is_compatible(cls, patient_gender: str, trial_gender: str) -> bool:
        """Check if patient gender is compatible with trial requirements."""
        patient_norm = cls.normalize(patient_gender)
        trial_norm = cls.normalize(trial_gender)
        
        # Trial accepts all genders
        if trial_norm == 'all':
            return True
        
        # Unknown gender - be permissive
        if patient_norm == 'unknown' or trial_norm == 'unknown':
            return True
        
        # Direct match
        return patient_norm == trial_norm


class BiomarkerMatcher:
    """Enhanced biomarker matching with regex patterns and synonyms."""
    
    # Biomarker patterns with synonyms and variations
    BIOMARKER_PATTERNS = {
        'HER2+': {
            'patterns': [r'HER2[\s\-]*(?:positive|pos|\+)', r'ERBB2[\s\-]*(?:positive|pos|\+)'],
            'synonyms': ['HER2-positive', 'HER2+', 'ERBB2+', 'ERBB2-positive', 'HER2/neu positive']
        },
        'HER2-': {
            'patterns': [r'HER2[\s\-]*(?:negative|neg|\-)', r'ERBB2[\s\-]*(?:negative|neg|\-)'],
            'synonyms': ['HER2-negative', 'HER2-', 'ERBB2-', 'ERBB2-negative', 'HER2/neu negative']
        },
        'ER+': {
            'patterns': [r'ER[\s\-]*(?:positive|pos|\+)', r'estrogen[\s\-]*receptor[\s\-]*(?:positive|pos|\+)'],
            'synonyms': ['ER-positive', 'ER+', 'estrogen receptor positive', 'ESR1+']
        },
        'ER-': {
            'patterns': [r'ER[\s\-]*(?:negative|neg|\-)', r'estrogen[\s\-]*receptor[\s\-]*(?:negative|neg|\-)'],
            'synonyms': ['ER-negative', 'ER-', 'estrogen receptor negative', 'ESR1-']
        },
        'PR+': {
            'patterns': [r'PR[\s\-]*(?:positive|pos|\+)', r'progesterone[\s\-]*receptor[\s\-]*(?:positive|pos|\+)'],
            'synonyms': ['PR-positive', 'PR+', 'progesterone receptor positive', 'PGR+']
        },
        'PR-': {
            'patterns': [r'PR[\s\-]*(?:negative|neg|\-)', r'progesterone[\s\-]*receptor[\s\-]*(?:negative|neg|\-)'],
            'synonyms': ['PR-negative', 'PR-', 'progesterone receptor negative', 'PGR-']
        },
        'EGFR': {
            'patterns': [r'EGFR[\s\-]*(?:mutation|mutant|mut)', r'EGFR[\s\-]*(?:positive|pos|\+)'],
            'synonyms': ['EGFR mutation', 'EGFR+', 'EGFR mutant', 'EGFR-positive']
        },
        'ALK': {
            'patterns': [r'ALK[\s\-]*(?:fusion|rearrangement|positive)', r'ALK[\s\-]*\+'],
            'synonyms': ['ALK fusion', 'ALK+', 'ALK rearrangement', 'ALK-positive']
        },
        'KRAS': {
            'patterns': [r'KRAS[\s\-]*(?:mutation|mutant|mut)', r'KRAS[\s\-]*G12[A-Z]'],
            'synonyms': ['KRAS mutation', 'KRAS mutant', 'KRAS G12C', 'KRAS G12D']
        },
        'BRCA': {
            'patterns': [r'BRCA[12]?[\s\-]*(?:mutation|mutant|mut)', r'BRCA[12]?\+'],
            'synonyms': ['BRCA mutation', 'BRCA1 mutation', 'BRCA2 mutation', 'BRCA+']
        },
        'PD-L1': {
            'patterns': [r'PD[\-]?L1[\s\-]*(?:positive|high|≥)', r'PD[\-]?L1[\s\-]*[>≥]\s*\d+%'],
            'synonyms': ['PD-L1 positive', 'PD-L1 high', 'PD-L1 ≥1%', 'PD-L1 >50%']
        },
        'MSI-H': {
            'patterns': [r'MSI[\s\-]*(?:high|H)', r'microsatellite[\s\-]*instability[\s\-]*high'],
            'synonyms': ['MSI-high', 'MSI-H', 'microsatellite instability high', 'dMMR']
        }
    }
    
    @classmethod
    def normalize_biomarker(cls, biomarker_str: str) -> str:
        """Normalize biomarker string to standard form."""
        if not biomarker_str:
            return ''
        
        cleaned = str(biomarker_str).strip()
        
        # Check against known patterns
        for standard_name, config in cls.BIOMARKER_PATTERNS.items():
            # Check synonyms
            for synonym in config['synonyms']:
                if synonym.lower() in cleaned.lower():
                    return standard_name
            
            # Check regex patterns
            for pattern in config['patterns']:
                if re.search(pattern, cleaned, re.IGNORECASE):
                    return standard_name
        
        # Return cleaned version if no match
        return cleaned
    
    @classmethod
    def extract_biomarkers(cls, text: str) -> List[str]:
        """Extract all biomarkers from text."""
        if not text:
            return []
        
        found_biomarkers = set()
        
        for standard_name, config in cls.BIOMARKER_PATTERNS.items():
            # Check patterns
            for pattern in config['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    found_biomarkers.add(standard_name)
                    break
            
            # Check synonyms
            for synonym in config['synonyms']:
                if synonym.lower() in text.lower():
                    found_biomarkers.add(standard_name)
                    break
        
        return list(found_biomarkers)
    
    @classmethod
    def check_biomarker_conflict(
        cls,
        patient_detected: List[str],
        patient_ruled_out: List[str],
        trial_required: List[str],
        trial_excluded: List[str]
    ) -> Tuple[bool, str]:
        """
        Check for biomarker conflicts between patient and trial.
        
        Returns:
            Tuple of (has_conflict, reason)
        """
        # Normalize all biomarkers
        patient_detected_norm = {cls.normalize_biomarker(b) for b in patient_detected if b}
        patient_ruled_out_norm = {cls.normalize_biomarker(b) for b in patient_ruled_out if b}
        trial_required_norm = {cls.normalize_biomarker(b) for b in trial_required if b}
        trial_excluded_norm = {cls.normalize_biomarker(b) for b in trial_excluded if b}
        
        # Check if patient has excluded biomarkers
        excluded_conflicts = patient_detected_norm & trial_excluded_norm
        if excluded_conflicts:
            return True, f"Patient has excluded biomarkers: {', '.join(excluded_conflicts)}"
        
        # Check if patient lacks required biomarkers
        missing_required = trial_required_norm - patient_detected_norm
        if missing_required:
            # Check if any missing ones are explicitly ruled out
            ruled_out_required = missing_required & patient_ruled_out_norm
            if ruled_out_required:
                return True, f"Patient lacks required biomarkers: {', '.join(ruled_out_required)}"
        
        return False, "No biomarker conflicts"


class GeographicCalculator:
    """Calculate geographic distances between locations."""
    
    # Major US city coordinates (lat, lon) for demo
    CITY_COORDS = {
        'new york': (40.7128, -74.0060),
        'los angeles': (34.0522, -118.2437),
        'chicago': (41.8781, -87.6298),
        'houston': (29.7604, -95.3698),
        'phoenix': (33.4484, -112.0740),
        'philadelphia': (39.9526, -75.1652),
        'san antonio': (29.4241, -98.4936),
        'san diego': (32.7157, -117.1611),
        'dallas': (32.7767, -96.7970),
        'san jose': (37.3382, -121.8863),
        'boston': (42.3601, -71.0589),
        'seattle': (47.6062, -122.3321),
        'denver': (39.7392, -104.9903),
        'washington': (38.9072, -77.0369),
        'atlanta': (33.7490, -84.3880),
        'miami': (25.7617, -80.1918),
        'baltimore': (39.2904, -76.6122),
        'cleveland': (41.4993, -81.6944),
        'minneapolis': (44.9778, -93.2650),
        'tampa': (27.9506, -82.4572)
    }
    
    # State centroids (lat, lon) for fallback
    STATE_COORDS = {
        'alabama': (32.806671, -86.791130),
        'alaska': (61.370716, -152.404419),
        'arizona': (33.729759, -111.431221),
        'arkansas': (34.969704, -92.373123),
        'california': (36.116203, -119.681564),
        'colorado': (39.059811, -105.311104),
        'connecticut': (41.597782, -72.755371),
        'delaware': (39.318523, -75.507141),
        'florida': (27.766279, -81.686783),
        'georgia': (33.040619, -83.643074),
        'hawaii': (21.094318, -157.498337),
        'idaho': (44.240459, -114.478828),
        'illinois': (40.349457, -88.986137),
        'indiana': (39.849426, -86.258278),
        'iowa': (42.011539, -93.210526),
        'kansas': (38.526600, -96.726486),
        'kentucky': (37.668140, -84.670067),
        'louisiana': (31.169546, -91.867805),
        'maine': (44.693947, -69.381927),
        'maryland': (39.063946, -76.802101),
        'massachusetts': (42.230171, -71.530106),
        'michigan': (43.326618, -84.536095),
        'minnesota': (45.694454, -93.900192),
        'mississippi': (32.741646, -89.678696),
        'missouri': (38.456085, -92.288368),
        'montana': (46.921925, -110.454353),
        'nebraska': (41.125370, -98.268082),
        'nevada': (38.313515, -117.055374),
        'new hampshire': (43.452492, -71.563896),
        'new jersey': (40.298904, -74.521011),
        'new mexico': (34.840515, -106.248482),
        'new york': (42.165726, -74.948051),
        'north carolina': (35.630066, -79.806419),
        'north dakota': (47.528912, -99.784012),
        'ohio': (40.388783, -82.764915),
        'oklahoma': (35.565342, -96.928917),
        'oregon': (44.572021, -122.070938),
        'pennsylvania': (40.590752, -77.209755),
        'rhode island': (41.680893, -71.511780),
        'south carolina': (33.856892, -80.945007),
        'south dakota': (44.299782, -99.438828),
        'tennessee': (35.747845, -86.692345),
        'texas': (31.054487, -97.563461),
        'utah': (40.150032, -111.862434),
        'vermont': (44.045876, -72.710686),
        'virginia': (37.769337, -78.169968),
        'washington': (47.400902, -121.490494),
        'west virginia': (38.491226, -80.954570),
        'wisconsin': (44.268543, -89.616508),
        'wyoming': (42.755966, -107.302490)
    }
    
    @classmethod
    def haversine_distance(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula.
        
        Returns:
            Distance in miles
        """
        # Radius of Earth in miles
        R = 3959.0
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    @classmethod
    def get_location_coords(cls, city: str = None, state: str = None) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location."""
        # Try city first
        if city:
            city_lower = city.lower().strip()
            if city_lower in cls.CITY_COORDS:
                return cls.CITY_COORDS[city_lower]
        
        # Fall back to state
        if state:
            state_lower = state.lower().strip()
            if state_lower in cls.STATE_COORDS:
                return cls.STATE_COORDS[state_lower]
        
        return None
    
    @classmethod
    def calculate_distance(
        cls,
        patient_city: str,
        patient_state: str,
        trial_city: str,
        trial_state: str
    ) -> Optional[float]:
        """
        Calculate distance between patient and trial locations.
        
        Returns:
            Distance in miles, or None if coordinates not found
        """
        patient_coords = cls.get_location_coords(patient_city, patient_state)
        trial_coords = cls.get_location_coords(trial_city, trial_state)
        
        if not patient_coords or not trial_coords:
            return None
        
        return cls.haversine_distance(
            patient_coords[0], patient_coords[1],
            trial_coords[0], trial_coords[1]
        )


# Example usage
if __name__ == "__main__":
    # Test gender normalization
    print("Gender Normalization Tests:")
    test_genders = ['Male', 'F', 'female', 'ALL', 'both', 'non-binary']
    for gender in test_genders:
        normalized = GenderNormalizer.normalize(gender)
        print(f"  '{gender}' -> '{normalized}'")
    
    print("\nGender Compatibility:")
    print(f"  Male patient, All trial: {GenderNormalizer.is_compatible('Male', 'All')}")
    print(f"  Female patient, Male trial: {GenderNormalizer.is_compatible('Female', 'Male')}")
    
    # Test biomarker matching
    print("\nBiomarker Extraction:")
    text = "Eligible patients must be HER2-positive and ER-positive, with PD-L1 ≥50%"
    biomarkers = BiomarkerMatcher.extract_biomarkers(text)
    print(f"  Found: {biomarkers}")
    
    # Test geographic distance
    print("\nGeographic Distance:")
    distance = GeographicCalculator.calculate_distance(
        "Boston", "Massachusetts",
        "New York", "New York"
    )
    print(f"  Boston to New York: {distance:.1f} miles")
