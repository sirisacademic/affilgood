import os
from whoosh import index
from whoosh.qparser import MultifieldParser, QueryParser, OrGroup, AndGroup
from whoosh.query import Term, Or, And, FuzzyTerm
from whoosh.analysis import StemmingAnalyzer
from .base_linker import BaseLinker
from .utils.text_utils import (
    get_variants_list, get_variants_country,
    get_stopwords, get_legal_entities, load_abbreviations
)
from .utils.translation_mappings import translate_institution_name
from unidecode import unidecode
from .constants import *
import math
from .utils.text_utils import *

ADJUST_FACTOR_THRESHOLD_NON_LATIN = 0.75
DEFAULT_CALIBRATION_POINTS = {
    'perfect_threshold': 200,
    'near_perfect_threshold': 100,
    'partial_threshold': 50,
    'low_threshold': 25,
    'non_match_threshold': 10
}

class WhooshLinker(BaseLinker):
    """Whoosh-based entity linker for retrieving organization candidates."""
    
    def __init__(
                self,
                data_manager=None,
                index_dir=None,
                rebuild_index=False,
                threshold_score=0.30,
                return_num_candidates=NUM_CANDIDATES_TO_RETURN,
                max_hits=10,
                run_score_calibration=False,
                normalization_max_score=1000.0,
                normalization_min_threshold=100.0,
                data_source="ror",
                org_types=None,     # WikiData org types
                countries=None,     # WikiData countries
                debug=False,
                use_cache=True,
                use_wikidata_labels_with_ror=False # Whether to add WikiData labels from previously downloaded file when generating the ROR indices
            ):
        """
        Initialize with data manager and search parameters.
        
        Args:
            data_manager: DataManager instance for handling data
            index_dir: Optional specific directory for the Whoosh index
            rebuild_index: Whether to force rebuilding the index
            threshold_score: Minimum score for valid candidates
            return_num_candidates: Number of candidates to return
            max_hits: Maximum hits to retrieve from Whoosh
            run_score_calibration: Whether to run score calibration
            normalization_max_score: Maximum score for normalization
            normalization_min_threshold: Minimum threshold for normalization
            data_source: Data source to use ("ror" or "wikidata")
            debug: Whether to show debug output
            use_cache: Whether to use caching
            use_wikidata_labels_with_ror: Whether to enrich ROR organizations with WikiData labels
        """
        super().__init__(use_cache=use_cache, data_source=data_source,
                        wikidata_org_types=org_types, wikidata_countries=countries)
        self.data_manager = data_manager
        self.threshold_score = threshold_score
        self.run_score_calibration = run_score_calibration
        self.normalization_max_score = 1500
        self.normalization_min_threshold = 20
        self.max_hits = max_hits
        self.debug = debug
        self.index_dir = index_dir
        self.rebuild_index = rebuild_index
        self.return_num_candidates = return_num_candidates
        self.data_source = data_source  # Store data source
        self.org_types = org_types
        self.countries = countries
        self.use_wikidata_labels_with_ror = use_wikidata_labels_with_ror
        # Initialization is deferred until we have a data_manager
        self.is_initialized = False
        # If data_manager is provided, initialize now
        if self.data_manager is not None:
            self.initialize()
       
    def initialize(self):
        """Initialize the Whoosh linker components with improved calibration."""
        print("WhooshLinker.initialize")
        
        if self.is_initialized:
            return
            
        if self.debug:
            print(f"WhooshLinker: setting self.index_dir={self.index_dir}")
            
        # Prepare the Whoosh index using source-agnostic approach
        if self.index_dir is None:
            # Get or create the index for this data source
            self.index_dir = self.data_manager.get_or_create_index(
                source=self.data_source,
                indices_type='whoosh',
                force_rebuild=self.rebuild_index,
                org_types=self.org_types,
                countries=self.countries,
                use_wikidata_labels_with_ror=self.use_wikidata_labels_with_ror
            )
        else:
            # If index_dir is provided, ensure it exists and create if needed
            if not os.path.exists(self.index_dir) or (self.rebuild_index and len(os.listdir(self.index_dir)) == 0):
                if self.debug:
                    print(f"Creating Whoosh index in {self.index_dir}")
                # Get latest data for this source
                if self.data_source == 'ror':
                    ror_file = self.data_manager.get_latest_ror()
                    if ror_file:
                        with open(ror_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.data_manager.create_whoosh_index(data, self.index_dir, source=self.data_source)
                elif self.data_source == 'wikidata':
                    data = self.data_manager.get_wikidata_organizations()
                    if data is not None:
                        self.data_manager.create_whoosh_index(data, self.index_dir, source=self.data_source)
                else:
                    raise ValueError(f"Unknown data source: {self.data_source}")

        if self.debug:
            print(f"WhooshLinker: setting self.index_dir={self.index_dir}")


        # Set up the Whoosh components
        self.analyzer = StemmingAnalyzer()
        self.stopwords = get_stopwords('u')
        self.legal_entities = get_legal_entities()
        self.abbreviations = load_abbreviations()
        
        # Set up the Whoosh index
        if self.debug:
            print(f"WhooshLinker: opening self.index_dir={self.index_dir}")
            
        self.ix = index.open_dir(self.index_dir)
        
        # Update field names to use source-agnostic versions
        self.name_parser = MultifieldParser(["name", "name_normalized"], schema=self.ix.schema, group=AndGroup)
        self.name_parser_flex = MultifieldParser(["name", "name_normalized"], schema=self.ix.schema, group=OrGroup)
        self.alias_parser = QueryParser("aliases_text", schema=self.ix.schema, group=AndGroup)
        self.alias_parser_flex = QueryParser("aliases_text", schema=self.ix.schema, group=OrGroup)
        
        # Consider types of matches for initial re-ranking.
        self.strong_keywords_set = {"name", "alias"}
        self.location_keywords_set = {"city", "region", "country"}
        
        # Run score calibration if enabled
        if self.run_score_calibration:
            stats = self._calibrate_scores()
            
            # Set calibration points based on the statistics
            self.calibration_points = self._compute_calibration_points(stats)
            
            # Set legacy normalization parameters for backward compatibility
            if stats["perfect_matches"]["count"] > 0:
                self.normalization_max_score = stats["perfect_matches"]["max"]
                if stats["near_perfect_matches"]["count"] > 0:
                    self.normalization_min_threshold = stats["near_perfect_matches"]["min"] * 0.8
                    
                if self.debug:
                    print(f"Calibrated normalization: max={self.normalization_max_score:.2f}, threshold={self.normalization_min_threshold:.2f}")
                    print(f"Calibration points: {self.calibration_points}")
        else:
            # Set default calibration points using existing normalization parameters
            self.calibration_points = DEFAULT_CALIBRATION_POINTS
        
        self.is_initialized = True
    
    def _adjust_scores_based_on_match_type(self, top_entities):
        """Optimizes score adjustments by checking name and location keyword matches."""
        for candidate, info in top_entities.items():
            # Convert descriptions to a set for faster lookup
            description_set = {desc.lower() for desc in info["matches"]}

            # Check for strong name match
            if any(kw in description_set for kw in self.strong_keywords_set):
                continue  # No need to penalize if strong name match exists

            # Check for location match
            if any(kw in description_set for kw in self.location_keywords_set):
                info["score"] *= 0.5  # Apply penalty for location-only match

        return top_entities

    def _calculate_category_stats(self, scores):
        """Calculate statistics for a category of scores."""
        if not scores:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "count": 0
            }
        
        import numpy as np
        
        return {
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "mean": sum(scores) / len(scores) if scores else 0,
            "median": sorted(scores)[len(scores)//2] if scores else 0,
            "count": len(scores)
        }

    def _normalize_score(self, raw_score, org_name=None):
        """
        Normalize a raw Whoosh score to a 0-1 range using an adaptive piecewise function.
        
        This enhanced function:
        1. Uses calibration points from _calibrate_scores
        2. Adjusts thresholds based on text characteristics (Latin vs. non-Latin script)
        3. Uses smoother transitions between score ranges
        4. Produces scores more comparable to DenseLinker's distribution
        
        Args:
            raw_score: The raw Whoosh score to normalize
            org_name: Optional organization name to check for script type
            
        Returns:
            float: Normalized score in the 0-1 range
        """
        # Ensure we have calibration points, use defaults if not
        if not hasattr(self, 'calibration_points'):
            # Default calibration points if not set from _calibrate_scores
            self.calibration_points = {
                "perfect_threshold": self.normalization_max_score * 0.85,
                "near_perfect_threshold": self.normalization_max_score * 0.55, 
                "partial_threshold": self.normalization_max_score * 0.25,
                "low_threshold": self.normalization_min_threshold,
                "non_match_threshold": self.normalization_min_threshold * 0.5
            }
        
        # Create a local copy of thresholds we can adjust based on script
        thresholds = {
            "perfect": self.calibration_points["perfect_threshold"],
            "near_perfect": self.calibration_points["near_perfect_threshold"],
            "partial": self.calibration_points["partial_threshold"],
            "low": self.calibration_points["low_threshold"],
            "non_match": self.calibration_points["non_match_threshold"]
        }
        
        # Adjust thresholds for non-Latin script organizations
        # They typically get lower raw scores but should still receive comparable normalized scores
        is_non_latin = False
        if org_name and contains_non_latin(org_name):
            is_non_latin = True
            adjustment_factor = ADJUST_FACTOR_THRESHOLD_NON_LATIN  # Lower thresholds for non-Latin scripts
            for key in thresholds:
                thresholds[key] = thresholds[key] * adjustment_factor
        
        # Target score ranges for different match categories
        scores = {
            "perfect": {
                "min": 0.85,  # Minimum score for perfect matches
                "max": 0.99   # Maximum score (never quite reaches 1.0)
            },
            "near_perfect": {
                "min": 0.70,
                "max": 0.85
            },
            "partial": {
                "min": 0.45,
                "max": 0.70
            },
            "low": {
                "min": 0.20,
                "max": 0.45
            },
            "non_match": {
                "min": 0.05,
                "max": 0.20
            },
            "very_low": {
                "min": 0.0,
                "max": 0.05
            }
        }
        
        # Apply sigmoid-like smoothing to scores within each range
        def smooth_transition(value, min_val, max_val):
            """Apply a sigmoid-like function to smooth transitions between ranges."""
            # Map value to range [0, 1]
            normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
            
            # Apply mild sigmoid-like smoothing
            # This gives a more natural transition by reducing the linearity
            if normalized <= 0:
                return 0
            elif normalized >= 1:
                return 1
            else:
                # This makes transitions slightly more curved
                return 0.5 + 0.5 * (2 * normalized - 1) / (1 + abs(2 * normalized - 1))
        
        # Determine which range the score falls into and map accordingly
        if raw_score >= thresholds["perfect"]:
            # Perfect match range
            position = smooth_transition(raw_score, thresholds["perfect"], thresholds["perfect"] * 1.5)
            return scores["perfect"]["min"] + position * (scores["perfect"]["max"] - scores["perfect"]["min"])
            
        elif raw_score >= thresholds["near_perfect"]:
            # Near-perfect match range
            position = smooth_transition(raw_score, thresholds["near_perfect"], thresholds["perfect"])
            return scores["near_perfect"]["min"] + position * (scores["near_perfect"]["max"] - scores["near_perfect"]["min"])
            
        elif raw_score >= thresholds["partial"]:
            # Partial match range
            position = smooth_transition(raw_score, thresholds["partial"], thresholds["near_perfect"])
            return scores["partial"]["min"] + position * (scores["partial"]["max"] - scores["partial"]["min"])
            
        elif raw_score >= thresholds["low"]:
            # Low match range
            position = smooth_transition(raw_score, thresholds["low"], thresholds["partial"])
            return scores["low"]["min"] + position * (scores["low"]["max"] - scores["low"]["min"])
            
        elif raw_score >= thresholds["non_match"]:
            # Non-match but above minimum threshold
            position = smooth_transition(raw_score, thresholds["non_match"], thresholds["low"])
            return scores["non_match"]["min"] + position * (scores["non_match"]["max"] - scores["non_match"]["min"])
            
        else:
            # Very low scores
            position = smooth_transition(raw_score, 0, thresholds["non_match"])
            return scores["very_low"]["min"] + position * (scores["very_low"]["max"] - scores["very_low"]["min"])

    def _clean_organization_name(self, name):
        """Normalize organization name by removing legal entities, stopwords, and applying transliteration."""
        if not name:
            return ""
        name = name.replace("'", " ")
        name = name.replace('"', " ")
        words = name.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords and word not in self.legal_entities]
        return " ".join(filtered_words)

    def _compute_calibration_points(self, stats):
        """
        Compute calibration points from statistics gathered during calibration.
        
        Args:
            stats: Dictionary of statistics from _calibrate_scores
            
        Returns:
            Dictionary of calibration points for use in _normalize_score
        """
        # Default values in case statistics are missing
        perfect_threshold = self.normalization_max_score * 0.85
        near_perfect_threshold = self.normalization_max_score * 0.55
        partial_threshold = self.normalization_max_score * 0.25
        low_threshold = self.normalization_min_threshold
        non_match_threshold = self.normalization_min_threshold * 0.5
        
        # Use actual statistics when available
        if stats["perfect_matches"]["count"] > 0:
            # Perfect threshold: 10% below the mean perfect score
            perfect_threshold = stats["perfect_matches"]["mean"] * 0.9
        
        if stats["near_perfect_matches"]["count"] > 0:
            # Near perfect threshold: median of near perfect scores
            near_perfect_threshold = stats["near_perfect_matches"]["median"]
        
        if stats["partial_matches"]["count"] > 0:
            # Partial threshold: median of partial scores
            partial_threshold = stats["partial_matches"]["median"]
        
        if stats["low_matches"]["count"] > 0:
            # Low threshold: median of low match scores
            low_threshold = stats["low_matches"]["median"]
        
        if stats["non_matches"]["count"] > 0:
            # Non-match threshold: max of non-match scores
            non_match_threshold = stats["non_matches"]["max"]
        
        # Ensure thresholds are properly ordered
        perfect_threshold = max(perfect_threshold, near_perfect_threshold * 1.2)
        near_perfect_threshold = max(near_perfect_threshold, partial_threshold * 1.2)
        partial_threshold = max(partial_threshold, low_threshold * 1.2)
        low_threshold = max(low_threshold, non_match_threshold * 1.2)
        
        return {
            "perfect_threshold": perfect_threshold,
            "near_perfect_threshold": near_perfect_threshold,
            "partial_threshold": partial_threshold,
            "low_threshold": low_threshold,
            "non_match_threshold": non_match_threshold
        }


    def _calibrate_scores(self, num_samples=200):
        """
        Calibrate score normalization with diverse sample types and better data integration.
        
        This enhanced calibration creates several categories of samples:
        1. Perfect matches (exact organization name and location)
        2. Near-perfect matches (slight variations to organization name)
        3. Partial matches (significant modifications to create medium scores)
        4. Low-scoring matches (heavily modified to create low scores)
        5. Non-matches (organization names unlikely to match anything in the index)
        
        Args:
            num_samples: Total number of samples to use for calibration
            
        Returns:
            Dictionary of score statistics
        """
        import random
        import string
        
        if self.debug:
            print(f"Calibrating Whoosh scores with {num_samples} diverse samples...")
        
        # Allocate samples across categories
        perfect_count = num_samples // 5
        near_perfect_count = num_samples // 5
        partial_count = num_samples // 5
        low_count = num_samples // 5
        non_match_count = num_samples - (perfect_count + near_perfect_count + partial_count + low_count)
        
        # Track used entity IDs to avoid duplicates
        used_ids = set()
        
        # Sample organizations directly from the index
        with self.ix.searcher() as searcher:
            # Get all documents - we'll try to get a varied sample
            all_docs = list(searcher.documents())
            
            # Attempt to get a diverse sample by:
            # 1. Getting organizations from different countries
            # 2. Getting organizations with different name lengths
            # 3. Including organizations with non-Latin characters
            
            country_orgs = {}
            latin_orgs = []
            non_latin_orgs = []
            
            for doc in all_docs:
                org_id = doc.get("id", "")
                if not org_id or org_id in used_ids:
                    continue
                    
                # Categorize by country
                country = doc.get("country_name", "unknown").lower()
                if country not in country_orgs:
                    country_orgs[country] = []
                
                # Categorize by script
                org_name = doc.get("name", "")
                if contains_non_latin(org_name):
                    non_latin_orgs.append(doc)
                else:
                    latin_orgs.append(doc)
                    
                country_orgs[country].append(doc)
                
            # Build our sample set with diversity in mind
            perfect_docs = []
            
            # First add some non-Latin organizations (if available)
            non_latin_sample_size = min(perfect_count // 4, len(non_latin_orgs))
            if non_latin_sample_size > 0:
                non_latin_sample = random.sample(non_latin_orgs, non_latin_sample_size)
                for doc in non_latin_sample:
                    org_id = doc.get("id", "")
                    if org_id and org_id not in used_ids:
                        used_ids.add(org_id)
                        perfect_docs.append(doc)
            
            # Then add organizations from diverse countries
            countries = list(country_orgs.keys())
            random.shuffle(countries)
            remaining_slots = perfect_count + near_perfect_count + partial_count + low_count - len(perfect_docs)
            countries_to_use = min(len(countries), max(5, remaining_slots // 2))
            
            for country in countries[:countries_to_use]:
                country_sample_size = min(remaining_slots // countries_to_use + 1, len(country_orgs[country]))
                country_sample = random.sample(country_orgs[country], country_sample_size)
                
                for doc in country_sample:
                    org_id = doc.get("id", "")
                    if org_id and org_id not in used_ids:
                        used_ids.add(org_id)
                        perfect_docs.append(doc)
                        
                        if len(perfect_docs) >= perfect_count + near_perfect_count + partial_count + low_count:
                            break
                
                if len(perfect_docs) >= perfect_count + near_perfect_count + partial_count + low_count:
                    break
            
            # If we still need more, add random organizations
            if len(perfect_docs) < perfect_count + near_perfect_count + partial_count + low_count:
                random.shuffle(latin_orgs)
                for doc in latin_orgs:
                    org_id = doc.get("id", "")
                    if org_id and org_id not in used_ids:
                        used_ids.add(org_id)
                        perfect_docs.append(doc)
                        
                        if len(perfect_docs) >= perfect_count + near_perfect_count + partial_count + low_count:
                            break
            
            # Shuffle final set to ensure random distribution across categories
            random.shuffle(perfect_docs)
            
            # Initialize score containers
            perfect_scores = []
            near_perfect_scores = []
            partial_scores = []
            low_scores = []
            non_match_scores = []
            
            # Create test organizations for each category
            for i, doc in enumerate(perfect_docs):
                # Extract organization details
                org_name = doc["name"]
                org_city = doc.get("city", "")
                org_country = doc.get("country_name", "")
                org_region = doc.get("region", "")
                org_location = f"{org_city}, {org_country}" if org_city and org_country else org_country
                
                # Check if this is a non-Latin organization (for special handling)
                is_non_latin = contains_non_latin(org_name)
                
                # Create category-specific test organization
                if i < perfect_count:
                    # 1. Perfect match - exact name and location
                    test_org = {
                        "main": org_name,
                        "suborg": "",
                        "city": org_city,
                        "country": org_country,
                        "region": org_region,
                        "location": org_location,
                        "span_text": f"{org_name}, {org_location}"
                    }
                    
                    # Get score for perfect match
                    perfect_candidates = self.get_candidate_matches(test_org)
                    if perfect_candidates:
                        perfect_scores.append(getattr(perfect_candidates[0], 'score', 0))
                        
                elif i < perfect_count + near_perfect_count:
                    # 2. Near-perfect match - slight modification
                    words = org_name.split()
                    
                    # Modify non-Latin names differently
                    if is_non_latin:
                        # For non-Latin, just add a generic term (preserves most of the original)
                        generic_terms = ["University", "Institute", "Center", "Academy"]
                        near_perfect_name = f"{org_name} {random.choice(generic_terms)}"
                    elif len(words) >= 3:
                        # Remove one word or swap two words
                        if random.random() > 0.5:
                            near_perfect_name = " ".join([w for i, w in enumerate(words) if i != random.randint(0, len(words)-1)])
                        else:
                            idx1, idx2 = random.sample(range(len(words)), 2)
                            words[idx1], words[idx2] = words[idx2], words[idx1]
                            near_perfect_name = " ".join(words)
                    elif len(words) == 2:
                        # Remove one word or add a common word
                        if random.random() > 0.5:
                            near_perfect_name = words[random.randint(0, 1)]
                        else:
                            common_words = ["Research", "Institute", "Center", "Academy", "College", "School"]
                            near_perfect_name = f"{org_name} {random.choice(common_words)}"
                    else:
                        # Add a common word
                        common_words = ["Research", "Institute", "Center", "Academy", "College", "School"]
                        near_perfect_name = f"{org_name} {random.choice(common_words)}"
                    
                    # For any non-Latin language that has translated names, we could also check aliases
                    if is_non_latin and hasattr(self.data_manager, 'translate_institution_name'):
                        translations = getattr(self.data_manager, 'translate_institution_name', lambda x: [])(org_name)
                        if translations and random.random() > 0.5:
                            # Use a translation instead
                            near_perfect_name = random.choice(translations)
                    
                    test_org = {
                        "main": near_perfect_name,
                        "suborg": "",
                        "city": org_city,
                        "country": org_country,
                        "region": org_region,
                        "location": org_location,
                        "span_text": f"{near_perfect_name}, {org_location}"
                    }
                    
                    # Get score for near perfect match
                    near_perfect_candidates = self.get_candidate_matches(test_org)
                    if near_perfect_candidates:
                        near_perfect_scores.append(getattr(near_perfect_candidates[0], 'score', 0))
                        
                elif i < perfect_count + near_perfect_count + partial_count:
                    # 3. Partial match - significant modification
                    words = org_name.split()
                    
                    if is_non_latin:
                        # For non-Latin names, try to use a partial match with location preserved
                        partial_name = org_name.split()[0] if len(org_name.split()) > 1 else org_name
                        
                        # If possible, get the first few characters (careful with UTF-8)
                        if len(partial_name) > 3:
                            try:
                                partial_name = partial_name[:3]  # This might not work well with some scripts
                            except:
                                pass  # Keep the original if splitting fails
                    elif len(words) >= 3:
                        # Keep only half the words
                        keep_count = max(1, len(words) // 2)
                        partial_name = " ".join(random.sample(words, keep_count))
                        
                        # Add some generic terms
                        generic_terms = ["University", "Institute", "College", "Department", "Center", "International"]
                        partial_name = f"{partial_name} {random.choice(generic_terms)}"
                    else:
                        # Use only the first word or abbreviation plus generic terms
                        if len(words[0]) > 3:
                            first_part = words[0][:3]  # Use first 3 characters
                        else:
                            first_part = words[0]
                            
                        generic_terms = ["University", "Institute", "College", "Department", "Center", "International"]
                        partial_name = f"{first_part} {random.choice(generic_terms)}"
                    
                    # Use only country, not city for partial matches
                    test_org = {
                        "main": partial_name,
                        "suborg": "",
                        "city": "",
                        "country": org_country,
                        "region": "",
                        "location": org_country,
                        "span_text": f"{partial_name}, {org_country}"
                    }
                    
                    # Get score for partial match
                    partial_candidates = self.get_candidate_matches(test_org)
                    if partial_candidates:
                        partial_scores.append(getattr(partial_candidates[0], 'score', 0))
                        
                else:
                    # 4. Low-scoring match - heavy modification
                    words = org_name.split()
                    
                    if is_non_latin:
                        # For non-Latin, create a simpler test
                        # Just use a generic academic term that might match many things
                        low_score_words = ["Academic", "Global", "Sciences", "Technology", "Regional", "National"]
                        low_score_name = f"{random.choice(low_score_words)} {random.choice(low_score_words)}"
                    elif words:
                        # Keep just one word (or part of it) from original
                        word = random.choice(words)
                        if len(word) > 3:
                            word = word[:3]  # Use just first few chars
                            
                        # Add completely different words
                        low_score_words = ["Academic", "Global", "Sciences", "Technology", "Regional", "National"]
                        random.shuffle(low_score_words)
                        low_score_name = f"{word} {' '.join(low_score_words[:2])}"
                    else:
                        # Completely synthetic but plausible name
                        low_score_words = ["Academic", "Global", "Sciences", "Technology", "Regional", "National"]
                        random.shuffle(low_score_words)
                        low_score_name = f"{low_score_words[0]} {low_score_words[1]}"
                    
                    # Use different country
                    countries = ["France", "Germany", "Brazil", "Japan", "Australia", "Canada", "South Africa"]
                    different_country = random.choice([c for c in countries if c != org_country])
                    
                    test_org = {
                        "main": low_score_name,
                        "suborg": "",
                        "city": "",
                        "country": different_country,
                        "region": "",
                        "location": different_country,
                        "span_text": f"{low_score_name}, {different_country}"
                    }
                    
                    # Get score for low match
                    low_candidates = self.get_candidate_matches(test_org)
                    if low_candidates:
                        low_scores.append(getattr(low_candidates[0], 'score', 0))
            
            # 5. Create non-matching organizations with random strings
            for _ in range(non_match_count):
                # Generate random organization name
                random_word_length = random.randint(5, 12)
                random_word = ''.join(random.choice(string.ascii_letters) for _ in range(random_word_length))
                non_match_name = f"{random_word} Organization"
                
                # Random location
                countries = ["Xanadu", "Atlantis", "Shangri-La", "El Dorado", "Camelot"]
                non_match_country = random.choice(countries)
                
                test_org = {
                    "main": non_match_name,
                    "suborg": "",
                    "city": "",
                    "country": non_match_country,
                    "region": "",
                    "location": non_match_country,
                    "span_text": f"{non_match_name}, {non_match_country}"
                }
                
                # Get score for non-match
                non_match_candidates = self.get_candidate_matches(test_org)
                if non_match_candidates:
                    non_match_scores.append(getattr(non_match_candidates[0], 'score', 0))
            
            # Calculate statistics for all categories
            stats = {
                "perfect_matches": self._calculate_category_stats(perfect_scores),
                "near_perfect_matches": self._calculate_category_stats(near_perfect_scores),
                "partial_matches": self._calculate_category_stats(partial_scores),
                "low_matches": self._calculate_category_stats(low_scores),
                "non_matches": self._calculate_category_stats(non_match_scores)
            }
            
            if self.debug:
                print(f"Calibration results:")
                print(f"  Perfect matches ({stats['perfect_matches']['count']}): " 
                      f"min={stats['perfect_matches']['min']:.2f}, "
                      f"max={stats['perfect_matches']['max']:.2f}, "
                      f"mean={stats['perfect_matches']['mean']:.2f}")
                print(f"  Near-perfect matches ({stats['near_perfect_matches']['count']}): "
                      f"min={stats['near_perfect_matches']['min']:.2f}, "
                      f"max={stats['near_perfect_matches']['max']:.2f}, "
                      f"mean={stats['near_perfect_matches']['mean']:.2f}")
                print(f"  Partial matches ({stats['partial_matches']['count']}): "
                      f"min={stats['partial_matches']['min']:.2f}, "
                      f"max={stats['partial_matches']['max']:.2f}, "
                      f"mean={stats['partial_matches']['mean']:.2f}")
                print(f"  Low matches ({stats['low_matches']['count']}): "
                      f"min={stats['low_matches']['min']:.2f}, "
                      f"max={stats['low_matches']['max']:.2f}, "
                      f"mean={stats['low_matches']['mean']:.2f}")
                print(f"  Non-matches ({stats['non_matches']['count']}): "
                      f"min={stats['non_matches']['min']:.2f}, "
                      f"max={stats['non_matches']['max']:.2f}, "
                      f"mean={stats['non_matches']['mean']:.2f}")
                
            return stats
    
    
    def _format_candidate(self, candidate):
        """
        Format a candidate for the entity field (previously ror field).
        
        Args:
            candidate: Candidate to format
            
        Returns:
            Formatted entity entry string
        """
        # Skip invalid candidates
        if not candidate or not candidate.get("id"):
            return ""
            
        # Get basic information
        entity_id = candidate['id']
        name = candidate.get('name', '')
        
        # Format URL based on data source
        id_url = self.data_manager.format_id_url(entity_id, self.data_source)
        
        # Get score (prefer direct_score from reranker if available)
        score = candidate.get('direct_score', candidate.get('enc_score', 0.0))
        
        # Format based on whether scores should be included
        if self.return_scores:
            return f"{name} {{{id_url}}}:{score:.2f}"
        else:
            return f"{name} {{{id_url}}}"
    
    """
    def _refine_names_alias_matches(self, results, search_text):
        # Refine search results by checking for overlaps in both names and aliases.
        search_tokens = set(search_text.lower().split())
        search_text_lower = search_text.lower()
        
        refined_results = []
        
        for hit in results:
            name = hit.get("name", "")
            name_lower = name.lower()
            aliases_list = hit.get("aliases_list", [])
            
            # Add the name itself to the potential matches list
            all_potential_matches = [name] + aliases_list
            
            best_match_score = 0
            best_match_text = None
            best_match_type = None
            
            # Check each potential match (name + aliases)
            for potential_match in all_potential_matches:
                if not potential_match:
                    continue
                    
                potential_match_lower = potential_match.lower()
                
                # 1. Exact match (highest priority)
                if potential_match_lower == search_text_lower:
                    match_score = 1.0
                    best_match_score = match_score
                    best_match_text = potential_match
                    best_match_type = "name" if potential_match == name else "alias"
                    break
                    
                # 2. One is substring of the other
                elif potential_match_lower in search_text_lower or search_text_lower in potential_match_lower:
                    # Calculate how much of the shorter string is contained in the longer one
                    shorter = min(len(potential_match_lower), len(search_text_lower))
                    longer = max(len(potential_match_lower), len(search_text_lower))
                    containment_score = shorter / longer * 0.9  # 90% of exact match
                    
                    if containment_score > best_match_score:
                        best_match_score = containment_score
                        best_match_text = potential_match
                        best_match_type = "name" if potential_match == name else "alias"
                
                # 3. Token overlap
                else:
                    match_tokens = set(potential_match_lower.split())
                    common_tokens = search_tokens.intersection(match_tokens)
                    
                    if common_tokens:
                        # Calculate Jaccard similarity: intersection / union
                        jaccard = len(common_tokens) / len(search_tokens.union(match_tokens))
                        overlap_score = jaccard * 0.8  # 80% of exact match
                        
                        if overlap_score > best_match_score:
                            best_match_score = overlap_score
                            best_match_text = potential_match
                            best_match_type = "name" if potential_match == name else "alias"
            
            # Apply score boost based on match quality
            if best_match_score > 0:
                # Scale the boost based on match quality (1.0 to 2.0)
                # Give a slight additional boost to name matches vs alias matches
                name_boost = 0.1 if best_match_type == "name" else 0
                boost_factor = 1.0 + best_match_score + name_boost
                hit.score *= boost_factor
            
                if self.debug:
                    match_type = f"EXACT ({best_match_score:.2f})" if best_match_score == 1.0 else f"PARTIAL ({best_match_score:.2f})"
                    print(f"=> Name candidate: {name} - best match for {search_text}: {match_type} {best_match_type}: '{best_match_text}' - Score boosted by {boost_factor:.2f}x. Updated hit score: {hit.score}")
            else:
                if self.debug:
                    print(f"=> Name candidate: {name} - NO MATCH for {search_text}: {name} or {aliases_list}")
            
            refined_results.append(hit)
        
        # Sort by score descending
        return sorted(refined_results, key=lambda x: x.score, reverse=True)
    """
    
    def _refine_names_alias_matches(self, results, search_text):
        """Refine search results by checking for overlaps in both names and aliases, with enhanced exact match handling."""
        search_tokens = set(search_text.lower().split())
        search_text_lower = search_text.lower()
        
        refined_results = []
        
        for hit in results:
            name = hit.get("name", "")
            name_lower = name.lower()
            aliases_list = hit.get("aliases_list", [])
            
            # Add the name itself to the potential matches list
            all_potential_matches = [name] + aliases_list
            
            best_match_score = 0
            best_match_text = None
            best_match_type = None
            is_exact_match = False  # Flag for exact matches
            
            # Check each potential match (name + aliases)
            for potential_match in all_potential_matches:
                if not potential_match:
                    continue
                    
                potential_match_lower = potential_match.lower()
                
                # 1. Exact match (highest priority)
                if potential_match_lower == search_text_lower:
                    match_score = 1.0
                    best_match_score = match_score
                    best_match_text = potential_match
                    best_match_type = "name" if potential_match == name else "alias"
                    is_exact_match = True  # Set exact match flag
                    break
                    
                # 2. One is substring of the other
                elif potential_match_lower in search_text_lower or search_text_lower in potential_match_lower:
                    # Calculate how much of the shorter string is contained in the longer one
                    shorter = min(len(potential_match_lower), len(search_text_lower))
                    longer = max(len(potential_match_lower), len(search_text_lower))
                    containment_score = shorter / longer * 0.9  # 90% of exact match
                    
                    if containment_score > best_match_score:
                        best_match_score = containment_score
                        best_match_text = potential_match
                        best_match_type = "name" if potential_match == name else "alias"
                
                # 3. Token overlap
                else:
                    match_tokens = set(potential_match_lower.split())
                    common_tokens = search_tokens.intersection(match_tokens)
                    
                    if common_tokens:
                        # Calculate Jaccard similarity: intersection / union
                        jaccard = len(common_tokens) / len(search_tokens.union(match_tokens))
                        overlap_score = jaccard * 0.8  # 80% of exact match
                        
                        if overlap_score > best_match_score:
                            best_match_score = overlap_score
                            best_match_text = potential_match
                            best_match_type = "name" if potential_match == name else "alias"
            
            # Apply score boost based on match quality
            if best_match_score > 0:
                # Check if this is an exact match AND it contains common terms
                if is_exact_match:
                    # Estimate term commonality using token count
                    token_count = len(search_tokens)
                    
                    # Names with more tokens often contain more common terms (like "hospital", "universitario")
                    # Apply an extra boost to compensate for common terms in longer names
                    common_terms_compensation = min(1.0, token_count / 3)  # 0.33 per token up to 3 tokens
                    
                    # Higher boost for exact matches
                    name_boost = 0.2 if best_match_type == "name" else 0.1
                    
                    # Apply an even stronger boost for exact matches
                    boost_factor = 2.0 + common_terms_compensation + name_boost
                    
                    if self.debug:
                        print(f"=> EXACT MATCH: {name} with {search_text} - " 
                              f"common_terms_compensation={common_terms_compensation:.2f}, "
                              f"boost_factor={boost_factor:.2f}")
                else:
                    # Scale the boost based on match quality (1.0 to 2.0)
                    # Give a slight additional boost to name matches vs alias matches
                    name_boost = 0.1 if best_match_type == "name" else 0
                    boost_factor = 1.0 + best_match_score + name_boost
                
                # Apply the boost
                hit.score *= boost_factor
            
                if self.debug:
                    match_type = f"EXACT ({best_match_score:.2f})" if is_exact_match else f"PARTIAL ({best_match_score:.2f})"
                    print(f"=> Name candidate: {name} - best match for {search_text}: {match_type} {best_match_type}: '{best_match_text}' - Score boosted by {boost_factor:.2f}x. Updated hit score: {hit.score}")
                
            else:
                if self.debug:
                    print(f"=> Name candidate: {name} - NO MATCH for {search_text}: {name} or {aliases_list}")
            
            refined_results.append(hit)
        
        # Sort by score descending
        return sorted(refined_results, key=lambda x: x.score, reverse=True)
    
    def get_single_prediction(self, organization, num_candidates=None):
        """
        Get predictions for one organization, returning multiple candidates in dictionary format.
        
        Args:
            organization: Organization to find candidates for
            num_candidates: Number of candidates to return (defaults to self.return_num_candidates)
            
        Returns:
            List of candidate dictionaries with standard fields
        """
        if num_candidates is None:
            num_candidates = self.return_num_candidates
                
        # Build affiliation string
        affiliation = []
        suborg_ner = organization.get('suborg', '')
        org_ner = organization['main']
        location_ner = organization['location']
        
        if suborg_ner:
            affiliation.append(suborg_ner)
        affiliation.append(org_ner)
        if location_ner:
            affiliation.append(location_ner)
            
        affiliation_string_ner = ', '.join(affiliation)
        
        if self.debug:
            print(f"\nWhooshLinker.get_single_prediction: Getting prediction for {affiliation_string_ner}")
        
        # Check cache first
        cached_results, is_reranked = self._get_predictions_from_cache(affiliation_string_ner, num_candidates)
        if cached_results is not None:
            if self.debug:
                print(f"WhooshLinker.get_single_prediction: Returning cached results")
            return cached_results
            
        # Get candidate matches
        candidate_matches = self.get_candidate_matches(organization)
        
        # Process candidates
        candidates_list = []
        
        # Detect if query contains non-Latin characters
        query_has_non_latin = contains_non_latin(org_ner)
        
        # Determine effective threshold based on script type
        if query_has_non_latin:
            effective_threshold = self.threshold_score * ADJUST_FACTOR_THRESHOLD_NON_LATIN  # Lower threshold for non-Latin scripts
        else:
            effective_threshold = self.threshold_score
        
        if candidate_matches:
            # Convert each match to the standard dictionary format
            for candidate in candidate_matches[:num_candidates]:
                entity_id = candidate.get('id', '')
                
                if entity_id:
                    raw_score = float(getattr(candidate, 'score', 0.0))
                    
                    # Get candidate name for script detection
                    candidate_name = candidate.get('name', '')
                    
                    # Use the enhanced normalization that handles script differences
                    normalized_score = self._normalize_score(raw_score, org_name=candidate_name)
                    
                    # Create standardized candidate dictionary
                    candidate_dict = {
                        "id": entity_id,
                        "name": candidate_name,
                        "enc_score": normalized_score,
                        "orig_score": raw_score,
                        "source": "whoosh",  # Retrieval method
                        "data_source": self.data_source,  # Data source
                        "query_org": organization,
                        "explanation": f"Matched with Whoosh with raw score {raw_score:.2f} → normalized to {normalized_score:.4f}"
                    }
                    
                    # Add additional fields if available
                    if 'city' in candidate:
                        candidate_dict['city'] = candidate['city']
                    if 'country_name' in candidate:
                        candidate_dict['country'] = candidate['country_name']
                    elif 'country' in candidate:
                        candidate_dict['country'] = candidate['country']
                    
                    # Add acronyms if available
                    if 'acronyms' in candidate:
                        candidate_dict['acronyms'] = candidate['acronyms']
                        
                    # Add script information
                    candidate_dict['is_non_latin'] = contains_non_latin(candidate_name)
                        
                    candidates_list.append(candidate_dict)
        
        # Filter candidates by effective threshold
        candidates_list = [c for c in candidates_list if c["enc_score"] >= effective_threshold]
        
        # If no results, return a placeholder
        if not candidates_list:
            candidates_list = [{
                "id": None,
                "name": None,
                "enc_score": 0.0,
                "orig_score": 0.0,
                "source": "whoosh",
                "data_source": self.data_source,
                "query_org": organization,
                "explanation": "No matches found or all below threshold"
            }]
        
        # Update cache
        self._update_predictions_cache(affiliation_string_ner, candidates_list, reranked=False)
        
        return candidates_list
        
    def get_candidate_matches(self, organization):
        """Perform search using name, aliases, parent, and location."""

        org_name = organization["main"]
        suborg = organization.get("suborg", "")
        city = organization["city"]
        region = organization["region"]
        country = organization["country"]

        if not org_name:
            return []

        name_variants = get_variants_list([org_name])
        name_variants.extend(translate_institution_name(org_name))
        clean_name_variants = {self._clean_organization_name(name_variant) for name_variant in name_variants if name_variant.strip()}
        clean_name_variants = {variant for variant in clean_name_variants if variant.strip()}
        
        # Get acronym variants
        acronym_variants = self.abbreviations.get(org_name.lower(), [])
        clean_name_variants.union(set(acronym_variants))
                   
        if suborg:
            suborg_variants = get_variants_list([suborg])
            suborg_variants.extend(translate_institution_name(suborg))
            clean_suborg_variants = {self._clean_organization_name(suborg_variant) for suborg_variant in suborg_variants if suborg_variant.strip()}
            clean_suborg_variants = {variant for variant in clean_suborg_variants if variant.strip()}
            combined_variants = {f"{suborg}, {org}" for org in clean_name_variants for suborg in clean_suborg_variants 
                                if org.strip() and suborg.strip()}
            clean_name_variants = combined_variants if combined_variants else clean_name_variants
        
        # Get country variants.
        country_variants = set()
        if country:
            country_variants.union(set(get_variants_country(country)))
        
        with self.ix.searcher() as searcher:
            # Build our queries
            name_queries = []
            location_queries = []
            # Store descriptions for debugging
            query_descriptions = []  
            location_descriptions = ["Location"]
            
            # Name variant queries with proper boosting
            for name_variant in clean_name_variants:
                # Add exact phrase query for multi-word terms (highest priority)
                if len(name_variant.split()) > 1:
                    exact_query = self.name_parser.parse(f'"{name_variant}"')  # Quote for exact phrase
                    exact_query = exact_query.with_boost(3.0)  # Triple weight for exact match
                    name_queries.append(exact_query)
                    query_descriptions.append(f"Exact name match: '{name_variant}'. Query: {exact_query}")

                # Regular term query (medium priority)
                name_query = self.name_parser.parse(name_variant)
                name_query = name_query.with_boost(1.5)  # 50% more weight
                name_queries.append(name_query)
                query_descriptions.append(f"Name match: '{name_variant}'. Query: {name_query}")
                
                # Alias query (medium priority) - now use aliases_text field
                alias_query = self.alias_parser.parse(name_variant)
                alias_query = alias_query.with_boost(1.3)  # 30% more weight
                name_queries.append(alias_query)
                query_descriptions.append(f"Alias match: '{name_variant}'. Query: {alias_query}")

                # Regular term query (lower priority - no boost)
                name_flex_query = self.name_parser_flex.parse(name_variant)
                name_queries.append(name_flex_query)
                query_descriptions.append(f"Name match (flex): '{name_variant}'. Query: {name_flex_query}")
                
                # Alias query (lower priority - no boost) - now use aliases_text field
                alias_flex_query = self.alias_parser_flex.parse(name_variant)
                name_queries.append(alias_flex_query)
                query_descriptions.append(f"Alias match (flex): '{name_variant}'. Query: {alias_flex_query}")
                
                # Fuzzy query (lowest priority - no boost)
                # Only add fuzzy query if the term is at least 3 characters long
                if len(name_variant) >= 10:
                    fuzzy_query = FuzzyTerm("name", name_variant, maxdist=2, boost=1.0)
                    name_queries.append(fuzzy_query)
                    query_descriptions.append(f"Fuzzy name match: '{name_variant}'. Query: {fuzzy_query}")

            # Location data.
            if city:
                city_query = Term("city", city.lower(), boost=1.5)  # Higher boost for city
                location_queries.append(city_query)
                location_descriptions.append(f"\n   City: '{city}'. Query: {city_query}")
            
            if region:
                region_query = Term("region", region.lower(), boost=1)
                location_queries.append(region_query)
                location_descriptions.append(f"\n   Region: '{region}'. Query: {region_query}")
                
            if country:
                country_query = Term("country_name", country.lower(), boost=1)
                location_queries.append(country_query)
                location_descriptions.append(f"\n   Country: '{country}'. Query: {country_query}")
            
            # Add country queries from variants
            for country in country_variants:
                if country.lower() != organization.get('country', '').lower():
                    country_variant_query = Term("country_name", country.lower(), boost=1)
                    location_queries.append(country_variant_query)
                    location_descriptions.append(f"\n   Country Variant: '{country}'. Query: {country_variant_query}")
                
            query_descriptions.append(f"Location: {', '.join(location_descriptions)}")
           
            # Combine name queries (OR).
            if name_queries:
                combined_name_query = Or(name_queries)
            else:
                combined_name_query = None

            # Combine location queries (OR).
            if location_queries:
                combined_location_query = Or(location_queries)
            else:
                combined_location_query = None
            
            # Combine queries.
            if combined_name_query and combined_location_query:
                final_query = And([combined_name_query, combined_location_query])
            elif combined_name_query:
                final_query = combined_name_query
            else:
                final_query = None

            if final_query:
                # After combining queries and running the final search:
                all_results = searcher.search(final_query, limit=self.max_hits)
                
                if all_results:
                    all_results = self._refine_names_alias_matches(all_results, org_name)

                if not all_results:
                    return []

                # Consider top-10 initially.
                top_entities = {
                    f"{hit.get('name', '')} ({hit.get('id', '')})": {
                        "score": getattr(hit, "score", 0),
                        "matches": hit.get("matches", {})
                    }
                    for hit in all_results[:10]
                }

                self._adjust_scores_based_on_match_type(top_entities)

                # Filter and format results.
                filtered_results = [hit for hit in all_results if getattr(hit, "score", 0) >= self.threshold_score]
                filtered_results.sort(key=lambda hit: hit.score, reverse=True)

                ror_seen = set()
                unique_results = []
                for hit in filtered_results:
                    entity_id = hit.get('id', '')
                    if entity_id and entity_id not in ror_seen:
                        ror_seen.add(entity_id)
                        unique_results.append(hit)
                return unique_results

            else:
                if self.debug:
                    print("\nNo query could be generated from the parsed data.")
                return []
            
            return []
