import os
import sys
from .base_linker import BaseLinker
from .constants import *
from .data_manager import DataManager
from .constants import *

# Ensure S2AFF is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH)))

class S2AFFLinker(BaseLinker):
    """S2AFF-based entity linker."""
    
    def __init__(
                self,
                data_manager=None,
                ror_dump_path=None,
                return_num_candidates=NUM_CANDIDATES_TO_RETURN,
                threshold_score=0.30,
                debug=False,
                rerank=True,
                use_cache=True,
                data_source="ror"):
        """
        Initialize the S2AFF linker.
        
        Args:
            data_manager (DataManager, optional): Data manager instance for data operations.
            ror_dump_path (str, optional): Path to ROR dump file. If provided, will override data_manager lookup.
            return_num_candidates (int): Number of candidates to return.
            threshold_score (float): Minimum score threshold for candidates.
            debug (bool): Whether to show debug output.
            rerank (bool): Whether to apply S2AFF's internal reranking.
            use_cache (bool): Whether to use caching.
            data_source (str): Data source identifier (kept for compatibility, S2AFF only works with ROR).
        """
        if data_source != "ror":
            if verbose:
                print(f"S2AFF linker only supports ROR data. Processing ROR data alongside other linkers for '{data_source}'.")
            data_source = "ror"  # Force to ROR
        
        # Call BaseLinker constructor with required parameters
        super().__init__(use_cache=use_cache, data_source=data_source)
        
        self.data_manager = data_manager
        self.ror_dump_path = ror_dump_path
        self.debug = debug
        self.rerank = rerank
        self.return_num_candidates = return_num_candidates
        self.threshold_score = threshold_score
        self.data_source = data_source  # Store data source (always "ror" for S2AFF)
        
        # These will be initialized later
        self.ror_index = None
        self.pairwise_model = None
        
        # Initialization is deferred until we have a data_manager or ror_dump_path
        self.is_initialized = False
        
        # If we have what we need, initialize now
        if self.ror_dump_path or (self.data_manager is not None):
            self.initialize()
    
    def initialize(self):
        """Initialize the S2AFF linker components."""
        if self.is_initialized:
            return
            
        # Prepare the necessary data
        if self.data_manager is not None:
            self._prepare_data()
        
        # Initialize S2AFF specific components
        try:
            # Import S2AFF modules here to avoid circular imports
            # Load ROR index
            if self.debug:
                print(f"Loading ROR index from {self.ror_dump_path}...")
            from s2aff.ror import RORIndex
            self.ror_index = RORIndex(self.ror_dump_path)
            
            # Load pairwise model for S2AFF
            if self.rerank:
                if self.debug:
                    print(f'Getting pairwise ROR LightGBM reranker...')
                from s2aff.model import PairwiseRORLightGBMReranker
                self.pairwise_model = PairwiseRORLightGBMReranker(self.ror_index)
            
            self.is_initialized = True
            if self.debug:
                print("S2AFF linker initialized successfully")
        except Exception as e:
            if self.debug:
                print(f"Error initializing S2AFF components: {e}")
            raise

    def _prepare_data(self):
        """Prepare S2AFF-specific data."""
        if self.data_manager is None:
            raise RuntimeError("Cannot prepare data: data_manager is not set")
        
        # Resolve missing S2AFF paths to URLs (excluding ROR data)
        self.data_manager.resolve_s2aff_paths()
        # Ensure S2AFF-specific files exist or are downloaded (excluding ROR data)
        self.data_manager.ensure_s2aff_files()
        
        # Get ROR dump from shared location instead of S2AFF-specific location
        self.ror_dump_path = self.data_manager.get_latest_ror(self.ror_dump_path)
        if not self.ror_dump_path:
            raise FileNotFoundError(f"Failed to get required ROR dump from shared location.")
        
        if self.debug:
            print(f"S2AFF using shared ROR dump: {self.ror_dump_path}")
            
            
    def get_single_prediction(self, organization, num_candidates=None):
        """
        Get S2AFF predictions for one input organization, returning multiple candidates
        in the standardized dictionary format.
        
        Args:
            organization: Organization dictionary with 'main', 'suborg', 'location', etc.
            num_candidates: Number of candidates to return
            
        Returns:
            List of candidate dictionaries with standardized fields
        """
        if num_candidates is None:
            num_candidates = self.return_num_candidates
        
        # Ensure initialization
        if not self.is_initialized:
            self.initialize()
        
        # Build affiliation components
        affiliation = []
        suborg = organization.get('suborg', '')
        org = organization['main']
        location = organization['location']
        
        if suborg:
            affiliation.append(suborg)
        affiliation.append(org)
        if location:
            affiliation.append(location)
        affiliation_string = ', '.join(affiliation)
        
        if self.debug:
            print(f"S2AFFLinker: Getting prediction for {affiliation_string}")
        
        # Create cache key with data source prefix for consistency
        cache_key = f"{self.data_source}:{affiliation_string}"
        
        # Check cache first
        cached_results, is_reranked = self._get_predictions_from_cache(cache_key, num_candidates)
        if cached_results is not None:
            if self.debug:
                print(f"S2AFFLinker: Returning cached results")
            return cached_results
        
        # Get candidates from S2AFF
        if self.debug:
            print(f"Retrieving first-stage candidates for org: {org}, location: {location}, suborg: {[suborg]}")
        
        first_stage_candidates, first_stage_scores = self.ror_index.get_candidates_from_main_affiliation(org, location, [suborg])
        
        # Filter candidates by score
        len_filtered_scores = len([
            s for s in first_stage_scores
            if s >= THRESHOLD_SCORE_FILTER_FIRSTAGE_EL
        ])
        candidates = first_stage_candidates[:len_filtered_scores][:NUM_CANDIDATES_TO_RERANK]
        scores = first_stage_scores[:len_filtered_scores][:NUM_CANDIDATES_TO_RERANK]
        
        if self.debug:
            candidates_with_scores = [f"{candidates[i]}:{scores[i]:.4f}" for i in range(len(candidates))]
            print(f"Candidates first stage filtered (threshold {THRESHOLD_SCORE_FILTER_FIRSTAGE_EL}, num. candidates:{NUM_CANDIDATES_TO_RERANK}): {candidates_with_scores}")
        
        # Initialize the results list
        candidates_list = []
        
        if candidates:
            # Apply reranking if enabled
            if self.rerank and self.pairwise_model:
                # Rerank candidates
                if self.debug:
                    print(f"Retrieving re-ranked candidates for affiliation_string: {affiliation_string}, candidates: {candidates}, scores: {scores}")
                
                reranked_candidates, reranked_scores = self.pairwise_model.predict(
                    affiliation_string, candidates, scores
                )
                
                if self.debug:
                    reranked_candidates_with_scores = [f"{reranked_candidates[i]}:{reranked_scores[i]:.4f}" for i in range(len(reranked_candidates))]
                    print(f"Re-ranked candidates: {reranked_candidates_with_scores}")
                
                # Use reranked results
                final_candidates = reranked_candidates
                final_scores = reranked_scores
                score_threshold = THRESHOLD_SCORE_RERANKED_EL
                explanation_suffix = "with S2AFF reranking"
            else:
                # Use first-stage results without reranking
                final_candidates = candidates
                final_scores = scores
                score_threshold = THRESHOLD_SCORE_FILTER_FIRSTAGE_EL
                explanation_suffix = "without reranking"
            
            # Convert to standardized dictionary format
            for i, (ror_idx, score) in enumerate(zip(final_candidates[:num_candidates], final_scores[:num_candidates])):
                if score >= score_threshold:
                    if (ror_idx in self.ror_index.ror_dict and
                        ROR_ID_FIELD in self.ror_index.ror_dict[ror_idx]):
                        
                        # Get organization info from ROR index
                        ror_org = self.ror_index.ror_dict[ror_idx]
                        predicted_id = ror_org[ROR_ID_FIELD]
                        
                        # Get full organization record for additional fields
                        full_org_record = self.ror_index.ror_dict.get(predicted_id, {})
                        predicted_name = full_org_record.get(ROR_NAME_FIELD, predicted_id)
                        
                        # Create standardized candidate dictionary
                        candidate_dict = {
                            "id": predicted_id,
                            "name": predicted_name,
                            "enc_score": float(score),
                            "orig_score": float(score),  # S2AFF doesn't distinguish between these
                            "source": "s2aff",
                            "data_source": self.data_source,
                            "query_org": organization,
                            "explanation": f"Matched with S2AFF {explanation_suffix} with score {score:.4f}"
                        }
                        
                        # Add additional fields if available from ROR record
                        if 'addresses' in full_org_record and full_org_record['addresses']:
                            address = full_org_record['addresses'][0]
                            if 'city' in address:
                                candidate_dict['city'] = address['city']
                            if 'country_name' in full_org_record.get('country', {}):
                                candidate_dict['country'] = full_org_record['country']['country_name']
                        
                        if 'acronyms' in full_org_record:
                            candidate_dict['acronyms'] = full_org_record['acronyms']
                        
                        candidates_list.append(candidate_dict)
        
        # If no results, return a placeholder in the expected format
        if not candidates_list:
            candidates_list = [{
                "id": None,
                "name": None,
                "enc_score": 0.0,
                "orig_score": 0.0,
                "source": "s2aff",
                "data_source": self.data_source,
                "query_org": organization,
                "explanation": "No matches found or all below threshold"
            }]
        
        # Update cache with standardized results
        self._update_predictions_cache(cache_key, candidates_list, reranked=False)
        
        return candidates_list

