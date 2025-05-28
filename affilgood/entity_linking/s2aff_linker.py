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
                debug=False,
                rerank=True):
        """
        Initialize the S2AFF linker.
        
        Args:
            data_manager (DataManager, optional): Data manager instance for data operations.
            ror_dump_path (str, optional): Path to ROR dump file. If provided, will override data_manager lookup.
        """
        super().__init__()
        self.data_manager = data_manager
        self.ror_dump_path = ror_dump_path
        self.debug = debug
        self.rerank = rerank
        self.return_num_candidates = return_num_candidates
        
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
        
        """
        # Get the ROR dump path if not already provided
        if not self.ror_dump_path and self.data_manager is not None:
            self.ror_dump_path = self.data_manager.get_latest_ror()
        
        if not self.ror_dump_path:
            raise FileNotFoundError("Failed to retrieve the latest ROR dump.")
        """
            
        # Initialize S2AFF specific components
        try:
            # Import S2AFF modules here to avoid circular imports
            # Load ROR index
            print(f"Loading ROR index from {self.ror_dump_path}...")
            from s2aff.ror import RORIndex
            self.ror_index = RORIndex(self.ror_dump_path)
            
            # Load pairwise model for S2AFF
            if self.rerank:
                print(f'Getting pairwise ROR LightGBM reranker...')
                from s2aff.model import PairwiseRORLightGBMReranker
                self.pairwise_model = PairwiseRORLightGBMReranker(self.ror_index)
            
            self.is_initialized = True
            print("S2AFF linker initialized successfully")
        except Exception as e:
            print(f"Error initializing S2AFF components: {e}")
            raise

    def _prepare_data(self):
        """Prepare S2AFF-specific data."""
        if self.data_manager is None:
            raise RuntimeError("Cannot prepare data: data_manager is not set")
        # Resolve missing paths to URLs
        self.data_manager.resolve_s2aff_paths()
        # Ensure files exist or are downloaded
        self.data_manager.ensure_s2aff_files()
        # Get or download ROR index.
        self.ror_dump_path = self.data_manager.get_latest_ror(self.ror_dump_path)
        if not self.ror_dump_path:
            raise FileNotFoundError(f"Failed to get required ROR dump.")
            
    def get_single_prediction(self, organization, num_candidates=None):
        """Get S2AFF predictions for one input organization, returning multiple candidates."""
        if num_candidates is None:
            num_candidates = self.return_num_candidates
        
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
            print(f"=> Getting prediction for {affiliation_string}")
        
        # Check cache first
        cached_results = self._get_predictions_from_cache(affiliation_string, num_candidates)
        if cached_results is not None:
            return cached_results
        
        # If not in cache, get candidates
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
            # Add both raw candidates and reranked candidates based on need
            
            # First, get raw candidates
            for i, (ror_idx, score) in enumerate(zip(candidates[:num_candidates], scores[:num_candidates])):
                if ror_idx in self.ror_index.ror_dict and ROR_ID_FIELD in self.ror_index.ror_dict[ror_idx]:
                    raw_id = self.ror_index.ror_dict[ror_idx][ROR_ID_FIELD]
                    raw_score = score
                    raw_name = self.ror_index.ror_dict[raw_id][ROR_NAME_FIELD] if ROR_NAME_FIELD in self.ror_index.ror_dict[raw_id] else raw_id
                    
                    # For combined retrieval mode, we could just return these candidates
                    if not hasattr(self, 'rerank') or not self.rerank:
                        candidates_list.append((raw_id, raw_name, raw_score))
            
            # If we want reranking and don't have enough candidates yet, get reranked ones
            if (hasattr(self, 'rerank') and self.rerank) or not candidates_list:
                # Rerank candidates
                if self.debug:
                    print(f"Retrieving re-ranked candidates for affiliation_string: {affiliation_string}, candidates: {candidates}, scores: {scores}")
                
                reranked_candidates, reranked_scores = self.pairwise_model.predict(
                    affiliation_string, candidates, scores
                )
                
                if self.debug:
                    reranked_candidates_with_scores = [f"{reranked_candidates[i]}:{reranked_scores[i]:.4f}" for i in range(len(reranked_candidates))]
                    print(f"Re-ranked candidates: {reranked_candidates_with_scores}")
                
                # Process all candidates up to num_candidates
                candidates_list = []  # Reset if we're doing reranking
                for i, (ror_idx, score) in enumerate(zip(reranked_candidates[:num_candidates], reranked_scores[:num_candidates])):
                    if score >= THRESHOLD_SCORE_RERANKED_EL:
                        if (ror_idx in self.ror_index.ror_dict and
                            ROR_ID_FIELD in self.ror_index.ror_dict[ror_idx]):
                            predicted_id = self.ror_index.ror_dict[ror_idx][ROR_ID_FIELD]
                            predicted_score = score
                            predicted_name = self.ror_index.ror_dict[predicted_id][ROR_NAME_FIELD] if ROR_NAME_FIELD in self.ror_index.ror_dict[predicted_id] else predicted_id
                            
                            # Add to candidates list
                            candidates_list.append((
                                predicted_id,
                                predicted_name,
                                predicted_score
                            ))
        
        # If we have no results, return a single None tuple
        if not candidates_list:
            candidates_list = [(None, None, 0.0)]
        
        # Update cache
        self._update_predictions_cache(affiliation_string, candidates_list)
        
        return candidates_list
      


