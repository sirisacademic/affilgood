import os
import sys
from .base_linker import BaseLinker
from .constants import *
from .data_manager import DataManager
from .constants import (
    MAINORG_NER_LABEL, SUBORG_NER_LABEL, CITY_NER_LABEL, REGION_NER_LABEL, COUNTRY_NER_LABEL, 
    NER_ENTITY_TEXT_FIELD, ROR_ID_FIELD, ROR_NAME_FIELD,
    CITY_OSM_LABEL, PROVINCE_OSM_LABEL, STATE_OSM_LABEL, COUNTRY_OSM_LABEL,
    THRESHOLD_SCORE_FILTER_FIRSTAGE_EL, THRESHOLD_SCORE_RERANKED_EL, NUM_CANDIDATES_TO_RERANK
)

# Ensure S2AFF is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH)))

class S2AFFLinker(BaseLinker):
    """S2AFF-based entity linker."""
    
    def __init__(self, data_manager=None, device="cpu", ror_dump_path=None, debug=False):
        """
        Initialize the S2AFF linker.
        
        Args:
            data_manager (DataManager, optional): Data manager instance for data operations.
            device (str, optional): Device to use for computations ('cpu' or 'cuda'). Defaults to "cpu".
            ror_dump_path (str, optional): Path to ROR dump file. If provided, will override data_manager lookup.
        """
        super().__init__()
        self.device = device
        self.data_manager = data_manager
        self.ror_dump_path = ror_dump_path
        self.debug = debug
        
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
        
        # Get the ROR dump path if not already provided
        if not self.ror_dump_path and self.data_manager is not None:
            self.ror_dump_path = self.data_manager.get_latest_ror()
        
        if not self.ror_dump_path:
            raise FileNotFoundError("Failed to retrieve the latest ROR dump.")
            
        # Initialize S2AFF specific components
        try:
            # Import S2AFF modules here to avoid circular imports
            from s2aff.ror import RORIndex
            from s2aff.model import PairwiseRORLightGBMReranker
            
            # Load ROR index
            print(f"Loading ROR index from {self.ror_dump_path}...")
            self.ror_index = RORIndex(self.ror_dump_path)
            
            # Load pairwise model for S2AFF
            print(f'Getting pairwise ROR LightGBM reranker...')
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
        # Download S2AFF data if needed
        self.data_manager.download_s2aff_data()
        # Update OpenAlex work counts
        self.data_manager.update_openalex_works_counts()
            
    def get_single_prediction(self, organization):
        """Get S2AFF predictions for one input organization."""
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
            print(f"Getting prediction for {affiliation_string_ner}")
        predicted_id, predicted_name, predicted_score = self._get_from_cache(affiliation_string_ner)
        if predicted_id is not None:
            return predicted_id, predicted_name, predicted_score
        # If not in cache, get candidates
        first_stage_candidates, first_stage_scores = self.ror_index.get_candidates_from_main_affiliation(
            org_ner,
            location_ner,
            [suborg_ner]
        )
        if self.debug:
            print(f"Candidates: {first_stage_candidates[:10]}")
            print(f"Scores: {first_stage_scores[:10]}")
        # Filter candidates by score
        len_filtered_scores = len([
            s for s in first_stage_scores
            if s >= THRESHOLD_SCORE_FILTER_FIRSTAGE_EL
        ])
        candidates = first_stage_candidates[:len_filtered_scores]
        scores = first_stage_scores[:len_filtered_scores]
        if candidates:
            # Rerank candidates
            reranked_candidates, reranked_scores = self.pairwise_model.predict(
                affiliation_string_ner, candidates[:NUM_CANDIDATES_TO_RERANK], scores[:NUM_CANDIDATES_TO_RERANK]
            )
            # Process top candidate
            top_rr_score = reranked_scores[0]
            if top_rr_score >= THRESHOLD_SCORE_RERANKED_EL:
                top_rr_ror_idx = reranked_candidates[0]
                if (top_rr_ror_idx in self.ror_index.ror_dict and
                    ROR_ID_FIELD in self.ror_index.ror_dict[top_rr_ror_idx]):
                    predicted_id = self.ror_index.ror_dict[top_rr_ror_idx][ROR_ID_FIELD]
                    predicted_score = top_rr_score
                    predicted_name = self.ror_index.ror_dict[predicted_id][ROR_NAME_FIELD] if ROR_NAME_FIELD in self.ror_index.ror_dict[predicted_id] else predicted_id
                    # Update cache
                    self._update_cache(affiliation_string_ner, predicted_id, predicted_name, predicted_score)
        return predicted_id, predicted_name, predicted_score
      


