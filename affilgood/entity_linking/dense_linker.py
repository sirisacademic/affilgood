import os
import torch
import json
import re
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer, util

from .base_linker import BaseLinker
from .constants import *

# Configure the specific logger for this module
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:  # Only add handlers if none exist
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set to the desired level


class DenseLinker(BaseLinker):
    """
    Entity linker using dense retrieval.
    """
    
    def __init__(
        self, 
        data_manager=None,
        encoder_path=None,
        batch_size=32,
        scores_span_text=False,
        return_num_candidates=NUM_CANDIDATES_TO_RETURN,
        threshold_score=0.30,
        use_hnsw=USE_HNSW_BY_DEFAULT,
        rebuild_index=False,
        debug=False,
        use_cache=True,
        data_source="ror",
        org_types=None,     # WikiData org types
        countries=None,     # WikiData countries
        use_wikidata_labels_with_ror=False # Whether to add WikiData labels from previously downloaded file when generating the ROR indices
    ):
        # Pass data_source to BaseLinker
        super().__init__(use_cache=use_cache, data_source=data_source,
                        wikidata_org_types=org_types, wikidata_countries=countries)
        self.data_manager = data_manager
        self.encoder_path = encoder_path if encoder_path else ENCODER_DEFAULT_MODEL
        self.batch_size = batch_size
        self.scores_span_text = scores_span_text
        self.return_num_candidates = return_num_candidates
        self.threshold_score = threshold_score
        self.debug = debug
        self.use_hnsw = use_hnsw
        self.rebuild_index = rebuild_index
        self.hnsw_index = None
        self.data_source = data_source  # Store data_source locally
        self.org_types = org_types
        self.countries = countries
        self.use_wikidata_labels_with_ror=use_wikidata_labels_with_ror
        
        # These will be initialized later
        self.encoder = None
        self.org_embeddings = None
        self.org_data = None
        
        # Initialize if data_manager is provided
        if self.data_manager is not None:
            self.initialize()
            
    def initialize(self):
        """Initialize the dense entity linker components."""
        if self.is_initialized:
            return
            
        # Prepare necessary data
        if self.data_manager is not None:
            self._prepare_data()
        
        try:           
            # Initialize encoder
            if self.debug:
                logger.info(f"Initializing encoder: {self.encoder_path}")
            self.encoder = SentenceTransformer(self.encoder_path)
            
            # Add special tokens if needed
            if hasattr(self.encoder, 'tokenizer') and all(token not in self.encoder.tokenizer.vocab for token in SPECIAL_TOKENS):
                logger.info("Adding special tokens to encoder tokenizer")
                self.encoder.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
                self.encoder._first_module().auto_model.resize_token_embeddings(len(self.encoder.tokenizer))
            
            # Process organization data and create organization representations
            if self.data_manager:
                self._load_organizations()
            
            self.is_initialized = True
            logger.info("Dense entity linker initialized successfully")
        except Exception as e:
            logger.info(f"Error initializing dense entity linker: {e}")
            raise
    
    def _prepare_data(self):
        """Prepare data for the dense entity linker."""
        # Implementation updated for source-agnostic approach
        if self.data_manager is None:
            raise RuntimeError("Cannot prepare data: data_manager is not set")

        # If using HNSW, ensure the index exists
        if self.use_hnsw:
            try:
                # Check if hnswlib is installed
                import hnswlib
                
                # Get or create the appropriate HNSW index for this data source
                hnsw_index_dir = self.data_manager.get_or_create_index(
                    source=self.data_source,  # Use the stored data_source
                    indices_type='hnsw',
                    encoder_path=self.encoder_path,
                    force_rebuild=self.rebuild_index,
                    org_types=self.org_types,
                    countries=self.countries,
                    use_wikidata_labels_with_ror=self.use_wikidata_labels_with_ror
                )
                
                # Store the hnsw_index_path attribute needed by the reranker
                self.hnsw_index_path = hnsw_index_dir
                
                # Load the index and metadata
                self.hnsw_index, self.hnsw_metadata = self.data_manager.load_hnsw_index(
                    hnsw_index_dir=hnsw_index_dir,
                    source=self.data_source  # Use the stored data_source
                )
                
                if self.hnsw_index and self.hnsw_metadata:
                    # Load organization data from metadata
                    self.org_data = self.hnsw_metadata.get("org_data", [])
                    logger.info(f"Loaded {len(self.org_data)} organizations from HNSW metadata")
                else:
                    self.use_hnsw = False
                    logger.info("Failed to load HNSW index. Falling back to PyTorch search.")
                    
            except ImportError:
                self.use_hnsw = False
                logger.info("hnswlib not installed. Falling back to PyTorch search.")
      
    def _load_organizations(self):
        """Load and encode organizations from the specified data source."""
        # If using and index and we've already loaded the data from metadata, skip this step
        if self.use_hnsw and hasattr(self, 'org_data') and self.org_data:
            return

        # Get organization data based on data source
        org_data = []
        
        if self.data_source == 'ror':
            # Get ROR data
            ror_dump_path = self.data_manager.get_latest_ror()
            if not ror_dump_path:
                raise FileNotFoundError("Failed to get required ROR dump.")
            
            # Load ROR data
            if self.debug:
                logger.info(f"Loading organizations from {ror_dump_path}")
            with open(ror_dump_path, 'r', encoding='utf-8') as f:
                org_data = json.load(f)
        elif self.data_source == 'wikidata':
            # Get WikiData organizations
            wikidata_orgs = self.data_manager.get_wikidata_organizations()
            if wikidata_orgs is None or wikidata_orgs.empty:
                raise ValueError("Failed to retrieve WikiData organizations")
                
            # Convert DataFrame to list of dicts
            org_data = wikidata_orgs.to_dict('records')
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
        
        # Format each organization
        self.org_data = []
        for org in org_data:
            # Use a source-agnostic approach to extract fields
            # Process the organization using DataManager's standardized format
            std_org = self.data_manager.process_organization_for_index(org, self.data_source)
            
            # Extract info for text representation
            org_id = std_org.get('id', '')
            name = std_org.get('name', '')
            acronyms = std_org.get('acronyms', '').split(' ||| ') if std_org.get('acronyms') else []
            city = std_org.get('city', '')
            country = std_org.get('country_name', '')
            parent = std_org.get('parent', '')
            
            # Create a basic organization data entry
            org_entry = {
                "id": org_id,
                "name": name,
                "acronyms": acronyms,
                "city": city,
                "country": country,
                "parent": parent,
                "data_source": self.data_source,
                "text_representations": []  # Will store all text formats
            }
            
            # Create canonical text representation
            canonical_text = f"[MENTION] {name}"
            if acronyms and len(acronyms) > 0:
                canonical_text += f" [ACRONYM] {acronyms[0]}"
            if parent:
                canonical_text += f" [PARENT] {parent}"
            if city:
                canonical_text += f" [CITY] {city}"
            if country:
                canonical_text += f" [COUNTRY] {country}"
                
            # Add canonical representation
            org_entry["text"] = canonical_text  # Store for later reference
            org_entry["text_representations"].append(canonical_text)
            
            # Add the organization entry to our data
            self.org_data.append(org_entry)
        
        # Compute embeddings in batches
        logger.info(f"Computing embeddings for {len(self.org_data)} organizations")
        
        # Get text representations
        texts = [org["text"] for org in self.org_data]
        
        # Compute embeddings in batches
        embeddings_list = []
        batch_size = self.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.encoder.encode(
                batch_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings_list.append(batch_embeddings)
        
        # Concatenate all embeddings
        self.org_embeddings = torch.cat(embeddings_list, dim=0)
        
        # Normalize embeddings
        self.org_embeddings = util.normalize_embeddings(self.org_embeddings)
        
        if self.debug:
            logger.info(f"Computed and normalized {len(self.org_data)} embeddings")
    
    def search_with_hnsw(self, query_embedding_normalized, top_k=20):
        """Search for similar embeddings using HNSW with enriched representations."""
        # Implementation unchanged from original
        if not self.hnsw_index:
            return None, None
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(query_embedding_normalized):
            # Remove batch dimension if present
            if query_embedding_normalized.dim() > 1 and query_embedding_normalized.shape[0] == 1:
                query_embedding_normalized = query_embedding_normalized.squeeze(0)
            query_embedding_normalized = query_embedding_normalized.cpu().numpy()
        
        # Get nearest neighbors from HNSW
        indices, distances = self.hnsw_index.knn_query(query_embedding_normalized, k=top_k * 2)  # Request more candidates
        
        # Flatten arrays if they're 2D
        indices = indices.flatten()
        distances = distances.flatten()
        
        # Convert distances to similarities
        scores = 1.0 - distances
        
        # Map embedding indices to organization indices
        if hasattr(self, 'hnsw_metadata') and 'embedding_to_org_mapping' in self.hnsw_metadata:
            # Get mapping
            embedding_to_org_mapping = self.hnsw_metadata['embedding_to_org_mapping']
            
            # Convert embedding indices to organization indices
            org_indices = [embedding_to_org_mapping[int(idx)] for idx in indices]
            
            # Remove duplicates while preserving order
            seen_orgs = set()
            unique_org_indices = []
            unique_scores = []
            
            for org_idx, score in zip(org_indices, scores):
                if org_idx not in seen_orgs:
                    seen_orgs.add(org_idx)
                    unique_org_indices.append(org_idx)
                    unique_scores.append(score)
                    
                    # Stop once we have enough unique organizations
                    if len(unique_org_indices) >= top_k:
                        break
            
            if self.debug:
                logger.info("\n\nTop HNSW matches:")
                for idx, (org_idx, score) in enumerate(zip(unique_org_indices, unique_scores)):
                    if idx >= top_k:
                        break
                    # Get organization information from metadata
                    if hasattr(self, 'hnsw_metadata') and 'org_data' in self.hnsw_metadata:
                        org_data = self.hnsw_metadata['org_data'][org_idx]
                        org_name = org_data.get('name', 'Unknown')
                        # Change from 'ror_id' to 'id' for consistency
                        org_id = org_data.get('id', 'Unknown ID')
                        logger.info(f"  {idx+1}. {org_name} ({org_id}) - Score: {score:.4f}")
            
            return unique_scores, unique_org_indices
        else:
            # If no mapping exists, just use the embedding indices directly
            if self.debug:
                logger.info(f"HNSW scores: {scores[:top_k]}")
            
            return scores.tolist()[:top_k], indices.tolist()[:top_k]
             
    def get_single_prediction(self, organization, num_candidates=None):
        """
        Get prediction for a single organization using dense retrieval and selective re-ranking.
        Returns list of candidates for the top-k candidates.
        """
        if num_candidates is None:
            num_candidates = self.return_num_candidates
        
        # Ensure initialization
        if not self.is_initialized:
            self.initialize()
        
        # Format the query
        affiliation = []
        org = organization.get('main', organization['span_text'])
        suborg = organization.get('suborg', '')
        location = organization['location']
        city = organization.get('city', '')
        country = organization.get('country', '')
        
        if suborg:
            affiliation.append(suborg)
        affiliation.append(org)
        if location:
            affiliation.append(location)
        
        affiliation_string = ', '.join(affiliation)
        
        # Extract acronym if present
        acronym = ''
        acronym_match = re.search(r'\(([A-Z]+)\)', org)
        if acronym_match:
            acronym = acronym_match.group(1)
            org = org.replace(acronym_match.group(0), '').strip()
        
        # Format for dense retrieval
        query = f"[MENTION] {org}"
        
        if acronym:
            query += f" [ACRONYM] {acronym}"        
        
        if suborg:
            query = f"[MENTION] {suborg} [PARENT] {org}"
        if city:
            query += f" [CITY] {city}"
        if country:
            query += f" [COUNTRY] {country}"
        
        if self.debug:
            logger.info(f"\n\nQuery: {query}")
                    
        # Include data source in cache key
        cache_key = f"{self.data_source}:{query}"
                    
        # Check cache first
        cached_results, is_reranked = self._get_predictions_from_cache(cache_key, num_candidates)
        if cached_results is not None:
            return cached_results
                    
        # Encode query
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        query_embedding_normalized = util.normalize_embeddings(query_embedding.unsqueeze(0))

        # Search for similar embeddings
        if self.use_hnsw and self.hnsw_index:
            # Use HNSW for faster search
            top_scores, top_indices = self.search_with_hnsw(query_embedding_normalized, NUM_CANDIDATES_TO_RETURN)
        else:
            # Use PyTorch for search (default)
            similarity_scores = util.pytorch_cos_sim(query_embedding_normalized, self.org_embeddings)[0]
            top_scores, top_indices = torch.topk(similarity_scores, min(NUM_CANDIDATES_TO_RETURN, len(self.org_embeddings)))
            top_scores = top_scores.cpu().tolist()
            top_indices = top_indices.cpu().tolist()
        
        # For each candidate in the results
        candidates = []
        
        if top_indices:
            for score, idx in zip(top_scores, top_indices):
                if score >= self.threshold_score:
                    # Get the organization info
                    org_info = self.org_data[idx].copy()  # Copy to avoid modifying original
                    
                    # Add additional fields for entity linking
                    org_info["enc_score"] = float(score)
                    org_info["orig_score"] = float(score)
                    org_info["source"] = "dense"
                    
                    # Add explanation
                    org_info["explanation"] = f"Matched via dense retrieval with score {score:.4f}"
                    
                    # Store the original query information
                    org_info["query_org"] = organization
                    
                    candidates.append(org_info)
                    
                    # Stop once we have enough candidates
                    if len(candidates) >= num_candidates:
                        break
        
        # Update cache with data source in key
        if candidates:
            self._update_predictions_cache(cache_key, candidates, reranked=False)
            return candidates
        else:
            # Return value if no results are retrieved
            no_results = [{
                "id": None, 
                "name": None, 
                "enc_score": 0.0, 
                "orig_score": 0.0, 
                "source": "dense", 
                "data_source": self.data_source,  # Include data source
                "explanation": "No matches found",
                "query_org": organization
            }]
            self._update_predictions_cache(cache_key, no_results, reranked=False)
            return no_results
