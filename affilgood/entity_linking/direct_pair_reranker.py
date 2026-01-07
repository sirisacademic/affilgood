import os
import sys
import torch
import json
import re
import time
import pickle
from typing import List, Dict, Tuple, Optional, Set, Any
from tqdm import tqdm
from sentence_transformers import CrossEncoder, util
import contextlib
from .constants import *

# Configure the specific logger for this module
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:  # Only add handlers if none exist
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)  # Set to the desired level

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class DirectPairReranker:
    """
    Direct text-pair reranker for affiliation-organization matching.
    
    This reranker compares the raw affiliation string against multiple text
    representations of each candidate organization to find the best match.
    Includes direct AffilGood pipeline integration.
    """
    
    def __init__(
        self, 
        model_name: str = DIRECT_PAIR_RERANKER_DEFAULT_MODEL,
        batch_size: int = 32,
        max_length: int = 256,
        device: Optional[str] = None,
        hnsw_metadata_path: Optional[str] = None,
        reranking_strategy: str = "max_score",  # Options: "max_score", "avg_score", "weighted_avg"
        use_special_tokens: bool = False,
        use_cache: bool = True,  # Control caching
        cache_dir: Optional[str] = None,  # Directory for persistent cache
        cache_expiration: int = 604800,  # 7 days in seconds
        debug: bool = False,
        supported_data_sources: List[str] = ["ror"]
    ):
        """
        Initialize the direct pair reranker.
        
        Args:
            model_name: Name or path of the cross-encoder model
            batch_size: Batch size for inference
            max_length: Maximum sequence length for inputs
            device: Device to run inference on ('cpu', 'cuda', etc.)
            hnsw_metadata_path: Path to HNSW metadata file or dictionary mapping sources to paths
            reranking_strategy: Strategy for combining scores across variants
            use_special_tokens: Whether to format queries with special tokens
            use_cache: Whether to use caching for reranker results
            cache_dir: Directory to store the persistent cache
            cache_expiration: Cache expiration time in seconds
            debug: Enable debug output
            supported_data_sources: List of data sources this reranker should support
        """
        logger.info(f"Initializing DirectPairReranker with model: {model_name}")
        
        with suppress_output():
            self.model = CrossEncoder(
                model_name=model_name,
                max_length=max_length,
                device=device,
                trust_remote_code=True
            )
        
        self.batch_size = batch_size
        self.reranking_strategy = reranking_strategy
        self.model_name = model_name
        self.supported_data_sources = supported_data_sources if isinstance(supported_data_sources, list) else [supported_data_sources]
        
        # Cache configuration
        self.use_cache = use_cache
        self.reranker_name = self.__class__.__name__
        self.cache_expiration = cache_expiration
        self.reranking_cache = {}
        
        # Set the debug flag
        self.debug = debug
        
        # Log cache status
        if debug:
            logger.info(f"Cache {'enabled' if use_cache else 'disabled'} for {self.reranker_name}")
            logger.info(f"Supported data sources: {self.supported_data_sources}")
        
        # Set up cache directory for persistent storage
        self.cache_dir = cache_dir if cache_dir else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'reranker_cache'
        )
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Cache file path specific to this reranker and model
            model_name_safe = re.sub(r'[^\w\-\.]', '_', model_name)
            self.cache_file = os.path.join(self.cache_dir, f"{self.reranker_name}_{model_name_safe}_cache.pkl")
            
            # Load cached results from disk if available
            self._load_cache_from_disk()
        else:
            self.cache_file = None
            # Ensure cache is empty when disabled
            self.reranking_cache = {}
            if debug:
                logger.info(f"Caching disabled for {self.reranker_name}")
        
        # Initialize storage for organization metadata by data source
        self.org_info_by_source = {source: {} for source in self.supported_data_sources}
        self.embedding_to_org_mapping_by_source = {}
        self.hnsw_index_by_source = {}
        
        # Handle metadata and HNSW indices for each data source
        if isinstance(hnsw_metadata_path, dict):
            # Dictionary mapping data sources to metadata paths
            for source, path in hnsw_metadata_path.items():
                if source in self.supported_data_sources and os.path.exists(path):
                    self._load_organization_info(path, source)
        elif hnsw_metadata_path and os.path.exists(hnsw_metadata_path):
            # Single path provided - assume it's for the first supported source
            first_source = self.supported_data_sources[0]
            self._load_organization_info(hnsw_metadata_path, first_source)
            if debug:
                logger.info(f"Using single metadata path for source {first_source}: {hnsw_metadata_path}")
        else:
            # No path provided - try default paths for each source
            for source in self.supported_data_sources:
                default_path = os.path.join(HNSW_INDICES_PATH, source, "org_index_meta.json")
                if os.path.exists(default_path):
                    self._load_organization_info(default_path, source)
                    if debug:
                        logger.info(f"Using default metadata path for {source}: {default_path}")
                else:
                    # Try alternative path format
                    alt_path = os.path.join(HNSW_INDICES_PATH, "{source}_org_index_meta.json")
                    if os.path.exists(alt_path):
                        self._load_organization_info(alt_path, source)
                        if debug:
                            logger.info(f"Using alternative metadata path for {source}: {alt_path}")
                    else:
                        logger.warning(f"No metadata found for source {source}. Attempted paths {default_path} and {alt_path}")
        
        # Log loaded data sources
        if debug:
            loaded_sources = [source for source in self.supported_data_sources if source in self.org_info_by_source and self.org_info_by_source[source]]
            logger.info(f"Successfully loaded metadata for sources: {loaded_sources}")
        
        # Whether to format the strings with special tokens
        self.use_special_tokens = use_special_tokens
    
    def _load_cache_from_disk(self) -> None:
        """Load cached predictions from disk if caching is enabled."""
        if not self.use_cache:
            return
            
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    disk_cache = pickle.load(f)
                
                # Prune expired entries
                current_time = time.time()
                valid_cache = {}
                expired_count = 0
                
                for key, value in disk_cache.items():
                    # Check if entry has timestamp and hasn't expired
                    if 'timestamp' in value and (current_time - value['timestamp']) < self.cache_expiration:
                        valid_cache[key] = value
                    else:
                        expired_count += 1
                
                # Update in-memory cache with valid entries
                self.reranking_cache.update(valid_cache)
                
                if self.debug:
                    logger.info(f"Loaded {len(valid_cache)} cached predictions for {self.reranker_name}")
                    if expired_count > 0:
                        logger.info(f"Removed {expired_count} expired cache entries")
                        
            except Exception as e:
                logger.error(f"Error loading cache from disk: {e}")
                # If loading fails, start with an empty cache
                self.reranking_cache = {}
    
    def _save_cache_to_disk(self) -> None:
        """Save cached predictions to disk if caching is enabled."""
        if not self.use_cache or not self.cache_file:
            return
            
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.reranking_cache, f)
                
            if self.debug:
                logger.info(f"Saved {len(self.reranking_cache)} entries to cache file: {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache to disk: {e}")
    
    def _load_organization_info(self, metadata_path: str, data_source: str = "ror") -> None:
        """
        Load organization information from HNSW metadata file for a specific data source.
        
        Args:
            metadata_path: Path to HNSW metadata file
            data_source: Data source identifier
        """
        try:
            logger.info(f"Loading {data_source} organization information from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract organization data for this source
            if 'org_data' in metadata:
                for org in metadata['org_data']:
                    if 'id' in org:
                        entity_id = org['id']
                        self.org_info_by_source[data_source][entity_id] = org
                
                # Extract mapping from embedding index to organization index if available
                if 'embedding_to_org_mapping' in metadata:
                    self.embedding_to_org_mapping_by_source[data_source] = metadata['embedding_to_org_mapping']
                
                logger.info(f"Loaded information for {len(metadata['org_data'])} {data_source} organizations")
        except Exception as e:
            logger.error(f"Error loading {data_source} organization information: {e}")
    
    def _get_org_variants_in_index(self, entity_id: str, data_source: str = "ror") -> List[str]:
        """
        Get all text variants for an organization based on data source.
        
        Args:
            entity_id: Entity ID of the organization
            data_source: Data source to use
            
        Returns:
            List of text representations for the organization
        """
        variants = []
        
        # Get organization info from the appropriate source dictionary
        if data_source in self.org_info_by_source:
            org_info = self.org_info_by_source[data_source].get(entity_id, {})
            
            # If we have organization data, extract all representations
            if org_info:
                # Add canonical text representation
                if 'text' in org_info:
                    variants.append(org_info['text'])
                
                # Add text_representations array if available
                if 'text_representations' in org_info and isinstance(org_info['text_representations'], list):
                    variants.extend(org_info['text_representations'])
        
        # Remove duplicates while preserving order
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant not in seen and variant:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def _extract_name_from_text(self, text: str) -> str:
        """Extract the organization name from a text representation with special tokens."""
        if '[MENTION]' in text:
            # Extract text after [MENTION]
            parts = text.split('[MENTION]')
            if len(parts) > 1:
                name_part = parts[1].strip()
                # Find the next special token if any
                for token in ['[ACRONYM]', '[PARENT]', '[CITY]', '[COUNTRY]']:
                    if token in name_part:
                        name_part = name_part.split(token)[0].strip()
                return name_part
        return text  # Fallback to the original text
    
    def _get_raw_text(self, text: str) -> str:
        """Extract raw text by removing special tokens."""
        if self.use_special_tokens:
            return text
            
        # Replace special tokens with natural language
        text = text.replace("[MENTION]", "").strip()
        text = re.sub(r'\[ACRONYM\] ([^\s]+)', r'(\1)', text).strip()
        text = text.replace("[PARENT]", "part of").strip()
        text = text.replace("[CITY]", ",").strip()
        text = text.replace("[COUNTRY]", ",").strip()
        
        # Clean up any double spaces, multiple commas, etc.
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+,', ',', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r',\s*$', '', text)
        
        return text
    
    def _generate_org_variants(self, organization: Dict[str, Any]) -> List[str]:
        """
        Generate multiple variants from the organization information.
        
        Args:
            organization: Dictionary containing organization details
            
        Returns:
            List of text variants for the organization
        """
        variants = []
        
        # Get the components
        main_org = organization.get('main', '')
        suborg = organization.get('suborg', '')
        city = organization.get('city', '')
        country = organization.get('country', '')
        span_text = organization.get('span_text', '')
        
        # Always add the full span text as one variant
        if span_text:
            variants.append(span_text)
        
        # Add combinations of main organization with location
        if main_org:
            # Main org only
            variants.append(main_org)
            
            # Main org with city
            if city:
                variants.append(f"{main_org}, {city}")
            
            # Main org with country
            if country:
                variants.append(f"{main_org}, {country}")
            
            # Main org with city and country
            if city and country:
                variants.append(f"{main_org}, {city}, {country}")
        
        # Add combinations with sub-organization
        if suborg and main_org:
            # Suborg with main org
            variants.append(f"{suborg}, {main_org}")
            
            # Suborg, main org with location
            if city and country:
                variants.append(f"{suborg}, {main_org}, {city}, {country}")
            elif city:
                variants.append(f"{suborg}, {main_org}, {city}")
            elif country:
                variants.append(f"{suborg}, {main_org}, {country}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants

    def _get_org_variants_for_candidate(self, candidate: Dict[str, Any], data_source: str = "ror") -> List[str]:
        """
        Get all possible text variants for a candidate organization from data in index.
        
        Args:
            candidate: Dictionary containing candidate organization details
            data_source: Data source being used
            
        Returns:
            List of text variants for the organization
        """
        variants = []
        entity_id = candidate.get('id', '')
        
        # Add the text from the candidate if available
        if 'text' in candidate:
            variant = candidate['text']
            variants.append(variant)
            # Add without city
            variants.append(re.sub(r'\[CITY\] ([^\s]+)', r'\1', variant))
        
        # Get additional variants from the organization information
        # Use source-specific methods if needed
        if data_source == "ror":
            additional_variants = self._get_org_variants_in_index(entity_id, data_source)
        elif data_source == "wikidata":
            additional_variants = self._get_org_variants_in_index(entity_id, data_source)
        else:
            additional_variants = self._get_org_variants_in_index(entity_id, data_source)
        
        # Add any new variants not already in the list
        for variant in additional_variants:
            if variant not in variants:
                variants.append(variant)
                variants.append(re.sub(r'\[CITY\] ([^\s]+)', r'\1', variant))
        
        # If still no variants but have a name, create a simple representation
        if not variants and 'name' in candidate:
            name = candidate['name']
            variants.append(f"[MENTION] {name}")
            variants.append(name)  # Also add plain name
        
        return variants

    def set_cache_enabled(self, enabled=True):
        """
        Enable or disable caching.
        
        Args:
            enabled: True to enable caching, False to disable
            
        Returns:
            dict: Status of the operation
        """
        old_status = self.use_cache
        self.use_cache = enabled
        
        # If enabling and cache was previously disabled
        if enabled and not old_status:
            # Initialize cache directory and file
            os.makedirs(self.cache_dir, exist_ok=True)
            model_name_safe = re.sub(r'[^\w\-\.]', '_', self.model_name)
            self.cache_file = os.path.join(self.cache_dir, f"{self.reranker_name}_{model_name_safe}_cache.pkl")
            
            # Load existing cache if available
            self._load_cache_from_disk()
            
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Cache enabled for {self.reranker_name}")
                
            return {"status": "Cache enabled", "loaded_entries": len(self.reranking_cache)}
        
        # If disabling and cache was previously enabled
        elif not enabled and old_status:
            # Save cache one last time before disabling
            if self.cache_file:
                self._save_cache_to_disk()
                
            saved_entries = len(self.reranking_cache)
            
            # Clear in-memory cache
            self.reranking_cache = {}
            self.cache_file = None
            
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Cache disabled for {self.reranker_name}")
                
            return {"status": "Cache disabled", "saved_entries": saved_entries}
        
        # No change
        return {"status": "Cache already " + ("enabled" if enabled else "disabled")}

    def get_from_cache(self, linker_name: str, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get reranked results from cache using the comprehensive cache key.
        
        Args:
            linker_name: Name of the linker that provided the candidates
            cache_key: Key identifying the specific query (already contains data source)
            
        Returns:
            Cached results if available, otherwise None
        """
        if not self.use_cache:
            return None
            
        # With the new comprehensive cache key, we don't need to append the linker name
        # The cache_key already contains the necessary organization details, linker info, and data source
        
        if cache_key in self.reranking_cache:
            cached_entry = self.reranking_cache[cache_key]
            
            # Check if cache entry is expired
            if 'timestamp' in cached_entry:
                current_time = time.time()
                if (current_time - cached_entry['timestamp']) > self.cache_expiration:
                    # Cache entry expired
                    return None
                    
            if hasattr(self, 'debug') and self.debug:
                logger.info(f"Using cached reranking results for {cache_key}")
                
            return cached_entry
        
        return None

    def update_cache(self, linker_name: str, cache_key: str, reranked_results: List[Dict[str, Any]]) -> None:
        """
        Update the reranking cache using the comprehensive cache key.
        
        Args:
            linker_name: Name of the linker that provided the candidates (for compatibility, not used)
            cache_key: Key identifying the specific query (already contains all necessary information)
            reranked_results: Results to cache
        """
        if not self.use_cache:
            return
        
        # With the new cache key format, we don't need to append linker name
        # Just use the key directly
        
        self.reranking_cache[cache_key] = {
            'results': reranked_results,
            'timestamp': time.time()
        }
        
        # Save to disk periodically (every 10 cache updates)
        if len(self.reranking_cache) % 10 == 0:
            self._save_cache_to_disk()
        
        
    def clear_cache(self, expired_only: bool = True) -> Dict[str, Any]:
        """
        Clear the reranking cache.
        
        Args:
            expired_only: If True, only clear expired entries. If False, clear all entries.
            
        Returns:
            dict: Status of the operation
        """
        if not self.use_cache:
            return {"status": "Cache is disabled", "cleared_entries": 0}
            
        if expired_only:
            # Remove only expired entries
            current_time = time.time()
            keys_to_remove = []
            
            for key, value in self.reranking_cache.items():
                if 'timestamp' in value and (current_time - value['timestamp']) > self.cache_expiration:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.reranking_cache[key]
                
            # Save updated cache to disk
            self._save_cache_to_disk()
            
            return {
                "status": "Cleared expired entries",
                "cleared_entries": len(keys_to_remove)
            }
        else:
            # Clear all entries
            entry_count = len(self.reranking_cache)
            self.reranking_cache = {}
            
            # Save updated cache to disk (empty)
            self._save_cache_to_disk()
            
            return {
                "status": "Cleared all entries",
                "cleared_entries": entry_count
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics or message if caching is disabled
        """
        if not self.use_cache:
            return {"status": "Cache is disabled"}
            
        total_entries = len(self.reranking_cache)
        
        # Count entries by type
        entries_by_linker = {}
        for key in self.reranking_cache:
            linker_name = key.split(':')[0]
            entries_by_linker[linker_name] = entries_by_linker.get(linker_name, 0) + 1
        
        # Count expired entries
        current_time = time.time()
        expired_count = 0
        for key, value in self.reranking_cache.items():
            if 'timestamp' in value and (current_time - value['timestamp']) > self.cache_expiration:
                expired_count += 1
        
        # Cache file info
        cache_file_size = 0
        if os.path.exists(self.cache_file):
            cache_file_size = os.path.getsize(self.cache_file) / (1024 * 1024)  # Size in MB
            
        return {
            "total_entries": total_entries,
            "entries_by_linker": entries_by_linker,
            "expired_entries": expired_count,
            "cache_file": self.cache_file,
            "cache_file_size_mb": round(cache_file_size, 2),
            "expiration_days": self.cache_expiration / (60 * 60 * 24),
            "status": "Enabled"
        }

    def _generate_cache_key(self, organization: Dict[str, Any], source_linker: str, data_source: str = "ror", active_linkers: List[str] = None) -> str:
        """
        Generate a more comprehensive cache key for an organization.
        
        Args:
            organization: The query organization dictionary
            source_linker: Name of the source linker
            data_source: Data source being used
            active_linkers: List of currently active linkers
            
        Returns:
            A unique cache key string
        """
        # Extract all relevant fields that could make this organization unique
        main_org = organization.get('main', '')
        suborg = organization.get('suborg', '')
        city = organization.get('city', '')
        country = organization.get('country', '')
        region = organization.get('region', '')
        
        # Combine fields into a single key, removing any empty values
        org_parts = [p for p in [suborg, main_org, city, region, country] if p]
        org_key = '_'.join(org_parts)
        
        # Add source linker and data source to make the key even more specific
        source_parts = [p for p in [org_key, source_linker, data_source] if p]
        
        # Add active linkers to ensure cache isolation between different linker combinations
        if active_linkers:
            linkers_str = '+'.join(sorted(active_linkers))  # Sort for consistency
            source_parts.append(linkers_str)
        
        composite_key = '_'.join(source_parts)
        
        # Sanitize key to remove problematic characters
        sanitized_key = re.sub(r'[^\w\-\.\+]', '_', composite_key)
        
        return sanitized_key

    # Reranking function
    def rerank(
        self, 
        organization: Dict[str, Any],
        candidates: List[Dict[str, Any]], 
        top_k: Optional[int] = NUM_CANDIDATES_TO_RETURN,
        return_scores: bool = True,
        show_progress_bar: bool = False,
        return_best_variant: bool = True,
        data_source: str = "ror",
        active_linkers: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate organizations using multiple query variants.
        
        Args:
            organization: The query organization
            candidates: List of candidate organizations to rerank
            top_k: Number of top candidates to return
            return_scores: Whether to include scores in results
            show_progress_bar: Whether to show progress during scoring
            return_best_variant: Whether to include best matching variant info
            data_source: Data source being used ("ror", "wikidata", etc.)
            
        Returns:
            List of reranked candidates
        """
        if not candidates:
            return []
        
        # Double-check cache is actually enabled before trying to use it
        use_cache = getattr(self, 'use_cache', False)
        
        # Get source linker
        source_linker = candidates[0].get('source') if candidates else None
        
        # Generate comprehensive cache key that includes data source and active linkers
        cache_key = self._generate_cache_key(organization, source_linker, data_source, active_linkers)
        
        # Check cache if enabled and if cache exists
        if use_cache and hasattr(self, 'reranking_cache') and self.reranking_cache:
            cached_entry = self.get_from_cache(source_linker, cache_key)
            if cached_entry and 'results' in cached_entry:
                if hasattr(self, 'debug') and self.debug:
                    logger.info(f"Using cached reranking results for {cache_key}")
                return cached_entry['results']
        
        # Generate multiple variants for the organization to be linked
        query_variants = self._generate_org_variants(organization)
        if self.debug:
            logger.info("\n\nGenerated query variants:")
            for i, variant in enumerate(query_variants):
                logger.info(f"  {i+1}: {variant}")
        
        # Prepare for batch scoring
        all_pairs = []  # (query_variant, candidate_variant) pairs
        pair_metadata = []  # (candidate_idx, candidate_variant_idx, query_variant_idx)

        for candidate_idx, candidate in enumerate(candidates):
            # Get all variants for this candidate from the data in the index
            # Pass the data source to get appropriate variants
            candidate_variants = self._get_org_variants_for_candidate(candidate, data_source)
            
            # Create pairs for all query variants against all org variants
            for cv_idx, candidate_variant in enumerate(candidate_variants):
                # Process text based on special tokens setting
                processed_candidate_variant = self._get_raw_text(candidate_variant) if not self.use_special_tokens else candidate_variant
                for qv_idx, query_variant in enumerate(query_variants):
                    # Process query strings
                    processed_query = self._get_raw_text(query_variant) if not self.use_special_tokens else query_variant
                    all_pairs.append((processed_query, processed_candidate_variant))
                    pair_metadata.append((candidate_idx, cv_idx, qv_idx))
        
        if self.debug:
            logger.info("\n\nScoring pairs (sample):")
            for i, pair in enumerate(all_pairs[:min(10, len(all_pairs))]):
                logger.info(f"  {i+1}: {pair}")
            if len(all_pairs) > 5:
                logger.info(f"  ... and {len(all_pairs) - 5} more pairs")
        
        # Score all pairs
        if all_pairs:
            scores = self.model.predict(
                all_pairs,
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar and len(all_pairs) > 100
            )
        else:
            return []
        
        # Collect best scores for each candidate across all query-org variant combinations
        candidate_results = []
        for i, candidate in enumerate(candidates):
            # Find all scores for this candidate
            candidate_scores = []
            variant_combinations = []
            
            # Get all candidate variants for this candidate (may need regenerating)
            candidate_variants = self._get_org_variants_for_candidate(candidate, data_source)
            
            # Find scores for this candidate
            for j, (candidate_idx, cv_idx, qv_idx) in enumerate(pair_metadata):
                if candidate_idx == i:
                    score = float(scores[j])
                    query_variant = query_variants[qv_idx]
                    candidate_variant = candidate_variants[cv_idx] if cv_idx < len(candidate_variants) else ""
                    
                    # Process variant text for consistency
                    candidate_variant = self._get_raw_text(candidate_variant) if not self.use_special_tokens else candidate_variant
                    
                    candidate_scores.append(score)
                    variant_combinations.append((query_variant, candidate_variant, score))
            
            if not candidate_scores:
                continue
            
            # Find the best score and its corresponding combination
            best_score = max(candidate_scores)
            best_idx = candidate_scores.index(best_score)
            best_query, best_variant, _ = variant_combinations[best_idx]
            
            # Create result dictionary
            result = candidate.copy()
            result['direct_score'] = best_score
            
            # Preserve data source information
            result['data_source'] = data_source
            
            # Add explanation
            enc_score = candidate.get('enc_score', 0.0)
            result['explanation'] = f"Reranked from {enc_score:.4f} to {best_score:.4f} using direct pair matching"
            
            # Add the best matching combination information
            if return_best_variant:
                result['best_variant'] = best_variant
                result['best_query'] = best_query
                result['variant_combinations'] = variant_combinations
            
            # Extract a clean name if needed
            if 'name' not in result and best_variant:
                result['name'] = self._extract_name_from_text(best_variant)
            
            candidate_results.append(result)
        
        # Sort by score in descending order
        candidate_results = sorted(candidate_results, key=lambda x: x['direct_score'], reverse=True)
        if self.debug and len(candidate_results)>0:
            logger.info(f"===> Top match: {candidate_results[0]['best_query']} - {candidate_results[0]['best_variant']}. Score: {candidate_results[0]['direct_score']}\n")
      
        # Limit to top_k if specified
        if top_k is not None:
            candidate_results = candidate_results[:top_k]
        
        # Remove scores if not requested
        if not return_scores:
            for result in candidate_results:
                if 'direct_score' in result:
                    del result['direct_score']
                if 'variant_combinations' in result:
                    del result['variant_combinations']
        
        # Update cache with results if caching is enabled
        if use_cache:
            self.update_cache(source_linker, cache_key, candidate_results)
            # Save to disk after significant processing
            if hasattr(self, '_save_cache_to_disk'):
                self._save_cache_to_disk()
        
        return candidate_results
