import os
import re
import sys
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor
from .data_manager import DataManager
from .constants import *
from .utils.text_utils import *
                
# Make sure that S2AFF is in the path to avoid changing the code in S2AFF implementations.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH)))

def convert(obj):
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert(i) for i in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

class EntityLinker:
    """Main entity linker class that coordinates multiple retrieval methods and reranking."""

    def __init__(
            self,
            linkers=[],
            linker_threshold=0.30,
            final_threshold=0.70,
            rerank=True,
            reranker=None,
            return_scores=True,
            num_candidates_to_return=NUM_CANDIDATES_TO_RETURN,
            num_candidates_to_rerank=NUM_CANDIDATES_TO_RERANK,
            detailed_results=False,
            verbose=False,
            use_cache=True,
            output_dir=OUTPUT_PARTIAL_CHUNKS,
            batch_size=32,
            data_sources=["ror"],
            use_wikidata_labels_with_ror=False,
            wikidata_org_types=None,
            wikidata_countries=None,
            data_source_configs=None):

        """
        Initialize with a list of linker instances.
        
        Args:
            linkers: List of linker instances or class names (strings).
            linker_threshold: Threshold for linker candidates
            final_threshold: Threshold for final results after reranking
            rerank: Whether to apply reranking
            reranker: Optional specific reranker to use
            return_scores: Whether to include scores in the results
            num_candidates_to_return: Number of candidates to return in the final results
            num_candidates_to_rerank: Number of candidates to consider for reranking
            detailed_results: Whether to include detailed results
            verbose: Whether to show verbose logging
            output_dir: Directory for partial output chunks
            batch_size: Batch size for processing
            data_sources: List of data sources to use (e.g., ["ror", "wikidata", ...])
            use_wikidata_labels_with_ror: Whether to enrich ROR organizations with WikiData labels
            wikidata_org_types: Organization types to include for WikiData (None for all)
            wikidata_countries: Countries to include for WikiData (None for all)
            data_source_configs: Dictionary of configurations for each data source
        """
        self.return_scores = return_scores
        self.verbose = verbose
        self.detailed_results = detailed_results
        self.use_cache = use_cache
        
        self.rerank = rerank
        self.reranker = reranker  # Optional custom reranker
        self.linker_threshold = linker_threshold if rerank else final_threshold
        self.final_threshold = final_threshold
        self.batch_size = batch_size
        
        self.num_candidates_to_return = num_candidates_to_return
        self.num_candidates_to_rerank = num_candidates_to_rerank
        
        # Set up output directory
        self.output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
        os.makedirs(self.output_path, exist_ok=True)
        
        # Store data sources
        self.data_sources = data_sources if isinstance(data_sources, list) else [data_sources]
        
        # Store configurations
        self.data_source_configs = data_source_configs or {}
        
        self.use_wikidata_labels_with_ror = use_wikidata_labels_with_ror
        
        # For backward compatibility, store WikiData parameters directly
        self.wikidata_org_types = wikidata_org_types
        self.wikidata_countries = wikidata_countries
        
        # Add WikiData configuration if not already present
        if "wikidata" in self.data_sources and "wikidata" not in self.data_source_configs:
            self.data_source_configs["wikidata"] = {
                "org_types": wikidata_org_types,
                "countries": wikidata_countries
            }
        
        if self.verbose:
            print(f"Initializing EntityLinker with data sources: {self.data_sources}")
            for source in self.data_sources:
                if source in self.data_source_configs:
                    print(f"  {source} configuration: {self.data_source_configs[source]}")
        
        # Initialize data manager
        self.data_manager = DataManager(verbose=self.verbose)
        
        # Initialize linkers and reranker
        self._initialize_linkers(linkers)
        self._initialize_reranker()
                    
    def _initialize_reranker(self):
        """Initialize the reranker component."""
        if not self.rerank:
            self.reranker = None
            return
            
        if self.reranker is not None:
            # Custom reranker was provided
            if self.verbose:
                print(f"Using provided reranker instance: {self.reranker.__class__.__name__}")
            return
        
        try:
            from .direct_pair_reranker import DirectPairReranker
            
            # Create a dictionary mapping data sources to their metadata paths
            hnsw_metadata_paths = {}
            for source in self.data_sources:
                # Get the path used by the DenseLinker for this source
                if source in self.linkers and 'Dense' in self.linkers[source]:
                    dense_linker = self.linkers[source]['Dense']
                    if hasattr(dense_linker, 'hnsw_index') and dense_linker.hnsw_metadata:
                        # Get the path from the dense linker's metadata
                        metadata_path = os.path.join(os.path.dirname(dense_linker.hnsw_index_path), "org_index_meta.json")
                        hnsw_metadata_paths[source] = metadata_path
            
            self.reranker = DirectPairReranker(
                debug=self.verbose, 
                batch_size=self.batch_size,
                use_cache=self.use_cache,
                supported_data_sources=self.data_sources,
                hnsw_metadata_path=hnsw_metadata_paths
            )
            
            if self.verbose:
                print(f"Created DirectPairReranker instance with supported data sources: {self.data_sources}")
        except Exception as e:
            self.reranker = None
            if self.verbose:
                print(f"Error creating reranker: {e}")
                    
    def _initialize_linkers(self, linkers):
        """
        Initialize linkers from instances or class names for each data source.
        
        Creates a nested dictionary:
        self.linkers = {
            'ror': {
                'linker_name': instance,
                ...
            },
            'wikidata': {
                'linker_name': instance,
                ...
            },
            'plugin_source': {
                'linker_name': instance,
                ...
            }
        }
        """
        self.linkers = {source: {} for source in self.data_sources}
        
        for data_source in self.data_sources:
            if self.verbose:
                print(f"Initializing linkers for data source: {data_source}")
                
            # Get configuration for this data source
            source_config = self.data_source_configs.get(data_source, {})
            source_config['verbose'] = self.verbose  # Add verbose flag to config
            
            # Prepare source-specific parameters
            source_params = {}
            
            # Fallback handling for built-in sources
            if data_source == "wikidata":
                source_params["org_types"] = self.wikidata_org_types
                source_params["countries"] = self.wikidata_countries
            elif data_source == "ror":
                source_params["use_wikidata_labels_with_ror"] = self.use_wikidata_labels_with_ror
                               
            # Initialize each linker
            for linker in linkers:
                if self.verbose:
                    print(f"  Initializing linker: {linker}")
                    
                if isinstance(linker, str):
                    # Initialize from class name
                    if linker == 'S2AFF':
                        from .s2aff_linker import S2AFFLinker
                        self.linkers[data_source][linker] = S2AFFLinker(
                                                        debug=self.verbose,
                                                        data_manager=self.data_manager,
                                                        data_source=data_source)
                    elif linker == 'Whoosh':
                        from .whoosh_linker import WhooshLinker
                        self.linkers[data_source][linker] = WhooshLinker(
                                                        debug=self.verbose,
                                                        use_cache=self.use_cache,
                                                        data_manager=self.data_manager,
                                                        threshold_score=self.linker_threshold,
                                                        return_num_candidates=self.num_candidates_to_return if not self.rerank else self.num_candidates_to_rerank,
                                                        data_source=data_source,
                                                        **source_params)
                    elif linker == 'Dense':
                        from .dense_linker import DenseLinker
                        self.linkers[data_source][linker] = DenseLinker(
                                                        debug=self.verbose,
                                                        use_cache=self.use_cache,
                                                        data_manager=self.data_manager,
                                                        batch_size=self.batch_size,
                                                        threshold_score=self.linker_threshold,
                                                        return_num_candidates=self.num_candidates_to_return if not self.rerank else self.num_candidates_to_rerank,
                                                        data_source=data_source,
                                                        **source_params)
                    else:
                        raise ValueError(f"Unknown linker type: {linker}")
                else:
                    # Already an instance - check if it supports data_source
                    linker_name = linker.__class__.__name__
                    if hasattr(linker, 'data_source'):
                        # Check if the linker is already configured for a specific data source
                        if linker.data_source != data_source:
                            if self.verbose:
                                print(f"Skipping {linker_name} for {data_source} as it's configured for {linker.data_source}")
                            continue
                    
                    self.linkers[data_source][linker_name] = linker
                    
                    # Set data_manager if not already set
                    if hasattr(linker, 'data_manager') and linker.data_manager is None:
                        linker.data_manager = self.data_manager
                        
                    # Set debug flag
                    if hasattr(linker, 'debug'):
                        linker.debug = self.verbose
                        
                    # For WikiData sources, update cache parameters if needed
                    if data_source == "wikidata" and hasattr(linker, 'wikidata_countries') and hasattr(linker, 'wikidata_org_types'):
                        linker.wikidata_countries = self.wikidata_countries
                        linker.wikidata_org_types = self.wikidata_org_types
    
    def process_in_chunks(self, entities, output_dir=None, chunk_size=None):
        """
        Process entities in chunks with parallelization.

        Args:
            entities: List of entity data to process
            output_dir: Directory for saving chunk results
            chunk_size: Number of entities per chunk

        Returns:
            List of processed results with combined entity_linking field
        """
        
        # Divide input into chunks
        chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE_EL
        chunks = [entities[i:i + chunk_size] for i in range(0, len(entities), chunk_size)]

        results_by_source = {}

        # Process each data source separately
        for data_source in self.data_sources:
            if self.verbose:
                print(f"\nProcessing data source: {data_source}")
                
            # Skip sources with no linkers
            if not self.linkers.get(data_source, {}):
                if self.verbose:
                    print(f"No linkers configured for data source: {data_source}")
                continue
                
            # Process chunks in parallel and collect results
            source_results = []
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EL) as executor:
                futures = [
                    executor.submit(self._process_chunk, chunk, data_source)
                    for chunk in chunks
                ]

                for idx, future in enumerate(futures):
                    chunk_result = future.result()
                    source_results.extend(chunk_result)

                    # Save intermediate results if required
                    if SAVE_CHUNKS_EL:
                        output_file = os.path.join(self.output_path, f"{data_source}_chunk_{idx}.json")
                        with open(output_file, "w") as f:
                            json.dump(json_serializable(chunk_result), f)
            
            # Store results for this data source
            results_by_source[data_source] = source_results
        
        # Combine results from all data sources
        final_results = self._combine_results(results_by_source)
        
        return final_results

    def _combine_results(self, results_by_source):
        """
        Combine results from different data sources into a unified structure.
        
        Args:
            results_by_source: Dictionary of data_source -> results
            
        Returns:
            Combined results with entity_linking structure
        """
        if not results_by_source:
            return []
            
        # Get any data source to determine the basic structure
        first_source = next(iter(results_by_source.values()))
        if not first_source:
            return []
            
        # Create basic structure from the first source
        combined_results = []
        for item in first_source:
            combined_item = {
                "raw_text": item["raw_text"],
                "span_entities": item["span_entities"],
                "ner": item.get("ner", []),
                "ner_raw": item.get("ner_raw", []),
                "osm": item.get("osm", []),
                "entity_linking": {},  # Will contain data source-specific results
                "language_info": item.get("language_info", {})
            }
            combined_results.append(combined_item)
        
        # Fill in entity_linking for each data source
        for i, base_item in enumerate(combined_results):
            # Create entity_linking section for each source
            for data_source, source_results in results_by_source.items():
                if i >= len(source_results):
                    continue
                    
                source_item = source_results[i]
                
                # Create structure for this data source
                source_data = {
                    "linked_orgs_spans": source_item.get("span_entity_links", [])
                }
                
                # Add "linked_orgs" consolidated field - join all spans with pipe separator
                source_data["linked_orgs"] = "|".join(filter(None, source_data["linked_orgs_spans"]))
                
                # Add detailed_orgs if present
                if self.detailed_results and "detailed_orgs" in source_item:
                    source_data["detailed_orgs"] = source_item["detailed_orgs"]
                
                # Store in the entity_linking structure
                base_item["entity_linking"][data_source] = source_data
        
        return combined_results

    def _process_chunk(self, chunk, data_source):
        """
        Process a single chunk for a specific data source through the entity linking pipeline.
        
        Args:
            chunk: A chunk of entity data
            data_source: Data source to process ("ror", "wikidata", etc.)
            
        Returns:
            List of processed entities with linking results
        """
        source_linkers = self.linkers.get(data_source, {})
        if not source_linkers:
            if self.verbose:
                print(f"No linkers available for data source: {data_source}")
            return chunk  # Return original chunk if no linkers

        # 1. Run all linkers in parallel to get candidates
        linker_results = self._run_all_linkers(chunk, source_linkers)
        
        # 2. Merge candidates from all linkers
        merged_results = self._merge_linker_results(linker_results, chunk)
        
        # 3. Apply reranking if requested and available
        if self.rerank and self.reranker:
            final_results = self._apply_reranking(merged_results, data_source)
        else:
            if self.verbose:
                print(f"\n!!! Processing entity linking without reranking step !!!\n")
            final_results = self._format_without_reranking(merged_results, data_source)
        
        # 4. Build detailed_orgs structure if requested
        if self.detailed_results:
            final_results = self._add_detailed_organization_results(final_results)
        
        # 5. Remove the original "org_candidates" field
        for result in final_results:
            if "org_candidates" in result:
                del result["org_candidates"]
        
        return final_results
        
    def _run_all_linkers(self, chunk, linkers):
        """
        Run all entity linkers on the chunk and collect their results.
        
        Args:
            chunk: A chunk of entity data
            linkers: Dictionary of linker name -> linker instance
            
        Returns:
            Dictionary of linker name -> linker results
        """
        linker_results = {}
        
        for name, linker in linkers.items():
            if self.verbose:
                print(f"Processing with {name}")
                
            # Process chunk with this linker
            results = linker.process_chunk_el(chunk, return_scores=True)
            linker_results[name] = results
            
        return linker_results
    
    def _merge_linker_results(self, linker_results, original_chunk):
        """
        Merge results from multiple linkers.
        
        Args:
            linker_results: Dictionary of linker name -> linker results
            original_chunk: Original chunk data for reference
            
        Returns:
            List of merged results
        """
        # If no results or only one linker, just return the results
        if not linker_results:
            return []
        
        if len(linker_results) == 1:
            # Just one linker, but still prepare the expected structure
            linker_name = next(iter(linker_results.keys()))
            results = linker_results[linker_name]
            
            # Make sure each result has the org_candidates field properly labeled with source
            for item in results:
                self._add_source_to_candidates(item, linker_name)
            
            return results
        
        # Multiple linkers - need to merge results
        merged_results = []
        
        # Get the number of items in the chunk (should be the same for all linkers)
        num_items = len(next(iter(linker_results.values())))
        
        for idx in range(num_items):
            # Start with common structure from the original chunk
            merged_item = {
                "raw_text": original_chunk[idx]["raw_text"],
                "span_entities": original_chunk[idx]["span_entities"],
                "ner": original_chunk[idx].get("ner", []),
                "ner_raw": original_chunk[idx].get("ner_raw", []),
                "osm": original_chunk[idx].get("osm", []),
                "span_entity_links": []
            }
            
            # Collect org_candidates from all linkers
            all_candidates = []
            
            for linker_name, results in linker_results.items():
                item_candidates = results[idx].get("org_candidates", [])
                
                # Add source information to these candidates
                for span_idx, span_candidates in enumerate(item_candidates):
                    labeled_span_candidates = []
                    for mention, candidates in span_candidates:
                        for candidate in candidates:
                            if "source" not in candidate:
                                candidate["source"] = linker_name
                        labeled_span_candidates.append((mention, candidates))
                    
                    # Replace with labeled candidates
                    item_candidates[span_idx] = labeled_span_candidates
                
                all_candidates.append(item_candidates)
                               
                # Collect span_entity_links (used to be ror_spans)
                if "span_entity_links" in results[idx]:
                    # Ensure merged_item has enough elements
                    while len(merged_item["span_entity_links"]) < len(results[idx]["span_entity_links"]):
                        merged_item["span_entity_links"].append("")
                        
                    # Merge span_entity_links, preferring non-empty values
                    for span_idx, span_link in enumerate(results[idx]["span_entity_links"]):
                        if span_idx < len(merged_item["span_entity_links"]):
                            if not merged_item["span_entity_links"][span_idx] and span_link:
                                merged_item["span_entity_links"][span_idx] = span_link
            
            # Merge all candidates into one structure
            merged_item["org_candidates"] = self._merge_candidates(all_candidates)
            
            # Add to results
            merged_results.append(merged_item)
        
        return merged_results
    
    def _add_source_to_candidates(self, item, source):
        """
        Add source information to all candidates in an item.
        
        Args:
            item: Result item containing candidates
            source: Source linker name
        """
        if "org_candidates" not in item:
            return
        
        for span_idx, span_candidates in enumerate(item["org_candidates"]):
            for candidate_idx, (mention, candidates) in enumerate(span_candidates):
                for candidate in candidates:
                    if "source" not in candidate:
                        candidate["source"] = source
    
    def _merge_candidates(self, candidates_list):
        """
        Merge organization candidates from multiple linkers.
        
        Args:
            candidates_list: List of org_candidates arrays from different linkers
            
        Returns:
            Merged org_candidates array
        """
        if not candidates_list:
            return []
        
        # Find the maximum number of spans
        max_spans = max([len(oc) for oc in candidates_list if oc], default=0)
        
        # Create merged structure
        merged_candidates = []
        
        # Process each span position
        for span_idx in range(max_spans):
            # Collect all candidates for this span
            span_mention_candidates = {}
            
            # Process each linker's candidates for this span
            for linker_candidates in candidates_list:
                # Skip if this linker doesn't have this span
                if span_idx >= len(linker_candidates):
                    continue
                
                # Process this linker's candidates for this span
                for mention, candidates in linker_candidates[span_idx]:
                    if mention not in span_mention_candidates:
                        span_mention_candidates[mention] = []
                    
                    # Add candidates, tracking all sources
                    for candidate in candidates:
                        # Skip invalid candidates
                        if not candidate.get("id"):
                            continue
                        
                        entity_id = candidate["id"]
                        source = candidate.get("source", "unknown")
                        
                        # Look for existing candidate with this ID
                        existing_candidate = next(
                            (c for c in span_mention_candidates[mention] if c.get("id") == entity_id),
                            None
                        )
                        
                        if existing_candidate:
                            # Update existing candidate
                            if "sources" not in existing_candidate:
                                existing_candidate["sources"] = [existing_candidate.get("source", "unknown")]
                            
                            if source not in existing_candidate["sources"]:
                                existing_candidate["sources"].append(source)
                            
                            # Update score if better
                            if candidate.get("enc_score", 0) > existing_candidate.get("enc_score", 0):
                                existing_candidate["enc_score"] = candidate["enc_score"]
                                
                                # Update other fields if needed
                                if "name" in candidate:
                                    existing_candidate["name"] = candidate["name"]
                                if "text" in candidate:
                                    existing_candidate["text"] = candidate["text"]
                        else:
                            # New candidate - make a copy to avoid modifying original
                            new_candidate = candidate.copy()
                            
                            # Add sources list
                            if "sources" not in new_candidate:
                                new_candidate["sources"] = [source]
                                
                            span_mention_candidates[mention].append(new_candidate)
            
            # Create the final span candidates structure
            span_candidates = []
            for mention, candidates in span_mention_candidates.items():
                # Sort candidates by score
                sorted_candidates = sorted(
                    candidates, 
                    key=lambda c: c.get("enc_score", 0), 
                    reverse=True
                )
                span_candidates.append((mention, sorted_candidates))
            
            merged_candidates.append(span_candidates)
        
        return merged_candidates

    def _should_skip_reranking(self, query_organization, candidates, 
                           absolute_threshold=0.90, 
                           relative_threshold=0.80, 
                           gap_threshold=0.10):
        """
        Determine if reranking can be skipped based on encoder scores.
        
        Args:
            query_organization: The organization being queried
            candidates: List of candidate organizations with encoder scores
            absolute_threshold: Minimum score for a clear match
            relative_threshold: Threshold for a good match if there's a gap
            gap_threshold: Required gap between top and second scores
            
        Returns:
            bool: True if reranking can be skipped, False otherwise
        """
        if not candidates or len(candidates) <= 1:
            return True
              
        # Determine how many candidates to examine
        top_n = min(self.num_candidates_to_rerank, len(candidates))
            
        # Check if all examined candidates have encoder scores
        if not all("enc_score" in candidate for candidate in candidates[:top_n]):
            return False
        
        # Get the scores of the top N candidates
        top_scores = [candidate["enc_score"] for candidate in candidates[:top_n]]
        
        # Check if the query contains non-Latin characters
        query_contains_non_latin_chars = False
        main_org = query_organization.get('main', '')
        if main_org:
            query_contains_non_latin_chars = contains_non_latin(main_org)
        
        # Adjust thresholds for non-Latin characters
        if query_contains_non_latin_chars:
            absolute_threshold -= 0.20  
            relative_threshold -= 0.20
            gap_threshold -= 0.10
        
        # If we are returning several candidates
        if self.num_candidates_to_return > 1:
            # If all scores are within a small range and all are high, reranking might not change much
            if (min(top_scores) > relative_threshold and 
                (top_scores[0] - top_scores[-1]) < gap_threshold/2):
                return True
                
        # If we are returning only the top candidate
        else:
            # First candidate is extremely confident
            if top_scores[0] > absolute_threshold:
                return True
                
            # Clear separation between first and rest
            if len(top_scores) > 1 and top_scores[0] > relative_threshold and (top_scores[0] - top_scores[1]) > gap_threshold:
                return True
            
        # Default: Reranking is recommended
        return False
           
    def _format_without_reranking(self, results, data_source):
        """
        Format results without applying reranking.
        For sub-organizations, include second-best candidate only if the top candidate
        is already included from another mention.
        
        Args:
            results: Results to format
            data_source: Data source to use
            
        Returns:
            Formatted results
        """
        for item in results:
            if 'org_candidates' not in item:
                continue
                    
            # Initialize span_entity_links field if needed
            if "span_entity_links" not in item:
                item["span_entity_links"] = []
                
            # Process each span
            for span_idx, span_candidates in enumerate(item['org_candidates']):
                # Ensure ror list has enough elements
                while len(item["span_entity_links"]) <= span_idx:
                    item["span_entity_links"].append("")  # Initialize as empty string
                
                span_entity_entries = []
                
                # FIRST PASS: Identify potential candidates and sub-organizations
                mention_candidates = {}  # {mention_idx: (mention, candidates, is_suborg)}
                # Track best scores by ROR ID - but only from top/second candidates
                id_best_scores = {}  # {id: (score, formatted_entry)}
                
                for mention_idx, (mention, candidates) in enumerate(span_candidates):
                    # Skip empty candidates
                    if not candidates:
                        continue
                        
                    # Get query organization (from first candidate)
                    query_org = candidates[0].get('query_org') if candidates else None
                    if not query_org:
                        continue
                    
                    # Check if this is a sub-organization
                    is_suborg = bool(query_org.get('suborg', ''))
                    
                    # Store candidates with metadata for the second pass
                    mention_candidates[mention_idx] = (mention, candidates, is_suborg, query_org)
                    
                    # Track best scores for top candidates only
                    for i, candidate in enumerate(candidates[:2]):  # Only consider top 2 candidates
                        entity_id = candidate.get("id")
                        if not entity_id:
                            continue
                            
                        score = candidate.get("enc_score", 0.0)
                        
                        # Update best score for this entity ID if it's better
                        if entity_id not in id_best_scores or score > id_best_scores[entity_id][0]:
                            id_best_scores[entity_id] = (score, candidate)
                
                # SECOND PASS: Collect top candidates from all mentions
                seen_ids = set()  # Track seen entity IDs to avoid duplicates
                top_candidates_ids = set()  # Track entity IDs of top candidates
                
                # First add top candidates from all mentions
                for mention_idx, (mention, candidates, is_suborg, query_org) in mention_candidates.items():
                    # Process just the top candidate for each mention
                    if not candidates or not candidates[0].get("id"):
                        continue
                        
                    top_candidate = candidates[0]
                    entity_id = top_candidate.get("id")
                    score = top_candidate.get("enc_score", 0.0)
                    
                    # Mark as a top candidate entity ID
                    top_candidates_ids.add(entity_id)
                    
                    # Adjust threshold based on script (non-Latin vs Latin)
                    adjusted_threshold = self.final_threshold
                    has_non_latin = False
                    if query_org.get('main', '') and contains_non_latin(query_org.get('main', '')):
                        adjusted_threshold -= 0.10
                        has_non_latin = True
                    
                    # Skip if below threshold
                    if score < adjusted_threshold:
                        if self.verbose:
                            threshold_explanation = f"below adjusted threshold {adjusted_threshold:.2f}" + (
                                " (non-Latin characters detected, original: {self.final_threshold:.2f})" if has_non_latin else "")
                            top_candidate["explanation"] = best_candidate["explanation"].split("(threshold")[0] + f" ({threshold_explanation})"
                            print(f"_format_without_reranking: Candidate {top_candidate.get('name', 'Unknown')} with score {score:.2f} {threshold_explanation}")
                        continue
                        
                    # Skip if already seen
                    if entity_id in seen_ids:
                        continue
                    
                    # Add to results using best score
                    seen_ids.add(entity_id)
                    best_score, best_candidate = id_best_scores[entity_id]
                    
                    # Update explanation with threshold info
                    if "explanation" in best_candidate:
                        threshold_info = f"threshold: {adjusted_threshold:.2f}"
                        if has_non_latin:
                            threshold_info += f", adjusted from {self.final_threshold:.2f} for non-Latin text"
                        best_candidate["explanation"] = best_candidate["explanation"].split("(threshold")[0] + f" ({threshold_info})"
                    
                    formatted_entity = self._format_entity_candidate(best_candidate, best_score, data_source)
                    if formatted_entity:
                        span_entity_entries.append(formatted_entity)
                
                # THIRD PASS: For sub-organizations, add second candidates if top candidate is already included
                for mention_idx, (mention, candidates, is_suborg, query_org) in mention_candidates.items():
                    # Only process sub-organizations with at least 2 candidates
                    if not is_suborg or len(candidates) < 2:
                        continue
                    
                    # Get the top candidate's entity ID
                    top_id = candidates[0].get("id")
                    if not top_id:
                        continue
                    
                    # Check if this entity ID appears in the top candidates of another mention
                    # AND has been included in the results
                    is_top_candidate_elsewhere = top_id in top_candidates_ids and top_id in seen_ids
                    
                    # Only add second candidate if the top one is already included elsewhere
                    if is_top_candidate_elsewhere:
                        second_candidate = candidates[1]
                        entity_id = second_candidate.get("id")
                        if not entity_id or entity_id in seen_ids:
                            continue
                            
                        score = second_candidate.get("enc_score", 0.0)
                        
                        # Adjust threshold based on script (non-Latin vs Latin)
                        adjusted_threshold = self.final_threshold
                        has_non_latin = False
                        if query_org.get('main', '') and contains_non_latin(query_org.get('main', '')):
                            adjusted_threshold -= 0.10
                            has_non_latin = True
                        
                        if score >= adjusted_threshold:
                            seen_ids.add(entity_id)
                            
                            # Use the best score version of this candidate
                            if entity_id in id_best_scores:
                                best_score, best_candidate = id_best_scores[entity_id]
                            else:
                                best_score = score
                                best_candidate = second_candidate
                            
                            # Update explanation with threshold and sub-org info
                            if "explanation" in best_candidate:
                                threshold_info = f"threshold: {adjusted_threshold:.2f}"
                                if has_non_latin:
                                    threshold_info += f", adjusted from {self.final_threshold:.2f} for non-Latin text"
                                best_candidate["explanation"] = best_candidate["explanation"].split("(threshold")[0] + f" ({threshold_info}, included as sub-org alternative)"
                            
                            formatted_entity = self._format_entity_candidate(best_candidate, best_score, data_source)
                                
                            if formatted_entity:
                                span_entity_entries.append(formatted_entity)
                                if self.verbose:
                                    print(f"_format_without_reranking: Adding {formatted_entity}")
                
                # Update entity_links field for this span - join with pipe separator
                item["span_entity_links"][span_idx] = "|".join(span_entity_entries)
        
        return results

    def _apply_reranking(self, results, data_source):
        """
        Apply reranking to results with threshold filtering.
        For sub-organizations, include second-best candidate only if the top candidate
        is already included from another mention.
        
        Args:
            results: The results to rerank
            data_source: The data source being processed
            
        Returns:
            The reranked results
        """
        if not self.reranker or not hasattr(self.reranker, 'rerank'):
            if self.verbose:
                print("_apply_reranking: Skipping reranking: No valid reranker available")
            return self._format_without_reranking(results, data_source)
        
        # Process each item
        for item in results:
        
            if 'org_candidates' not in item:
                continue
                    
            # Initialize span_entity_links field if needed
            if "span_entity_links" not in item:
                item["span_entity_links"] = []
                
            # Process each span
            for span_idx, span_candidates in enumerate(item['org_candidates']):
                # Ensure span_entity_links list has enough elements
                while len(item["span_entity_links"]) <= span_idx:
                    item["span_entity_links"].append("")  # Initialize as empty string
                
                span_entity_entries = []
                
                # FIRST PASS: Apply reranking and identify potential candidates
                # Store candidates keyed by mention index
                mention_candidates = {}  # {mention_idx: (mention, candidates, is_suborg)}
                # Track best scores by entity ID
                id_best_scores = {}  # {id: (score, formatted_entry)}
                
                for mention_idx, (mention, candidates) in enumerate(span_candidates):
                    # Skip empty or invalid candidates
                    if not candidates or not candidates[0].get("id"):
                        continue
                    
                    # Get query organization
                    query_org = candidates[0].get('query_org')
                    if not query_org:
                        continue
                        
                    # Check if this is a sub-organization
                    is_suborg = bool(query_org.get('suborg', ''))
                    
                    # Decide whether to skip reranking
                    skip_reranking = self._should_skip_reranking(query_org, candidates)
                    
                    # Process candidates for this mention
                    if skip_reranking:
                        if self.verbose:
                            print(f"_apply_reranking: Skipping reranking for '{mention}': High confidence in retrieval scores")
                        
                        # Use top candidate as is
                        reranked_candidates = candidates
                    else:
                        # Apply reranking
                        try:
                            reranked_candidates = self.reranker.rerank(
                                query_org,
                                candidates,
                                top_k=self.num_candidates_to_return,
                                return_scores=True,
                                data_source=data_source  # Pass data source to reranker
                            )
                            
                            # Add explanation
                            for candidate in reranked_candidates:
                                if "explanation" not in candidate:
                                    score = candidate.get("direct_score", 0.0)
                                    orig_score = candidate.get("enc_score", 0.0)
                                    candidate["explanation"] = f"Reranked from {orig_score:.4f} to {score:.4f} using direct pair matching"
                            
                            # Update the original candidates with reranked ones
                            span_candidates[mention_idx] = (mention, reranked_candidates)
                            
                        except Exception as e:
                            if self.verbose:
                                print(f"_apply_reranking: Error during reranking: {e}")
                            reranked_candidates = candidates  # Fall back to original candidates
                    
                    # Store candidates with metadata for the second pass
                    mention_candidates[mention_idx] = (mention, reranked_candidates, is_suborg, query_org)
                    
                    # Track best scores for top candidates only
                    for i, candidate in enumerate(reranked_candidates[:2]):  # Only consider top 2 candidates
                        entity_id = candidate.get("id")
                        if not entity_id:
                            continue
                            
                        score = candidate.get("direct_score", candidate.get("enc_score", 0.0))
                        
                        # Update best score for this entity ID if it's better
                        if entity_id not in id_best_scores or score > id_best_scores[entity_id][0]:
                            id_best_scores[entity_id] = (score, candidate)
                
                # SECOND PASS: Collect top candidates from all mentions
                seen_ids = set()  # Track seen entity IDs to avoid duplicates
                top_candidates_ids = set()  # Track entity IDs of top candidates
                
                # First add top candidates from all mentions that meet the threshold
                for mention_idx, (mention, candidates, is_suborg, query_org) in mention_candidates.items():
                    if not candidates:
                        continue
                        
                    # Get the top candidate
                    top_candidate = candidates[0]
                    entity_id = top_candidate.get("id")
                    if not entity_id:
                        continue
                        
                    # Mark as a top candidate entity ID
                    top_candidates_ids.add(entity_id)
                    
                    # Get score
                    score = top_candidate.get("direct_score", top_candidate.get("enc_score", 0.0))
                    
                    # Adjust threshold based on script (non-Latin vs Latin)
                    adjusted_threshold = self.final_threshold
                    has_non_latin = False
                    if query_org.get('main', '') and contains_non_latin(query_org.get('main', '')):
                        adjusted_threshold -= 0.10
                        has_non_latin = True
                    
                    # Skip if below threshold
                    if score < adjusted_threshold:
                        if self.verbose:
                            threshold_explanation = f"below adjusted threshold {adjusted_threshold:.2f}" + (
                                f" (non-Latin characters detected, original: {self.final_threshold:.2f})" if has_non_latin else "")
                            print(f"_apply_reranking: Candidate {top_candidate.get('name', 'Unknown')} with score {score:.2f} {threshold_explanation}")
                        continue

                    # Skip if already seen this entity ID
                    if entity_id in seen_ids:
                        continue
                    
                    # Add to results using best score
                    seen_ids.add(entity_id)
                    best_score, best_candidate = id_best_scores[entity_id]
                    
                    # Update explanation with threshold info
                    if "explanation" in best_candidate:
                        threshold_info = f"threshold: {adjusted_threshold:.2f}"
                        if has_non_latin:
                            threshold_info += f", adjusted from {self.final_threshold:.2f} for non-Latin text"
                        if "threshold" in best_candidate["explanation"]:
                            best_candidate["explanation"] = best_candidate["explanation"].split("(threshold")[0] + f" ({threshold_info})"
                        else:
                            best_candidate["explanation"] += f" ({threshold_info})"
                    
                    formatted_entity = self._format_entity_candidate(best_candidate, best_score, data_source)
                    if formatted_entity:
                        span_entity_entries.append(formatted_entity)
                        
                    if self.verbose:
                        print(f"\n_apply_reranking: second pass: added: {formatted_entity}. score {score} above threshold {adjusted_threshold}. best_score: {best_score}")
                
                # THIRD PASS: For sub-organizations, add second candidates if top candidate is already included
                for mention_idx, (mention, candidates, is_suborg, query_org) in mention_candidates.items():
                    # Only process sub-organizations with at least 2 candidates
                    if not is_suborg or len(candidates) < 2:
                        continue
                    
                    # Get the top candidate's entity ID
                    top_id = candidates[0].get("id")
                    if not top_id:
                        continue
                    
                    # Check if this entity ID appears in the top candidates of another mention
                    # AND has been included in the results
                    is_top_candidate_elsewhere = top_id in top_candidates_ids and top_id in seen_ids
                    
                    # Only add second candidate if the top one is already included elsewhere
                    if is_top_candidate_elsewhere:
                        second_candidate = candidates[1]
                        entity_id = second_candidate.get("id")
                        if not entity_id or entity_id in seen_ids:
                            continue
                            
                        score = second_candidate.get("direct_score", second_candidate.get("enc_score", 0.0))
                        
                        # Adjust threshold based on script (non-Latin vs Latin)
                        adjusted_threshold = self.final_threshold
                        has_non_latin = False
                        if query_org.get('main', '') and contains_non_latin(query_org.get('main', '')):
                            adjusted_threshold -= 0.10
                            has_non_latin = True
                        
                        if score >= adjusted_threshold:
                            seen_ids.add(entity_id)
                            
                            # Use the best score version of this candidate
                            if entity_id in id_best_scores:
                                best_score, best_candidate = id_best_scores[entity_id]
                            else:
                                best_score = score
                                best_candidate = second_candidate
                            
                            # Update explanation with threshold and sub-org info
                            if "explanation" in best_candidate:
                                threshold_info = f"threshold: {adjusted_threshold:.2f}"
                                if has_non_latin:
                                    threshold_info += f", adjusted from {self.final_threshold:.2f} for non-Latin text"
                                
                                if "threshold" in best_candidate["explanation"]:
                                    best_candidate["explanation"] = best_candidate["explanation"].split("(threshold")[0] + f" ({threshold_info}, included as sub-org alternative)"
                                else:
                                    best_candidate["explanation"] += f" ({threshold_info}, included as sub-org alternative)"
                                    
                            formatted_entity = self._format_entity_candidate(best_candidate, best_score, data_source)
                                
                            if formatted_entity:
                                span_entity_entries.append(formatted_entity)
                                if self.verbose:
                                    print(f"_apply_reranking: third pass: added {formatted_entity}")
                    
                # Update entity_links field for this span - join with pipe separator
                item["span_entity_links"][span_idx] = "|".join(span_entity_entries)
            
        return results


    def _format_entity_candidate(self, candidate, score, data_source):
        """
        Format a candidate for the entity field.
        
        Args:
            candidate: Candidate to format
            data_source: Data source being used
            
        Returns:
            Formatted entity entry string
        """
        # Skip invalid candidates
        if not candidate or not candidate.get("id"):
            return ""
            
        # Get basic information
        entity_id = candidate['id']
        name = candidate.get('name', '')
        
        # Use data manager to format URL if available
        if hasattr(self, 'data_manager') and self.data_manager and hasattr(self.data_manager, 'format_id_url'):
            id_url = self.data_manager.format_id_url(entity_id, data_source)
        else:
            # Fallback formatting based on data source
            if data_source == 'ror':
                id_url = f"https://ror.org/{entity_id}" if not entity_id.startswith('https://') else entity_id
            elif data_source == 'wikidata':
                id_url = f"https://www.wikidata.org/wiki/{entity_id}" if not entity_id.startswith('https://') else entity_id
            else:
                id_url = entity_id
        
        # Format based on whether scores should be included
        if self.return_scores:
            return f"{name} {{{id_url}}}:{score:.2f}"
        else:
            return f"{name} {{{id_url}}}"
    
    def _add_detailed_organization_results(self, results, show_num_candidates=NUM_CANDIDATES_TO_RETURN):
        """
        Add a detailed_orgs field to results with structured organization information.
        This provides essential information without the verbosity of org_candidates.
        
        Args:
            results: List of result dictionaries to augment
        
        Returns:
            Updated results list with detailed_orgs field
        """
        for result in results:

            if "org_candidates" not in result:
                continue
                
            detailed_orgs = []
            
            # Process each span
            for span_candidates in result["org_candidates"]:
                span_data = []
                
                # Process each mention in the span
                for mention, candidates in span_candidates:
                
                    # Create mention entry
                    mention_entry = {
                        "mention": mention
                    }
                
                    # If wanted reduce the number of candidates included in the output to show_num_candidates.
                    top_candidates = []
                    
                    for i, candidate in enumerate(candidates[:show_num_candidates]):
                        # Extract only the essential information
                        candidate_info = {
                            "rank": i + 1,
                            "id": candidate.get("id", ""),
                            "name": candidate.get("name", ""),
                        }
      
                        # Add location info if available
                        location_parts = []
                        if "city" in candidate and candidate["city"]:
                            location_parts.append(candidate["city"])
                        if "country" in candidate and candidate["country"]:
                            location_parts.append(candidate["country"])
                        if location_parts:
                            candidate_info["location"] = ", ".join(location_parts)

                        if "orig_score" in candidate:
                            candidate_info["initial_retriever_score"] = round(candidate["orig_score"], 4)
      
                        if "enc_score" in candidate:
                            candidate_info["final_retriever_score"] = round(candidate["enc_score"], 4)
      
                        if "direct_score" in candidate:
                            candidate_info["reranker_score"] = round(candidate["direct_score"], 4)
                        
                        if "explanation" in candidate:
                            candidate_info["explanation"] = candidate["explanation"]
                        
                        # Include source linker info
                        source = candidate.get("source", "")
                        if source:
                            candidate_info["source"] = source
                        
                        # Include data source
                        data_source = candidate.get("data_source", "")
                        if data_source:
                            candidate_info["data_source"] = data_source
                        
                        top_candidates.append(candidate_info)
                    
                    # Add candidates to mention entry
                    mention_entry["candidates_above_threshold"] = top_candidates
                    
                    span_data.append(mention_entry)
                
                # Add this span's data to detailed_orgs
                if span_data:
                    detailed_orgs.append(span_data)
            
            # Add detailed_orgs to the result
            if detailed_orgs:
                result["detailed_orgs"] = detailed_orgs
        
        return results
