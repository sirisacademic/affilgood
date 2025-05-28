import time
import os
import json
import pickle
from abc import ABC, abstractmethod
from .constants import *

class BaseLinker(ABC):
    """Base class for entity linkers with shared functionalities."""

    def __init__(self, use_cache=True, cache_dir=None, cache_expiration=604800, data_source="ror", 
                wikidata_countries=None, wikidata_org_types=None):
        """
        Initialize the base linker.
        
        Args:
            use_cache: Whether to use caching (both in-memory and persistent). Default is True.
            cache_dir: Directory to store the persistent cache. If None, defaults to 
                      a directory named 'linker_cache' in the current directory.
            cache_expiration: Cache expiration time in seconds. Default is 7 days.
            data_source: Data source to use ("ror", "wikidata", etc). Default is "ror".
            wikidata_countries: Countries to filter WikiData results (for cache key generation)
            wikidata_org_types: Organization types to filter WikiData results (for cache key generation)
        """
        self.cached_predictions = {}  # In-memory cache
        self.linker_name = self.__class__.__name__  # Store the linker class name
        self.is_initialized = False
        self.data_source = data_source  # Store the data source
        
        # Store WikiData filter parameters for cache key generation
        self.wikidata_countries = wikidata_countries
        self.wikidata_org_types = wikidata_org_types
        
        # Cache toggle
        self.use_cache = use_cache
        
        # Persistent cache configuration
        self.cache_dir = cache_dir if cache_dir else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'linker_cache'
        )
        self.cache_expiration = cache_expiration
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Generate a unique cache file path based on data source and WikiData filters
            if self.data_source == "wikidata" and (self.wikidata_countries or self.wikidata_org_types):
                # Create filter-specific cache file names for WikiData
                countries_str = self._format_filter_for_filename(self.wikidata_countries)
                org_types_str = self._format_filter_for_filename(self.wikidata_org_types)
                
                # Build a unique cache filename for this specific filter combination
                self.cache_file = os.path.join(
                    self.cache_dir,
                    f"{self.linker_name}_{self.data_source}_{countries_str}_{org_types_str}_cache.pkl"
                )
            else:
                # Standard cache file for ROR or unfiltered WikiData
                self.cache_file = os.path.join(
                    self.cache_dir,
                    f"{self.linker_name}_{self.data_source}_cache.pkl"
                )
            
            # Load cached predictions from disk if available
            self._load_cache_from_disk()
        else:
            self.cache_file = None
            if (hasattr(self, 'verbose') and self.verbose) or (hasattr(self, 'debug') and self.debug):
                print(f"Caching disabled for {self.linker_name} ({self.data_source})")
                
    def _format_filter_for_filename(self, filter_param):
        """Format a filter parameter for use in a filename."""
        if filter_param is None:
            return "all"
        elif isinstance(filter_param, list):
            if len(filter_param) == 0:
                return "none"
            elif len(filter_param) == 1:
                # Single item list, use the value
                return str(filter_param[0]).replace(" ", "_")
            else:
                # Multiple items, use a hash to keep filename reasonable
                import hashlib
                items_str = "_".join(sorted(str(item) for item in filter_param))
                return hashlib.md5(items_str.encode()).hexdigest()[:8]
        else:
            # Single value, just convert to string
            return str(filter_param).replace(" ", "_")

    def _load_cache_from_disk(self):
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
                self.cached_predictions.update(valid_cache)
                
                if (hasattr(self, 'verbose') and self.verbose) or (hasattr(self, 'debug') and self.debug):
                    print(f"Loaded {len(valid_cache)} cached predictions for {self.linker_name} ({self.data_source}) from file {self.cache_file}")
                    if expired_count > 0:
                        print(f"Removed {expired_count} expired cache entries")
                        
            except Exception as e:
                print(f"Error loading cache from disk: {e}")
                # If loading fails, start with an empty cache
                self.cached_predictions = {}
    
    def _save_cache_to_disk(self):
        """Save cached predictions to disk if caching is enabled."""
        if not self.use_cache or not self.cache_file:
            return
            
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cached_predictions, f)
                
            if (hasattr(self, 'verbose') and self.verbose) or (hasattr(self, 'debug') and self.debug):
                print(f"Saved {len(self.cached_predictions)} entries to cache file: {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache to disk: {e}")

    @abstractmethod
    def initialize(self):
        """Abstract method for initializing entity linking components."""
        pass  # This does nothing in the parent class but must be implemented in subclasses.

    @abstractmethod
    def get_single_prediction(self, organization, num_candidates):
        """Abstract method for getting predictions for one organization."""
        pass  # This does nothing in the parent class but must be implemented in subclasses.
   
    def _get_predictions_from_cache(self, cache_key, num_candidates=NUM_CANDIDATES_TO_RETURN):
        """Retrieve multiple predictions from the cache if caching is enabled."""
        if not self.use_cache:
            return None, False
            
        # Include linker name and data source in the cache key
        # We don't need to include WikiData filters here since they're already in the cache file name
        linker_specific_key = f"{self.linker_name}:{self.data_source}:{cache_key}"
        
        if linker_specific_key in self.cached_predictions:
            cached_entry = self.cached_predictions[linker_specific_key]
            predictions = cached_entry.get('predictions', [])
            reranked = cached_entry.get('reranked', False)
            
            # Check if cache entry is expired
            if 'timestamp' in cached_entry:
                current_time = time.time()
                if (current_time - cached_entry['timestamp']) > self.cache_expiration:
                    # Cache entry expired
                    return None, False
            
            # Check if this is the "no results" placeholder
            if predictions and predictions[0]['id'] is None:
                return predictions, reranked
                
            # If we have enough valid cached results or there's no way to get more
            if len(predictions) >= num_candidates:
                return predictions[:num_candidates], reranked
                
            # Not enough results in cache
            return None, False
        
        return None, False

    def _update_predictions_cache(self, cache_key, predictions, reranked=False):
        """Update cache with linker-specific key and save to disk if caching is enabled."""
        if not self.use_cache:
            return
            
        # Include linker name and data source in the cache key
        # WikiData filters are handled by using different cache files
        linker_specific_key = f"{self.linker_name}:{self.data_source}:{cache_key}"
        
        self.cached_predictions[linker_specific_key] = {
            'predictions': predictions,
            'reranked': reranked,
            'timestamp': time.time()
        }
        
        # Save to disk periodically (every 10 cache updates)
        if len(self.cached_predictions) % 10 == 0:
            self._save_cache_to_disk()
   
    def process_chunk_el(self, chunk, return_scores=True):
        """Process chunks using the specific entity linkers."""
        # Ensure initialization before processing
        if not self.is_initialized:
            self.initialize()
        
        chunk_to_process = []
        chunk_index_map = []  # Track the position in text_list and span_entities index
        
        # Prepare chunks for processing
        for idx, item in enumerate(chunk):
            raw_text = item.get("raw_text", '')
            span_entities = item.get("ner_raw", [])
            osm_entities = item.get("osm", [])
            spans = item.get("span_entities", [])  # Get the span texts
            
            if len(span_entities) > 0:
                for span_idx, (span, osm) in enumerate(zip(span_entities, osm_entities)):
                    # Get the corresponding span text if available
                    span_text = spans[span_idx] if span_idx < len(spans) else raw_text
                    chunk_to_process.append((span, osm, raw_text, span_text))
                    chunk_index_map.append((idx, span_idx))
            else:
                    chunk_to_process.append((raw_text, {}, raw_text, raw_text))
                    chunk_index_map.append((idx, 0))
        
        # Process chunks
        processed_list = []
        for ner, osm, raw_text, span_text in chunk_to_process:
            result = {}
            result['grouped_entities'] = self.get_entity_groupings(ner)
            result['el_input_organizations'] = self.get_el_input_organizations(
                result['grouped_entities'], 
                osm,
                span_text
            )
            
            # Get predictions
            predictions, scores, explanations, all_candidates = self.get_predicted_labels(result['el_input_organizations'])
            
            result[COL_PREDICTIONS_EL] = predictions
            result[COL_PREDICTIONS_SCORES_EL] = scores
            result['explanations'] = explanations
            result['org_candidates'] = all_candidates
            processed_list.append(result)
        
        # Format results with source-specific information
        results = [{
            "raw_text": item["raw_text"],
            "span_entities": item["span_entities"],
            "ner": item.get('ner', []),
            "ner_raw": item.get("ner_raw", []),
            "osm": item.get("osm", []),
            "span_entity_links": [], # Linked entities for each individual span
            "org_candidates": [],
            "data_source": self.data_source  # Include data source
        } for item in chunk]
        
        # Map results back to original structure
        for idx, _ in enumerate(chunk_to_process):
            entities = processed_list[idx]
            item_idx, ror_idx = chunk_index_map[idx]
            
            # Ensure correct list sizes
            while len(results[item_idx]["span_entity_links"]) <= ror_idx:
                results[item_idx]["span_entity_links"].append("")
                
            # Assign entity prediction
            results[item_idx]["span_entity_links"][ror_idx] = (
                entities[COL_PREDICTIONS_SCORES_EL] if return_scores else entities[COL_PREDICTIONS_EL]
            )
            
            # Assign all candidates retrieved
            results[item_idx]["org_candidates"].append(entities.get("org_candidates", []))
        
        # Save cache to disk after chunk processing if caching is enabled
        if self.use_cache:
            self._save_cache_to_disk()
        
        return results
    
    def extract_query_org_from_chunk_item(self, item):
        """
        Extract query organization information from a chunk item using
        the standard NER processing logic.
        
        Args:
            item: A chunk item containing NER data
            
        Returns:
            dict: Query organization dictionary or None if not extractable
        """
        if 'ner' not in item or not item['ner']:
            return None
            
        try:
            # Process the NER data using existing methods
            entity_groupings = self.get_entity_groupings(item['ner'])
            
            # Use the existing method to convert groupings to organizations
            osm = item.get('osm', [])
            span_text = item.get('span_entities', [''])[0] if 'span_entities' in item else ''
            organizations = self.get_el_input_organizations(entity_groupings, osm, span_text)
            
            # Return the first valid organization if any
            if organizations:
                return organizations[0]
            
            return None
        except Exception as e:
            print(f"Error extracting query organization: {e}")
            return None
    
    def get_predicted_labels(self, organizations, num_candidates=NUM_CANDIDATES_TO_RETURN):
        """Get predicted labels for organizations."""
        # Original dictionaries for backward compatibility
        predicted_names = {}
        predicted_scores = {}
        explanations = {}
        
        # New structure for the top-k candidates
        all_org_candidates = []  # List of (mention, [candidates]) tuples
        
        # Track which main_orgs were attempted and whether linked
        main_org_link_status = {}  # main_org → linked (True/False)

        for org in organizations:
            main_org = org.get("main", "").strip()
            if not main_org:
                continue
                    
            # Construct the mention text
            mention_text = main_org
            if org.get("suborg", "").strip():
                mention_text = f"{org['suborg']}, {main_org}"
            
            # Get multiple predictions
            candidates = self.get_single_prediction(org, num_candidates=num_candidates)
            
            # Track whether we got any valid prediction
            has_valid_link = False
            
            # Process highest-scoring candidate for backward compatibility
            if candidates and candidates[0].get("id"):  # If there's at least one valid candidate
                has_valid_link = True
                top_candidate = candidates[0]
                predicted_id = top_candidate["id"]
                predicted_names[predicted_id] = top_candidate.get("name", predicted_id)
                predicted_scores[predicted_id] = top_candidate.get("enc_score", 0.0)
                # If there is an explanation, add it.
                if "explanation" in top_candidate:
                    explanations[predicted_id] = top_candidate["explanation"]
            
            # Add all candidates to the new structure
            if candidates and candidates[0].get("id"):  # If there's at least one valid candidate
                all_org_candidates.append((mention_text, candidates))
            else:
                all_org_candidates.append((mention_text, []))
            
            # Update link status for this main organization
            main_org_link_status[main_org] = has_valid_link

        # Get URL formatting function from data manager if available
        format_id_url = None
        if hasattr(self, 'data_manager') and self.data_manager and hasattr(self.data_manager, 'format_id_url'):
            format_id_url = self.data_manager.format_id_url

        # Format single predictions based on data source
        predicted_names_ids = []
        for predicted_id in predicted_names:
            name = predicted_names[predicted_id]
            # Format URL for current data source
            if format_id_url:
                url = format_id_url(predicted_id, self.data_source)
            else:
                # Fallback to generic formatting
                url = predicted_id
            predicted_names_ids.append(f'{name} {{{url}}}')
        
        predicted_labels = '|'.join(predicted_names_ids)
        
        # Format scores
        predicted_labels_scores = []
        for pid in predicted_names:
            name = predicted_names[pid]
            # Format URL for current data source
            if format_id_url:
                url = format_id_url(pid, self.data_source)
            else:
                # Fallback to generic formatting
                url = pid
            score = predicted_scores[pid]
            predicted_labels_scores.append(f'{name} {{{url}}}:{score:.2f}')
        
        predicted_labels_scores = '|'.join(predicted_labels_scores)
        
        return predicted_labels, predicted_labels_scores, explanations, all_org_candidates

   
    def get_entity_groupings(self, entities, fallback=True):
        """Generate groupings from NER output with access to span text."""
        if type(entities)==str:
            return [{'ORG': entities}]
        entity_groupings = []
        for pos, entity in enumerate(entities):
            if self._should_ignore_entity(entity):
                continue
            parsed_entity = {}
            search_pos = pos + 1
            if entity[NER_ENTITY_TYPE_FIELD] == SUBORG_NER_LABEL:
                parent, pos_parent = self._get_first_entity(
                    entities[search_pos:], MAINORG_NER_LABEL, search_right=True
                )
                if not parent and fallback:
                    parent, _ = self._get_first_entity(
                        entities[:pos], MAINORG_NER_LABEL, search_right=False
                    )
                if parent:
                    parsed_entity[SUBORG_NER_LABEL] = entity[NER_ENTITY_TEXT_FIELD].strip()
                    parsed_entity[MAINORG_NER_LABEL] = parent[NER_ENTITY_TEXT_FIELD].strip()
            elif entity[NER_ENTITY_TYPE_FIELD] == MAINORG_NER_LABEL:
                parsed_entity[MAINORG_NER_LABEL] = entity[NER_ENTITY_TEXT_FIELD].strip()
            if parsed_entity:
                self._add_location_info(entities, search_pos, parsed_entity)
                if parsed_entity not in entity_groupings:
                    entity_groupings.append(parsed_entity)
        return entity_groupings
   
    def get_el_input_organizations(self, grouped_entities, osm, span_text):
        """Get input organizations for entity linking"""
        organizations = []
        for group in grouped_entities:
            if MAINORG_NER_LABEL not in group or not group[MAINORG_NER_LABEL]:
                continue
            organization = {
                'main': group[MAINORG_NER_LABEL],
                'suborg': (group[SUBORG_NER_LABEL] 
                          if SUBORG_NER_LABEL in group and group[SUBORG_NER_LABEL] 
                          else '')
            }
            # Extract location information from group or OSM
            location_info = self._get_location_info(group, osm)
            # Assign extracted values
            organization['city'] = location_info['city']
            organization['region'] = location_info['region']
            organization['country'] = location_info['country']
            # Construct 'location' as "City, Country" (excluding region)
            organization['location'] = ', '.join(filter(None, [organization['city'], organization['country']]))
            organization['span_text'] = span_text
            if organization not in organizations:
                organizations.append(organization)
        return organizations

    def _add_location_info(self, entities, search_pos, parsed_entity):
        """Add hierarchical location information to parsed entity."""
        for label in [CITY_NER_LABEL, REGION_NER_LABEL, COUNTRY_NER_LABEL]:
            entity, pos = self._get_first_entity(entities[search_pos:], label)
            if entity:
                parsed_entity[label] = entity[NER_ENTITY_TEXT_FIELD].strip()
                search_pos += pos + 1
       
    def _get_location_info(self, group, osm):
        """Extract location information from group or OSM."""
        # Handle None values
        if osm is None:
            osm = {}
        return {
            'city': osm.get(CITY_OSM_LABEL, "") or group.get(CITY_NER_LABEL, ""),
            'region': osm.get(STATE_OSM_LABEL, "") or osm.get(PROVINCE_OSM_LABEL, "") or group.get(REGION_NER_LABEL, ""),
            'country': osm.get(COUNTRY_OSM_LABEL, "") or group.get(COUNTRY_NER_LABEL, "")
        }
              
    def _should_ignore_entity(self, entity):
        """Check if entity should be ignored"""
        if not isinstance(entity, dict):
            return True
        return (len(entity.get(NER_ENTITY_TEXT_FIELD, "")) < MIN_LENGTH_NER_ENTITY or
                entity[NER_ENTITY_TEXT_FIELD].startswith(IGNORE_NER_ENTITY_PREFIX))

    def _get_first_entity(self, entities, entity_type, search_right=True):
        """Find first entity of a specified type."""
        entities = entities if search_right else reversed(entities)
        for pos, entity in enumerate(entities):
            if entity[NER_ENTITY_TYPE_FIELD] == entity_type and not self._should_ignore_entity(entity):
                return entity, pos
        return None, 0

    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics or message if caching is disabled
        """
        if not self.use_cache:
            return {"status": "Cache is disabled"}
            
        total_entries = len(self.cached_predictions)
        
        # Count entries by type
        entries_by_linker = {}
        for key in self.cached_predictions:
            linker_parts = key.split(':')[0:2]  # Get linker name and data source
            linker_key = ':'.join(linker_parts)
            entries_by_linker[linker_key] = entries_by_linker.get(linker_key, 0) + 1
        
        # Count expired entries
        current_time = time.time()
        expired_count = 0
        for key, value in self.cached_predictions.items():
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
        
    def clear_cache(self, expired_only=True):
        """
        Clear the cache.
        
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
            
            for key, value in self.cached_predictions.items():
                if 'timestamp' in value and (current_time - value['timestamp']) > self.cache_expiration:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cached_predictions[key]
                
            # Save updated cache to disk
            self._save_cache_to_disk()
            
            return {
                "status": "Cleared expired entries",
                "cleared_entries": len(keys_to_remove)
            }
        else:
            # Clear all entries
            entry_count = len(self.cached_predictions)
            self.cached_predictions = {}
            
            # Save updated cache to disk (empty)
            self._save_cache_to_disk()
            
            return {
                "status": "Cleared all entries",
                "cleared_entries": entry_count
            }
    
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
            self.cache_file = os.path.join(self.cache_dir, f"{self.linker_name}_{self.data_source}_cache.pkl")
            
            # Load existing cache if available
            self._load_cache_from_disk()
            
            return {"status": "Cache enabled", "loaded_entries": len(self.cached_predictions)}
        
        # If disabling and cache was previously enabled
        elif not enabled and old_status:
            # Save cache one last time before disabling
            self._save_cache_to_disk()
            saved_entries = len(self.cached_predictions)
            
            # Clear in-memory cache
            self.cached_predictions = {}
            self.cache_file = None
            
            return {"status": "Cache disabled", "saved_entries": saved_entries}
        
        # No change
        return {"status": "Cache already " + ("enabled" if enabled else "disabled")}
