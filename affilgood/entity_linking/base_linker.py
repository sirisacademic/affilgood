from abc import ABC, abstractmethod
from .constants import *

class BaseLinker(ABC):
    """Base class for entity linkers with shared functionalities."""
    def __init__(self):
        self.CACHED_PREDICTED_ID = {}
        self.CACHED_PREDICTED_NAME = {}
        self.CACHED_PREDICTED_ID_SCORE = {}
        self.is_initialized = False

    @abstractmethod
    def initialize(self):
        """Abstract method for initializing entity linking components."""
        pass  # This does nothing in the parent class but must be implemented in subclasses.

    @abstractmethod
    def get_single_prediction(self, organization):
        """Abstract method for getting predictions for one organization."""
        pass  # This does nothing in the parent class but must be implemented in subclasses.

    def process_chunk_el(self, chunk, return_scores=True):
        """Process chunks using the specific entity linkers."""
        # Ensure initialization before processing
        if not self.is_initialized:
            self.initialize()
        
        chunk_to_process = []
        chunk_index_map = []  # Track the position in text_list and span_entities index
        
        # Prepare chunks for processing
        for idx, item in enumerate(chunk):
            span_entities = item.get("ner_raw", [])
            osm_entities = item.get("osm", [])
            for span_idx, (span, osm) in enumerate(zip(span_entities, osm_entities)):
                chunk_to_process.append((span, osm))
                chunk_index_map.append((idx, span_idx))
        
        # Process chunks
        processed_list = []
        for ner, osm in chunk_to_process:
            result = {}
            result['grouped_entities'] = self.get_entity_groupings(ner)
            result['el_input_organizations'] = self.get_el_input_organizations(
                result['grouped_entities'], 
                osm
            )
            predictions, scores = self.get_predicted_labels(
                result['el_input_organizations']
            )
            result[COL_PREDICTIONS_EL] = predictions
            result[COL_PREDICTIONS_SCORES_EL] = scores
            processed_list.append(result)
        
        # Format results
        results = [{
            "raw_text": item["raw_text"],
            "span_entities": item["span_entities"],
            "ner": item.get('ner', []),
            "ner_raw": item.get("ner_raw", []),
            "osm": item.get("osm", []),
            "ror": []
        } for item in chunk]
        
        # Map results back to original structure
        for idx, _ in enumerate(chunk_to_process):
            entities = processed_list[idx]
            item_idx, ror_idx = chunk_index_map[idx]
            while len(results[item_idx]["ror"]) <= ror_idx:
                results[item_idx]["ror"].append("")
            results[item_idx]["ror"][ror_idx] = entities[COL_PREDICTIONS_SCORES_EL] if return_scores else entities[COL_PREDICTIONS_EL]
        
        return results

    def get_predicted_labels(self, organizations):
        """Get predicted labels for organizations"""
        predicted_names = {}
        predicted_scores = {}
        for organization in organizations:
            predicted_id, predicted_name, predicted_score = self.get_single_prediction(
                organization
            )
            if predicted_id:
                predicted_id = predicted_id.replace(ROR_URL, "")
                if predicted_id not in predicted_names:
                    predicted_names[predicted_id] = predicted_name
                if (predicted_id not in predicted_scores or 
                    predicted_scores[predicted_id] < predicted_score):
                    predicted_scores[predicted_id] = predicted_score
        # Format predictions
        predicted_names_ids = [
            f'{predicted_names[predicted_id]} {{{ROR_URL}{predicted_id}}}' 
            for predicted_id in predicted_names
        ]
        predicted_labels = '|'.join(predicted_names_ids)
        predicted_labels_scores = '|'.join([
            f'{predicted_names[predicted_id]} {{{ROR_URL}{predicted_id}}}:{predicted_scores[predicted_id]:.2f}'
            for predicted_id in predicted_names
        ])
        return [predicted_labels, predicted_labels_scores]

    def get_entity_groupings(self, entities, fallback=True):
        """Generate groupings from NER output."""
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

    def get_el_input_organizations(self, grouped_entities, osm):
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

    def _get_location_string(self, group, osm):
        """Create a cache-friendly location string."""
        location_info = self._get_location_info(group, osm)
        return "|".join(filter(None, [location_info['city'], location_info['region'], location_info['country']]))
               
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

    def _get_from_cache(self, cache_key):
        """Retrieve results from the cache."""
        if cache_key in self.CACHED_PREDICTED_ID:
            return (
                self.CACHED_PREDICTED_ID[cache_key],
                self.CACHED_PREDICTED_NAME[cache_key],
                self.CACHED_PREDICTED_ID_SCORE[cache_key]
            )
        return None, None, None

    def _update_cache(self, cache_key, predicted_id, predicted_name, predicted_score):
        """Update the cache with new results."""
        self.CACHED_PREDICTED_ID[cache_key] = predicted_id
        self.CACHED_PREDICTED_NAME[cache_key] = predicted_name
        self.CACHED_PREDICTED_ID_SCORE[cache_key] = predicted_score







