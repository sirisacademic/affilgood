# ner.py
import sys
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, logging as transformers_logging
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from transformers.utils.logging import disable_progress_bar
import re
import numpy as np

def clean_whitespaces(text):
  return re.sub(r'\s+', ' ', str(text).strip())

#DEFAULT_NER_MODEL = "SIRIS-Lab/affilgood-NER-multilingual"
DEFAULT_NER_MODEL = "nicolauduran45/affilgood-ner-multilingual-v2"

# NER entity Labels. TODO: Unify with entity_linking/constants.py
MAINORG_NER_LABEL = 'ORG'
SUB_NER_LABEL = 'SUB'
SUBORG_NER_LABEL = 'SUBORG'
CITY_NER_LABEL = 'CITY'
REGION_NER_LABEL = 'REGION'
COUNTRY_NER_LABEL = 'COUNTRY'

# Entities containing words with these prefixes and labeled as "SUB" are changed to "SUBORG"
SUB_TO_SUBORG = ["cent", "inst", "lab"]

class NER:
    def __init__(
        self,
        model_path=None,
        device=0,
        chunk_size=10000,
        max_parallel=10,
        batch_size=64,
        #threshold_score=0.75,
        fix_predicted_words=True,
        sub_to_suborg=True,
        title_case=False,
        verbose=False
    ):
    
        self.model_path = DEFAULT_NER_MODEL if model_path is None else model_path
    
        # Initialize pipeline model and tokenizer
        self.model = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(self.model_path),
            tokenizer=AutoTokenizer.from_pretrained(self.model_path),
            aggregation_strategy="simple",
            device=device
        )

        if not verbose:
            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()

        # Configuration settings
        self.device = device
        self.chunk_size = chunk_size
        self.max_parallel = max_parallel
        self.batch_size = batch_size
        #self.threshold_score = threshold_score
        self.fix_predicted_words = fix_predicted_words
        self.sub_to_suborg = sub_to_suborg
        self.title_case = title_case

    def refine_spans_by_geography(self, ner_results):
        """
        Refine spans when multiple geographical entities are detected in a single span.
        
        Parameters:
        - ner_results (list): The output of recognize_entities with NER annotations.
        
        Returns:
        - list: Refined spans with geographical boundaries considered.
        """
        refined_results = []
        
        for item in ner_results:
            raw_text = item['raw_text']
            span_entities = item['span_entities']
            ner = item.get('ner', [])
            ner_raw = item.get('ner_raw', [])
            
            # Skip items without NER results
            if not ner or not ner_raw:
                refined_results.append(item)
                continue
            
            # Process each span
            refined_spans = []
            refined_ner = []
            refined_ner_raw = []
            
            for span_idx, (span, span_ner, span_ner_raw) in enumerate(zip(span_entities, ner, ner_raw)):
                # Check if this span has multiple geographical entity pairs
                org_entities = []
                city_entities = []
                country_entities = []
                
                # Extract all geographic entities with positions
                geo_entities = []
                
                for entity in span_ner_raw:
                    if entity['entity_group'] == MAINORG_NER_LABEL:
                        org_entities.append(entity)
                    elif entity['entity_group'] == CITY_NER_LABEL:
                        city_entities.append(entity)
                        geo_entities.append(entity)
                    elif entity['entity_group'] == COUNTRY_NER_LABEL:
                        country_entities.append(entity)
                        geo_entities.append(entity)
                
                # If we don't have multiple geographic entities or only one organization, keep as is
                if len(geo_entities) <= 1 or len(org_entities) <= 1:
                    refined_spans.append(span)
                    refined_ner.append(span_ner)
                    refined_ner_raw.append(span_ner_raw)
                    continue
                
                # Sort all entities by start position
                all_entities = sorted(span_ner_raw, key=lambda e: e['start'])
                
                # Find potential split points - places where we transition from one geographic region to another
                split_points = []
                
                for i in range(len(all_entities) - 1):
                    current = all_entities[i]
                    next_entity = all_entities[i + 1]
                    
                    # Check for transitions that suggest a new organization entry
                    is_geo_entity = current['entity_group'] in [CITY_NER_LABEL, COUNTRY_NER_LABEL]
                    is_next_org = next_entity['entity_group'] == MAINORG_NER_LABEL or next_entity['entity_group'] == SUBORG_NER_LABEL
                    has_gap = next_entity['start'] - current['end'] > 3  # More permissive gap
                    
                    if is_geo_entity and is_next_org and has_gap:
                        # Find text between them - this is where we'll split
                        text_between = span[current['end']:next_entity['start']].strip()
                        
                        # More permissive check for delimiters or significant gap
                        is_delimiter = any(delim in text_between for delim in ['.', ';', ',', ':', '-'])
                        has_number = any(char.isdigit() for char in text_between)
                        is_long_gap = len(text_between) > 2
                        
                        if is_delimiter or has_number or is_long_gap:
                            split_points.append((current['end'], next_entity['start']))
                                
                # If no split points found, keep as is
                if not split_points:
                    refined_spans.append(span)
                    refined_ner.append(span_ner)
                    refined_ner_raw.append(span_ner_raw)
                    continue
                
                # Create new spans based on split points
                start_idx = 0
                for end_idx, next_start_idx in split_points:
                    # Extract the substring for this span
                    new_span = span[start_idx:end_idx].strip()
                    
                    # Find entities that belong to this span
                    new_span_ner_raw = [
                        {
                            'entity_group': e['entity_group'],
                            'score': e['score'],
                            'word': e['word'],
                            'start': e['start'] - start_idx,  # Adjust positions relative to new span
                            'end': e['end'] - start_idx
                        }
                        for e in span_ner_raw if e['start'] >= start_idx and e['end'] <= end_idx
                    ]
                    
                    # Create NER dict for this span
                    new_span_ner = {}
                    for e in new_span_ner_raw:
                        entity_group = e['entity_group']
                        if entity_group not in new_span_ner:
                            new_span_ner[entity_group] = []
                        new_span_ner[entity_group].append(e['word'])
                    
                    # Add to refined results
                    refined_spans.append(new_span)
                    refined_ner.append(new_span_ner)
                    refined_ner_raw.append(new_span_ner_raw)
                    
                    # Update start index for next span
                    start_idx = next_start_idx
                
                # Add the final span after the last split point
                if start_idx < len(span):
                    new_span = span[start_idx:].strip()
                    
                    # Find entities that belong to this span
                    new_span_ner_raw = [
                        {
                            'entity_group': e['entity_group'],
                            'score': e['score'],
                            'word': e['word'],
                            'start': e['start'] - start_idx,  # Adjust positions relative to new span
                            'end': e['end'] - start_idx
                        }
                        for e in span_ner_raw if e['start'] >= start_idx
                    ]
                    
                    # Create NER dict for this span
                    new_span_ner = {}
                    for e in new_span_ner_raw:
                        entity_group = e['entity_group']
                        if entity_group not in new_span_ner:
                            new_span_ner[entity_group] = []
                        new_span_ner[entity_group].append(e['word'])
                    
                    # Add to refined results
                    refined_spans.append(new_span)
                    refined_ner.append(new_span_ner)
                    refined_ner_raw.append(new_span_ner_raw)
            
            # Create a new item with refined spans
            refined_item = {
                'raw_text': raw_text,
                'span_entities': refined_spans,
                'ner': refined_ner,
                'ner_raw': refined_ner_raw
            }
            
            refined_results.append(refined_item)
        
        return refined_results

    def recognize_entities(self, text_list, batch_size=None):
        """
        Parameters:
        - text_list (list of dict): Each dict should contain "raw_text" and "span_entities".
        - batch_size (int, optional): Overrides the default batch size for model inference.

        Returns:
        - list of dicts: NER-annotated results for each span entity in the input.
        """

        batch_size = batch_size if batch_size is not None else self.batch_size
        
        # Flatten all span_entities into a single list for batch processing
        spans_to_process = []
        span_index_map = []  # Track the position in text_list and span_entities index

        for idx, item in enumerate(text_list):
            span_entities = item.get("span_entities", [])
            for span_idx, span in enumerate(span_entities):
                # Clean and optionally apply title case to the span text
                span = clean_whitespaces(span)
                if self.title_case:
                    span = span.title()
                spans_to_process.append(span)
                span_index_map.append((idx, span_idx))  # Record which item and span this belongs to

        # Run the span identification model on the entire batch
        outputs = list(self.model(
            KeyDataset(Dataset.from_dict({"text": spans_to_process}), "text"),
            batch_size=batch_size
        ))

        # Initialize the results structure for each item in text_list
        results = [{"raw_text": item["raw_text"], "span_entities": item["span_entities"], "ner": [], "ner_raw": []} for item in text_list]

        for idx, span in enumerate(spans_to_process):
            # Map each output back to the corresponding text_list item and span_entities index
            entities = outputs[idx]
            entities = [entry for entry in entities if entry['word']]
            if self.fix_predicted_words:
                entities = self.fix_words(span, entities)
            if self.sub_to_suborg:
                entities = self.change_suborg(entities)
            
            # Clean and merge spans
            cleaned_entities = self.combine_entities(entities)
            cleaned_entities = self.clean_and_merge_entities(cleaned_entities)

            # Organize ner_entities by entity_group
            ner_entities = {}
            for entity in cleaned_entities:
                entity_group = entity.get("entity_group")
                word = entity.get("word", "")
                if entity_group not in ner_entities:
                    ner_entities[entity_group] = []
                ner_entities[entity_group].append(word)

            # Append ner entities for the current span to the correct entry in results
            item_idx, span_idx = span_index_map[idx]
            # Ensure that each item in "ner" corresponds to each span in "span_entities"
            if len(results[item_idx]["ner"]) <= span_idx:
                results[item_idx]["ner"].append({})
            results[item_idx]["ner"][span_idx] = ner_entities

            if len(results[item_idx]["ner_raw"]) <= span_idx:
                results[item_idx]["ner_raw"].append({})
            results[item_idx]["ner_raw"][span_idx] = cleaned_entities#ner_entities
            
        # Refine span based on identified geographical entities if needed.
        results = self.refine_spans_by_geography(results)

        return results
      
    def change_suborg(self, entities):
        """
        Changes label from SUB to SUBORG in particular cases.
        """
        for entity in entities:
            if entity["entity_group"] == SUB_NER_LABEL and any([word.startswith(pref) for word in entity["word"].lower().split() for pref in SUB_TO_SUBORG]):
                entity["entity_group"] = SUBORG_NER_LABEL
        return entities
      
    def fix_words(self, raw_text, entities):
        """
        Adjusts entity text based on character offsets and fixes missing closing parentheses at the end of entities.
        """
        for entity in entities:
            start, end = entity["start"], entity["end"]
            entity_text = raw_text[start:end]
            
            # Check specifically for unclosed parenthesis at the end of the entity
            last_open_paren = entity_text.rfind('(')
            last_close_paren = entity_text.rfind(')')
            
            # If there's an opening parenthesis after the last closing one (or no closing one)
            if last_open_paren > -1 and (last_close_paren == -1 or last_open_paren > last_close_paren):
                # Look ahead in the raw text for the next closing parenthesis
                next_paren_pos = raw_text.find(')', end)
                
                if next_paren_pos > -1:
                    # Check that there are no delimiters between our entity end and the closing parenthesis
                    text_between = raw_text[end:next_paren_pos]
                    # No spaces, commas, semicolons, periods, or other delimiters
                    if not any(delim in text_between for delim in [' ', ',', ';', ':', '.', '\n', '\t']):
                        # Extend the end boundary to include the closing parenthesis
                        new_end = next_paren_pos + 1
                        entity["end"] = new_end
                        entity_text = raw_text[start:new_end]
            
            # Set the word to the adjusted text
            entity["word"] = entity_text
            
        return entities

    def clean_and_merge_entities(self, entities, min_score=0):
        """
        Cleans and merges entities based on score and adjacency criteria.

        Parameters:
        - entities (list of dict): List of entity spans, each with "start", "end", "word", and "score".
        - min_score (float): Minimum score threshold for keeping entities.

        Returns:
        - list of dict: A list of cleaned and merged entities.
        """
        # Step 1: Filter out entities with score below min_score
        entities = [entity for entity in entities if entity.get("score", 0) >= min_score]

        # Step 2: Merge consecutive entities if they are split and the next starts with lowercase or a number
        merged_entities = []
        i = 0

        while i < len(entities):
            # Check if the current entity can be merged with the next one
            current_entity = entities[i]
            
            if (i + 1 < len(entities)):
                next_entity = entities[i + 1]
                # Conditions for merging:
                # - Current entity's end matches next entity's start (no space between them)
                # - Next entity's word starts with a lowercase letter or a number
                if (current_entity['end'] == next_entity['start'] and 
                    (next_entity['word'][0].islower() or next_entity['word'][0].isdigit())):
                    # Merge words and adjust end position
                    merged_word = current_entity['word'] + next_entity['word']
                    merged_entity = {
                        "entity_group": current_entity['entity_group'],
                        "score": min(current_entity['score'], next_entity['score']),  # Keep the lower score
                        "word": merged_word,
                        "start": current_entity['start'],
                        "end": next_entity['end']
                    }
                    merged_entities.append(merged_entity)
                    i += 2  # Skip the next entity as it is now merged
                    continue

            # If no merging is needed, add the current entity as is
            merged_entities.append(current_entity)
            i += 1

        return merged_entities
    
    def combine_entities(self, entities):
        try:
            combined_entities = []
            i = 0

            while i < len(entities):
                current_entity = entities[i]

                # Check if we can merge with the next entity
                if i < len(entities) - 1:
                    next_entity = entities[i + 1]

                    # Conditions to combine entities:
                    if (current_entity['word'][-1].islower() and
                        next_entity['word'][0].islower()):

                        # Merge the text, adjust the end position, and take the average score
                        merged_entity = {
                            'start': current_entity['start'],
                            'end': next_entity['end'],
                            'entity_group': next_entity['entity_group'],
                            'score': (current_entity['score'] + next_entity['score']) / 2,
                            'word': current_entity['word'] + next_entity['word']
                        }

                        # Add the merged entity to the list
                        combined_entities.append(merged_entity)
                        # Skip the next entity since it's merged
                        i += 2
                        continue

                # If no merge occurred, add the current entity as-is
                combined_entities.append(current_entity)
                i += 1

            return np.array(combined_entities, dtype=object)
        except:
            return entities
