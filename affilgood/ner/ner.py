# ner.py
import sys
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import re

def clean_whitespaces(text):
#---------------------------
  return re.sub(r'\s+', ' ', str(text).strip())

class NER:
    def __init__(
        self,
        model_path="nicolauduran45/affilgood-ner-multilingual-v2",
        device=0,
        chunk_size=10000,
        max_parallel=10,
        batch_size=64,
        #threshold_score=0.75,
        fix_predicted_words=True,
        title_case=False,
        #col_raw_text="raw_text",
        #col_span_entities="raw_affiliation_string",
        #col_potential_error="potential_error_span",
        #span_entity_type_field="entity_group",
        #span_entity_text_field="word",
        #span_entity_score_field="score",
        #span_entity_start_field="start",
        #span_entity_end_field="end"
    ):
        # Initialize pipeline model and tokenizer
        self.model = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(model_path),
            tokenizer=AutoTokenizer.from_pretrained(model_path),
            aggregation_strategy="simple",
            device=device
        )
        
        # Configuration settings
        self.device = device
        self.chunk_size = chunk_size
        self.max_parallel = max_parallel
        self.batch_size = batch_size
        #self.threshold_score = threshold_score
        self.fix_predicted_words = fix_predicted_words
        self.title_case = title_case

    def recognize_entities(self, text_list):
        """
        Process a list of text data for span identification.

        Parameters:
        - text_list (list of dict): List of dictionaries, each containing "raw_text" and "span_entities".

        Returns:
        - List of dicts: Each dict contains the original text, span entities, and ner entities grouped by entity group for each span entity.
        """
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
        outputs = list(tqdm(self.model(KeyDataset(Dataset.from_dict({"text": spans_to_process}), "text"), batch_size=self.batch_size)))

        # Initialize the results structure for each item in text_list
        results = [{"raw_text": item["raw_text"], "span_entities": item["span_entities"], "ner": []} for item in text_list]

        for idx, span in enumerate(spans_to_process):
            # Map each output back to the corresponding text_list item and span_entities index
            entities = outputs[idx]
            if self.fix_predicted_words:
                entities = self.fix_words(span, entities)
            
            # Clean and merge spans
            cleaned_entities = self.clean_and_merge_entities(entities)

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

        return results
    
    def fix_words(self, raw_text, entities):
        """
        Adjusts entity text based on character offsets to ensure correct word extraction.
        """
        for entity in entities:
            start, end = entity["start"], entity["end"]
            entity["word"] = raw_text[start:end]
        return entities

    # Identify potential errors in entity splitting
    def clean_and_merge_entities(self,entities, min_score=0.75):
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

        # Step 2: Merge consecutive entities if they are split and the next starts with lowercase
        merged_entities = []
        i = 0

        while i < len(entities):
            # Check if the current entity can be merged with the next one
            current_entity = entities[i]
            
            if (i + 1 < len(entities)):
                next_entity = entities[i + 1]
                # Conditions for merging:
                # - Current entity's end matches next entity's start
                # - Next entity's word starts with a lowercase letter
                if (current_entity['end'] == next_entity['start'] and 
                    next_entity['word'][0].islower()):
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

            # If no merging, add the current entity as is
            merged_entities.append(current_entity)
            i += 1

        return merged_entities