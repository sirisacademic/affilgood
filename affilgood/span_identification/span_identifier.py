# span_identifier.py
import sys
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import re

DEFAULT_SPAN_MODEL = 'SIRIS-Lab/affilgood-span-multilingual'

def clean_whitespaces(text):
  return re.sub(r'\s+', ' ', str(text).strip())

class SpanIdentifier:
    def __init__(
        self,
        model_path=None,
        device=0,
        chunk_size=10000,
        max_parallel=10,
        batch_size=64,
        threshold_score=0.75,
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
    
        if model_path is None:
            model_path = DEFAULT_SPAN_MODEL
    
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
        self.threshold_score = threshold_score
        self.fix_predicted_words = fix_predicted_words
        self.title_case = title_case

    def identify_spans(self, text_list):
        """
        Process a list of text data for span identification.

        Parameters:
        - text_list (list of str): List of strings containing text data.

        Returns:
        - List of dicts: Each dict contains the original text, a list of identified spans, and a list of potential errors.
        """
        # Handle the case where text_list is a string
        if isinstance(text_list, str):
            text_list = [text_list]
            
        # Clean and optionally apply title case to each text entry
        text_list = [clean_whitespaces(text) for text in text_list]
        if self.title_case:
            text_list = [text.title() for text in text_list]

        # Run the span identification model
        outputs = []
        for out in self.model(KeyDataset(Dataset.from_dict({"text": text_list}), "text"), batch_size=self.batch_size):
            outputs.append(out)

        # Process results
        results = []
        for i, raw_text in enumerate(text_list):
            entities = outputs[i]
            if self.fix_predicted_words:
                entities = self.fix_words(raw_text, entities)

            # Merge and clean spans
            cleaned_entities = self.clean_and_merge_entities(entities)
            span_entities = [entity.get("word", "") for entity in cleaned_entities]
            #potential_error = self.potential_errors(entities)  # Now returns a list of booleans

            # Add the processed data for the current text to the results
            results.append({
                "raw_text": raw_text,
                "span_entities": span_entities,
            })

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


        # def potential_errors(self, entities):
    #     """
    #     Detects potential errors in entity spans based on spacing issues.
    #     """
    #     potential_error = False
    #     previous_entity_end = 0
    #     for entity in entities:
    #         start = entity.get("start", 0)
    #         if start > 0 and start - previous_entity_end < 1:
    #             potential_error = True
    #             break
    #         previous_entity_end = entity.get("end", 0)
    #     return potential_error
    
