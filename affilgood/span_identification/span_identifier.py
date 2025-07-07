import sys
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset
import re
import torch
from typing import List, Dict, Union, Any

# DEFAULT MODEL
DEFAULT_SPAN_MODEL = 'nicolauduran45/affilgood-span-multilingual-v2'

def clean_whitespaces(text):
    return re.sub(r'\s+', ' ', str(text).strip())

class SpanIdentifier:
    def __init__(
        self,
        model_path=None,
        device=None,
        batch_size=64,
        threshold_score=0.75,
        fix_predicted_words=True,
        title_case=False,
    ):
        self.model_path = model_path if model_path else DEFAULT_SPAN_MODEL
        self.device = 0 if device is None and torch.cuda.is_available() else -1 if device is None else device

        # Initialize pipeline model and tokenizer
        self.model = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(self.model_path),
            tokenizer=AutoTokenizer.from_pretrained(self.model_path),
            aggregation_strategy="simple",
            device=self.device
        )

        self.batch_size = batch_size
        self.threshold_score = threshold_score
        self.fix_predicted_words = fix_predicted_words
        self.title_case = title_case

    def identify_spans(self, text_list, batch_size=None):
    
        batch_size = batch_size if batch_size is not None else self.batch_size
    
        if isinstance(text_list, str):
            text_list = [text_list]

        text_list = [clean_whitespaces(text) for text in text_list]
        if self.title_case:
            text_list = [text.title() for text in text_list]

        outputs = self.model(
            text_list,
            batch_size=batch_size
        )

        if len(outputs) != len(text_list):
            raise RuntimeError("Mismatch between input texts and model outputs")

        results = []
        for raw_text, entities in zip(text_list, outputs):
            if self.fix_predicted_words:
                entities = self.fix_words(raw_text, entities)

            cleaned_entities = self.clean_and_merge_entities(entities)
            span_entities = [entity.get("word", "") for entity in cleaned_entities]

            results.append({
                "raw_text": raw_text,
                "span_entities": span_entities,
            })

        return results

    def fix_words(self, raw_text, entities):
        for entity in entities:
            start, end = entity["start"], entity["end"]
            entity["word"] = raw_text[start:end]
        return entities

    def clean_and_merge_entities(self, entities, min_score=0.75):
        entities = [entity for entity in entities if entity.get("score", 0) >= min_score]
        merged_entities = []
        i = 0
        while i < len(entities):
            current_entity = entities[i]
            if (i + 1 < len(entities)):
                next_entity = entities[i + 1]
                if (current_entity['end'] == next_entity['start'] and 
                    next_entity['word'][0].islower()):
                    merged_word = current_entity['word'] + next_entity['word']
                    merged_entity = {
                        "entity_group": current_entity['entity_group'],
                        "score": min(current_entity['score'], next_entity['score']),
                        "word": merged_word,
                        "start": current_entity['start'],
                        "end": next_entity['end']
                    }
                    merged_entities.append(merged_entity)
                    i += 2
                    continue
            merged_entities.append(current_entity)
            i += 1
        return merged_entities

