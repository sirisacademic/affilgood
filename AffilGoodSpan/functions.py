############################################
### AffilGoodSpan/functions.py
### Functions for AffilGoodSpan

import sys
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

from AffilGoodSpan.config import *
from utils.functions import *

# Function to process a single chunk with KeyDataset
def process_chunk_span(span_pipeline, df_chunk):
#------------------------------------------------------------------
  expanded_rows = []
  df_chunk[COL_RAW_TEXT] = df_chunk.apply(lambda row: clean_whitespaces(row[COL_RAW_TEXT]), axis=1)
  if TITLE_CASE_SPAN:
    df_chunk[COL_RAW_TEXT] = df_chunk.apply(lambda row: row[COL_RAW_TEXT].title(), axis=1)
  outputs = []
  for out in tqdm(span_pipeline(KeyDataset(Dataset.from_pandas(df_chunk), COL_RAW_TEXT), batch_size=BATCH_SIZE_PIPELINE_SPAN)):
    outputs.append(out)
  for index, row in df_chunk.iterrows():
    raw_text = row[COL_RAW_TEXT]
    entities = outputs[index]
    if FIX_PREDICTED_WORDS_SPAN:
      entities = fix_words(raw_text, entities)
    potential_error = potential_errors(entities)
    for entity in entities:
      expanded_rows.append({
        COL_RAW_TEXT: raw_text,
        COL_SPAN_ENTITIES: entity[SPAN_ENTITY_TEXT_FIELD],
        COL_POTENTIAL_ERROR_SPAN: potential_error
      })
  return pd.DataFrame(expanded_rows)

def span_pipeline(model_name_path, device=0):
#------------------------------------------------------------------
  span_model = AutoModelForTokenClassification.from_pretrained(model_name_path)
  span_tokenizer = AutoTokenizer.from_pretrained(model_name_path)
  span_pipeline = pipeline('ner', model=span_model, tokenizer=span_tokenizer, aggregation_strategy='simple', device=device)
  return span_pipeline

# Fix "word" strings to match the offsets
def fix_words(raw_text, entities):
#------------------------------------------------------------------
  for entity in entities:
    entity[SPAN_ENTITY_TEXT_FIELD] = raw_text[entity[SPAN_ENTITY_START_FIELD]:entity[SPAN_ENTITY_END_FIELD]]
  return entities

# Identify potential errors in entity splitting
def potential_errors(entities):
#------------------------------------------------------------------
  potential_error = False
  previous_entity_end = 0
  for entity in entities:
    potential_error = (entity[SPAN_ENTITY_START_FIELD] > 0 and entity[SPAN_ENTITY_START_FIELD] - previous_entity_end < 1)
    previous_entity_end = entity[SPAN_ENTITY_END_FIELD]
    if potential_error:
      break
  return potential_error

