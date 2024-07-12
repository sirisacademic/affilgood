############################################
### AffilGoodNER/functions.py
### Functions NER

import sys
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

from AffilGoodNER.config import *
from AffilGoodSpan.config import COL_POTENTIAL_ERROR_SPAN
from utils.functions import *

# Function to process a single chunk with KeyDataset.
def process_chunk_ner(ner_pipeline, df_chunk):
#------------------------------------------------------------------
  outputs = [] 
  drop_columns = [] 
  df_chunk[COL_RAW_AFFILIATION] = df_chunk.apply(lambda row: clean_whitespaces(row[COL_RAW_AFFILIATION]), axis=1)
  if TITLE_CASE_NER:
    df_chunk[COL_RAW_AFFILIATION] = df_chunk.apply(lambda row: row[COL_RAW_AFFILIATION].title(), axis=1)
  for out in tqdm(ner_pipeline(KeyDataset(Dataset.from_pandas(df_chunk), COL_RAW_AFFILIATION), batch_size=BATCH_SIZE_PIPELINE_NER)):
    outputs.append(out)
  df_chunk[COL_NER_ENTITIES] = outputs
  if FIX_PREDICTED_WORDS_NER:
    df_chunk[COL_NER_ENTITIES] = df_chunk.apply(lambda row: fix_words(row[COL_RAW_AFFILIATION], row[COL_NER_ENTITIES]), axis=1)
  if COL_POTENTIAL_ERROR_NER:
    df_chunk[COL_POTENTIAL_ERROR_NER] = df_chunk.apply(lambda row: potential_errors(row[COL_NER_ENTITIES]), axis=1)
  if COL_POTENTIAL_ERROR_SPAN in df_chunk:
    drop_columns.append(COL_POTENTIAL_ERROR_SPAN)
  if drop_columns:
    df_chunk = df_chunk.drop(columns=drop_columns)
  return df_chunk

def ner_pipeline(model_name_path, device=0):
#--------------------------------------------------------------
  ner_model = AutoModelForTokenClassification.from_pretrained(model_name_path)
  ner_tokenizer = AutoTokenizer.from_pretrained(model_name_path)
  ner_pipeline = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy='simple', device=device)
  return ner_pipeline

### Fix "word" strings to match the offsets.
def fix_words(raw_affil_string, entities):
#------------------------------------------
  for entity in entities:
    entity[NER_ENTITY_TEXT_FIELD] = raw_affil_string[entity[NER_ENTITY_START_FIELD]:entity[NER_ENTITY_END_FIELD]]
  return entities
   
### Identify potential errors in entity splitting.
def potential_errors(entities):
#------------------------------------------
  potential_error = False
  previous_entity_end = 0
  for entity in entities:
    potential_error = (entity[NER_ENTITY_START_FIELD] > 0 and entity[NER_ENTITY_START_FIELD]-previous_entity_end < 1)
    previous_entity_end = entity[NER_ENTITY_END_FIELD]
    if potential_error:
      break
  return potential_error
  
  
