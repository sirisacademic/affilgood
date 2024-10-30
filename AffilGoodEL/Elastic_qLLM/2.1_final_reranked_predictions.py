#!/usr/bin/env python3

import re
import pandas as pd

from config import DATA_DIR, DATASETS

### For multiple-organizations predictions.
INPUT_COLUMNS_ELASTIC = [
  'fidx',
  'idx',
  'orig_raw_affiliation_string',
  'raw_affiliation_string',
  'gold_label',
  'predicted_label'
]

INPUT_COLUMNS_RERANKED = [
  'fidx',
  'idx',
  'orig_raw_affiliation_string',
  'raw_affiliation_string',
  'ror_id'
]

RENAME_ELASTIC_COLUMNS = {'gold_label': 'label'}
# Columns to merge on.
MERGE_COLUMNS = ['fidx', 'idx', 'orig_raw_affiliation_string', 'raw_affiliation_string']
DROP_AFTER_MERGE_COLUMNS = ['raw_affiliation_string', 'ror_id']
RENAME_AFTER_MERGE_COLUMNS = {'orig_raw_affiliation_string': 'raw_affiliation_string'}
GROUPBY_COLUMNS = ['fidx','idx','raw_affiliation_string', 'label']
FINAL_RESULTS_COLUMNS = ['raw_affiliation_string', 'label', 'predicted_label']

COL_PREDICTIONS_ELASTIC = 'predicted_label'
COL_PREDICTIONS_LLM = 'ror_id'

NONE_PREDICTION_LLM = 'None'

#=========================== FUNCTIONS ============================

def handle_bad_lines(bad_line):
#-----------------------------
  print(f"Bad line: {bad_line}")
  return bad_line

def get_final_prediction(predicted_labels, ror_id):
#-------------------------------------------------
  final_prediction = ''
  pred_labels = predicted_labels.split('|')
  if len(pred_labels) > 0:
    # If there is no prediction by the LLM we take the first one by Elasticsearch.
    if not ror_id:
      final_prediction = pred_labels[0]
    elif ror_id != NONE_PREDICTION_LLM:
      # If the LLM predicted a ROR id within the candidates we get the full label.
      for pred_label in pred_labels:
        if ror_id in pred_label:
          final_prediction = pred_label
          break
  # Remove city, country in square brackets.
  if final_prediction:
    final_prediction = re.sub(f'\[.+?\] ', '', final_prediction)
  return final_prediction
    
def get_combined_multiple_predictions(predicted_labels_list):
#-----------------------------------------------------------
  return '|'.join(set([l.strip() for l in predicted_labels_list if l.strip() and l.strip != 'None']))

#============================== MAIN ==============================

for DATASET in DATASETS:

  PATH_DIR_PREDICTIONS = f'output/{DATA_DIR}/{DATASET}'

  FILE_PREDICTIONS_ELASTIC = f'predictions_elastic_{DATA_DIR}_{DATASET.lower()}.tsv'
  FILE_RERANKED_PREDICTIONS_LLM = f'reranked_predictions_elastic_{DATA_DIR}_{DATASET.lower()}.tsv'
  FILE_NAME_OUTPUT = f'final_predictions_elastic_reranked_{DATA_DIR}_{DATASET.lower()}.tsv'

  PATH_INPUT_FILE_ELASTIC = f'{PATH_DIR_PREDICTIONS}/{FILE_PREDICTIONS_ELASTIC}'
  PATH_INPUT_FILE_LLM = f'{PATH_DIR_PREDICTIONS}/{FILE_RERANKED_PREDICTIONS_LLM}'
  PATH_OUTPUT_FILE = f'{PATH_DIR_PREDICTIONS}/{FILE_NAME_OUTPUT}'

  print(f'Reading file {PATH_INPUT_FILE_ELASTIC}')
  pred_elastic = pd.read_csv(
    PATH_INPUT_FILE_ELASTIC,
    sep='\t',
    usecols=INPUT_COLUMNS_ELASTIC,
    converters={'fidx': int, 'idx': int},
    on_bad_lines=handle_bad_lines,
    encoding='utf-8',
    engine='python'
  )
  pred_elastic.fillna('', inplace=True)
  pred_elastic.rename(columns=RENAME_ELASTIC_COLUMNS, inplace=True)

  print(f'Reading file {PATH_INPUT_FILE_LLM}')
  # We have to pass na_filter to preserve the None values.
  pred_llm = pd.read_csv(
    PATH_INPUT_FILE_LLM,
    sep='\t',
    usecols=INPUT_COLUMNS_RERANKED,
    na_filter=False,
    converters={'fidx': int, 'idx': int},
    on_bad_lines=handle_bad_lines,
    encoding='utf-8',
    engine='python'
  )
  pred_llm.fillna('', inplace=True)

  pred_pipeline = pred_elastic.merge(pred_llm, on=MERGE_COLUMNS, how='left')
  pred_pipeline.fillna('', inplace=True)
  pred_pipeline[COL_PREDICTIONS_ELASTIC] = pred_pipeline.apply(lambda row:\
                                            get_final_prediction(row[COL_PREDICTIONS_ELASTIC], row[COL_PREDICTIONS_LLM]), axis=1)
  pred_pipeline.drop(columns=DROP_AFTER_MERGE_COLUMNS, inplace=True)
  pred_pipeline.rename(columns=RENAME_AFTER_MERGE_COLUMNS, inplace=True)

  # We have to combine the results.
  pred_pipeline.drop_duplicates(inplace=True)
  pred_pipeline = pred_pipeline.groupby(GROUPBY_COLUMNS).agg({COL_PREDICTIONS_ELASTIC: list}).reset_index()
  pred_pipeline[COL_PREDICTIONS_ELASTIC] = pred_pipeline.apply(lambda row: get_combined_multiple_predictions(row[COL_PREDICTIONS_ELASTIC]), axis=1)

  # Save results.
  pred_pipeline[FINAL_RESULTS_COLUMNS].to_csv(PATH_OUTPUT_FILE, sep='\t', index=False)

  print(f'Saved output file to {PATH_OUTPUT_FILE}')


