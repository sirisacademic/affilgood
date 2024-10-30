#!/usr/bin/env python3

import sys
sys.path.append('functions')

import re
import ner
import pandas as pd
from config import THRESHOLD_MAIN_ORG, THRESHOLD_SUB_ORG, DATA_DIR, DATASETS

# Only the ones present in the source file will be considered.
KEEP_COLUMNS = [
  'fidx',
  'idx',
  'raw_affiliation_string',
  'label',
  'gold_label',
  'predicted_label',
  'predicted_score',
  'type'
]

COL_PREDICTIONS_EL = 'predicted_label'
COL_SCORES_EL = 'predicted_score'
COL_TYPE_NER = 'type'

RENAME_COLUMNS = {'gold_label': 'label'}

DROP_RENAME = ('orig_raw_affiliation_string', 'raw_affiliation_string')

GROUP_BY_COLUMNS = ['fidx', 'idx', 'raw_affiliation_string', 'label']
SAVE_COLUMNS = ['raw_affiliation_string', 'label', 'predicted_label']

#=========================== FUNCTIONS ============================

def get_final_prediction(predicted_labels, predicted_scores, org_type):
#------------------------------------------------------------------------
  final_prediction = ''
  labels = predicted_labels.split('|')
  threshold = THRESHOLD_SCORES[org_type] if org_type else THRESHOLD_SCORES[ner.MAINORG_LABEL]
  scores = [float(score) if score else 0 for score in predicted_scores.split('|')]
  if len(labels) > 0:
    final_prediction = labels[0] if scores[0] > threshold else ''
    # Remove city, country in square brackets.
    final_prediction = re.sub(f'\[.+?\] ', '', final_prediction)
  return final_prediction
    

def combine_predictions(df_pred):
#----------------------------------------
  groupby_columns = [col for col in GROUP_BY_COLUMNS if col in df_pred]
  df_combined = df_pred.groupby(groupby_columns).agg({COL_PREDICTIONS_EL:list}).reset_index()
  df_combined[COL_PREDICTIONS_EL] = df_combined.apply(lambda row: '|'.join(set([l.strip() for l in row[COL_PREDICTIONS_EL] if l])), axis=1)
  df_combined.sort_values(by=groupby_columns, inplace=True, ignore_index=True)
  return df_combined

#============================== MAIN ==============================

THRESHOLD_SCORES = {}
THRESHOLD_SCORES[ner.MAINORG_LABEL] = THRESHOLD_MAIN_ORG
THRESHOLD_SCORES[ner.SUBORG_LABEL] = THRESHOLD_SUB_ORG

for DATASET in DATASETS:

  SUBDIR_NER_OUTPUT = f'{DATA_DIR}/{DATASET}'

  # Entity linking output path.
  EL_OUTPUT_DIR = f'output/{SUBDIR_NER_OUTPUT}'
  EL_OUTPUT_FILE_NAME = f'predictions_elastic_{SUBDIR_NER_OUTPUT.replace("/", "_").lower()}.tsv'
  
  # Path input file with retrieved organizations.
  PATH_INPUT_FILE = f'{EL_OUTPUT_DIR}/{EL_OUTPUT_FILE_NAME}'
  
  # Output file final predictions.
  PATH_OUTPUT_FILE = f'{EL_OUTPUT_DIR}/final_predictions_elastic_no_reranked_{SUBDIR_NER_OUTPUT.replace("/", "_").lower()}_thres_{THRESHOLD_SCORES[ner.MAINORG_LABEL]}.tsv'
  

  pred_elastic = pd.read_csv(PATH_INPUT_FILE, sep='\t')
  pred_elastic.fillna('', inplace=True)
  #pred_elastic.drop(columns=DROP_RENAME[1], inplace=True)
  #pred_elastic.rename(columns={DROP_RENAME[0]: DROP_RENAME[1]}, inplace=True)
  pred_elastic = pred_elastic[[col for col in KEEP_COLUMNS if col in pred_elastic]]
  pred_elastic.rename(columns=RENAME_COLUMNS, inplace=True)

  # If there is no 'type' column we assume all to be main organizations.
  if COL_TYPE_NER not in pred_elastic:
    pred_elastic[COL_TYPE_NER] = ner.MAINORG_LABEL

  # Keep predictions above threshold (considering type of organization being linked).
  pred_elastic[COL_PREDICTIONS_EL] = pred_elastic.apply(lambda row:\
                                      get_final_prediction(row[COL_PREDICTIONS_EL], row[COL_SCORES_EL], row[COL_TYPE_NER]), axis=1)

  # Combine predictions when multiple organizations per affiliation string.                                    
  pred_elastic = combine_predictions(pred_elastic)

  # Save to TSV file.
  pred_elastic[SAVE_COLUMNS].to_csv(PATH_OUTPUT_FILE, sep='\t', index=False)

  print(f'Saved output file to {PATH_OUTPUT_FILE}')


