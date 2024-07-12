############################################
### utils/functions.py
### Functions UTIL

import os
import datetime
import glob
import pandas as pd
import json
import re
import numpy as np

def clean_whitespaces(text):
#---------------------------
  return re.sub(r'\s+', ' ', str(text).strip())
 
def read_file(file_path, converters=None):
#----------------------------------------
  if file_path.endswith('.json'):
    try:
      # If it is a serialized dictionary.
      df = pd.read_json(file_path, orient='index').fillna('')
    except AttributeError as e:
      # If it is a serialized list.
      df = pd.read_json(file_path).fillna('')
  elif file_path.endswith('.csv'):
    df = pd.read_csv(file_path, converters=converters).fillna('')
  elif file_path.endswith('.tsv'):
    df = pd.read_csv(file_path, sep='\t', converters=converters).fillna('')
  elif file_path.endswith('.parquet'):
    df = pd.read_parquet(file_path).fillna('')
  elif file_path.endswith('.pkl'):
    df = pd.read_pickle(file_path).fillna('')
  else:
    raise ValueError(f'Unsupported file extension for file: {file_path}')
  return df

def read_file_chunks(file_path, chunk_size=0, filter_column='', filter_value='', converters=None):
#------------------------------------------------------------------------------
  chunks = []
  df = read_file(file_path, converters=converters)
  if filter_column and filter_column in df:
    df = df[df[filter_column] != filter_value]
  # Get chunks.
  if chunk_size:
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]   
  else:
    chunks = [df] 
  return chunks

def write_output(file_path, df, columns=[]):
#------------------------------------------------------------------------------
  if columns:
    df = df[columns]
  if file_path.endswith('.json'):
    df.to_json(file_path, orient='index', indent=4)
  elif file_path.endswith('.csv'):
    df.to_csv(file_path, index=False)
  elif file_path.endswith('.tsv'):
    df.to_csv(file_path, sep='\t', index=False)
  elif file_path.endswith('.parquet'):
    df.to_parquet(file_path, index=False)
  elif file_path.endswith('.pkl'):
    df.reset_index(drop=True).to_pickle(file_path)
  else:
    raise ValueError(f'Unsupported file extension for file: {file_path}')

def days_since_last_update(file_path):
#------------------------------------------------------------------------------
  modification_time = os.path.getmtime(file_path)
  modification_date = datetime.datetime.fromtimestamp(modification_time)
  current_date = datetime.datetime.now()
  time_difference = current_date - modification_date
  return time_difference.days



