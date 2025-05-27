#!/usr/bin/env python3

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Substring of URLs to omit when downloading S2AFF data
OMIT_S2AFF = ['ner_model', 'training', 'ror-data.json']
S2AFF_PATH = '.'

#----------------- FUNCTIONS ------------------

def get_s3_client():
#---------------------------------
  return boto3.client('s3', config=Config(signature_version=UNSIGNED))

def list_s3_objects(bucket, prefix):
#---------------------------------
  s3 = get_s3_client()
  paginator = s3.get_paginator('list_objects_v2')
  response_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
  objects = []
  for page in response_iterator:
    if 'Contents' in page:
      for obj in page['Contents']:
        objects.append(obj)
  return objects

def should_skip_s3_key(s3_key):
#---------------------------------
  for substring in OMIT_S2AFF:
    if substring in s3_key:
      return True
  return False

def download_file(bucket, s3_key, local_path):
#---------------------------------
  s3 = get_s3_client()
  local_dir = os.path.dirname(local_path)
  try:
    os.makedirs(local_dir, exist_ok=True)
  except Exception as e:
    print(f'Failed to create directory {local_dir}: {e}')
  try:
    print(f'Downloading {s3_key} to {local_path}')
    s3.download_file(bucket, s3_key, local_path)
  except Exception as e:
    print(f'Failed to download {s3_key} to {local_path}: {e}')

def sync_s3_to_local(bucket, prefix, local_dir):
#---------------------------------
  objects = list_s3_objects(bucket, prefix)
  for obj in objects:
    s3_key = obj['Key']
    relative_path = s3_key[len(prefix):].lstrip('/')  # Ensure no leading slash
    local_path = os.path.join(local_dir, relative_path)
    # Check if local file exists or if S3 key should be skipped
    if os.path.exists(local_path) or should_skip_s3_key(s3_key):
      print(f'Skipping {s3_key}')
      continue
    try:
      download_file(bucket, s3_key, local_path)
    except Exception as e:
      print(f'Failed to sync {s3_key} to {local_path}: {e}')

def download_s2aff_data():
#---------------------------------
  bucket_name = 'ai2-s2-research-public'
  prefix = 's2aff-release'
  s2aff_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH))
  s2aff_data_directory = f'{s2aff_directory}/data'
  sync_s3_to_local(bucket_name, prefix, s2aff_data_directory)


#-------------------- MAIN --------------------

if __name__ == '__main__':
  download_s2aff_data()

