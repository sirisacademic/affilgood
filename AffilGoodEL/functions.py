############################################
### AffilGoodEL/functions.py
### Functions entity linking

import re
import os
import sys
import requests
import zipfile
import json
import boto3
import gzip
import pandas as pd
from tqdm import tqdm

from io import BytesIO
from botocore import UNSIGNED
from botocore.config import Config

from AffilGoodEL.config import *
from AffilGoodNER.config import *
from utils.functions import *

S2AFF_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH))
sys.path.insert(0, S2AFF_ABS_PATH)
from s2aff.consts import PATHS

CACHED_PREDICTED_ID = {}
CACHED_PREDICTED_ID_SCORE = {}

# To track progress in applying function to dataframe.
tqdm.pandas()

# Get currently existing latest ROR dump, if any.
def get_local_ror_dumps(path_data):
#---------------------------------
  # get a list of all of the json files in the directory that that have the text 'ror-data' in them
  # and sort it in order of most recent version to least recent version
  json_files = [os.path.join(path_data, f) for f in os.listdir(path_data) 
              if os.path.isfile(os.path.join(path_data, f)) and "ror-data.json" in f]
  json_files.sort(reverse=True)
  return json_files

# Code based on S2AFF/s2aff/data/download_latest_ror.py
def get_latest_ror():
#---------------------------------
  ROR_DUMP_FILE = None
  path_data = f'{S2AFF_ABS_PATH}/data'
  local_ror_dumps = get_local_ror_dumps(path_data)
  # Make a GET request to the Zenodo API endpoint
  print(f'Retrieving latest ROR dump from {ROR_DUMP_LINK}')
  try:
    response = requests.get(ROR_DUMP_LINK)
    response.raise_for_status()
    # Get the download URL of the most recent ROR record
    download_url = response.json()["hits"]["hits"][0]["files"][0]["links"]["self"]
    file_name = response.json()["hits"]["hits"][0]["files"][0]["key"]
    file_path = f'{path_data}/{file_name}'
    if os.path.exists(file_path):
      print(f'Latest ROR dump {file_path} already exists. Skipping download.')
    else:
      # Download the record
      print(f'Downloading ROR dump from {download_url} to {file_path}')
      response = requests.get(download_url) 
      response.raise_for_status()
      with open(file_path, "wb") as f:
        f.write(response.content)
      if os.path.exists(file_path):
        print(f'Extracting ROR dump from {file_path}')
        # unzip the file_name and delete the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
          zip_ref.extractall(path=path_data)
        #os.remove(file_path)
        local_ror_dumps = get_local_ror_dumps(path_data)
  except:
    pass
  if len(local_ror_dumps) > 0:
    ROR_DUMP_FILE = local_ror_dumps[0]
  return f'{ROR_DUMP_FILE}'

# Code based on S2AFF/scripts/update_openalex_works_counts.py
def update_openalex_works_counts():
#----------------------------------
  """
  Go through every single file from s3://openalex/data/institutions/updated_date=*/part-***.gz
  Stream the jsonl files inside there
  """
  bucket = "openalex"
  prefix = "data/institutions/"
  s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
  paginator = s3.get_paginator("list_objects_v2")
  pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
  works_count_dict = {}
  for page in pages:
    for obj in page["Contents"]:
      print(obj)
      if obj["Key"].endswith(".gz"):
        print("Working on", obj["Key"])
        obj = s3.get_object(Bucket=bucket, Key=obj["Key"])
        with gzip.GzipFile(fileobj=BytesIO(obj["Body"].read())) as f:
          for line in f:
            line = json.loads(line)
            ror_id = line["ror"]
            works_count = line["works_count"]
            works_count_dict[ror_id] = works_count
  # convert works_count_dict to a dataframe
  df = pd.DataFrame.from_dict(works_count_dict, orient="index", columns=["works_count"]).reset_index()
  df.columns = ["ror", "works_count"]
  df.to_csv(PATHS["openalex_works_counts"], index=False)

# Entities predicted by the NER that are to be ignored.
def ignore_entity(entity):
#------------------------
  ignore = entity[NER_ENTITY_SCORE_FIELD] < THRESHOLD_SCORE_NER\
    or len(entity[NER_ENTITY_TEXT_FIELD]) < MIN_LENGTH_NER_ENTITY\
    or entity[NER_ENTITY_TEXT_FIELD].startswith(IGNORE_NER_ENTITY_PREFIX)
  return ignore

# Find the first entity of a certain type in a list of entities.
# Used, for instance, to find the first ORG to the right of a SUBORG.
def get_first_entity(entities, entity_type):
#------------------------------------------
  for pos, entity in enumerate(entities):
    if entity[NER_ENTITY_TYPE_FIELD] == entity_type and not ignore_entity(entity):
      return entity, pos
  return None, 0

# Generate groupings from the NER output, used to generate candidate affiliations to be linked with S2AFF.
def get_entity_groupings(entities):
#---------------------------------
  entity_groupings = []
  for pos, entity in enumerate(entities):
    if not ignore_entity(entity):
      parsed_entity = {}
      search_pos = pos+1
      if entity[NER_ENTITY_TYPE_FIELD]==SUBORG_NER_LABEL:
        # If it is a sub-organization, get parent organization -> first ORG found to the right.
        # If no parent found, the entity is ignored as it is most probably mis-predicted.
        # (See, for instance, "Univ. Grenoble Alpes, CEA, LITEN, DTS, LSEI, F-38000, Grenoble, France")
        # Note: It could be the case for French affilations that there are several parents preceeding it.
        # In general, those are not institutions that are included in ROR (to be confirmed).
        parent, pos_parent = get_first_entity(entities[search_pos:], MAINORG_NER_LABEL)
        if parent:
          search_pos += pos_parent + 1
          parsed_entity[SUBORG_NER_LABEL] = entity[NER_ENTITY_TEXT_FIELD].strip()
          parsed_entity[MAINORG_NER_LABEL] = parent[NER_ENTITY_TEXT_FIELD].strip()
        else:
          continue
      elif entity[NER_ENTITY_TYPE_FIELD]==MAINORG_NER_LABEL:
        parsed_entity[MAINORG_NER_LABEL] = entity[NER_ENTITY_TEXT_FIELD].strip()
      if entity[NER_ENTITY_TYPE_FIELD]==SUBORG_NER_LABEL or entity[NER_ENTITY_TYPE_FIELD]==MAINORG_NER_LABEL:
        # Look for city/region/country to the right of the MAINORG_NER_LABEL.
        # City
        city, pos_city = get_first_entity(entities[search_pos:], CITY_NER_LABEL)
        if city:
          search_pos += pos_city + 1
          parsed_entity[CITY_NER_LABEL] = city[NER_ENTITY_TEXT_FIELD].strip()
        # Region
        region, pos_region = get_first_entity(entities[search_pos:], REGION_NER_LABEL)
        if region:
          search_pos += pos_region + 1
          parsed_entity[REGION_NER_LABEL] = region[NER_ENTITY_TEXT_FIELD].strip()
        # Country
        country, pos_country = get_first_entity(entities[search_pos:], COUNTRY_NER_LABEL)
        if country:
          parsed_entity[COUNTRY_NER_LABEL] = country[NER_ENTITY_TEXT_FIELD].strip()
      if parsed_entity and parsed_entity not in entity_groupings:
        entity_groupings.append(parsed_entity)
  return entity_groupings

# Get the input about organizations, locations and sub-organizations necessary for S2AFF.
def get_el_input_organizations(grouped_entities):
#------------------------------------------------
  organizations = []
  for group in grouped_entities:
    organization = {}
    # Groupings without a main organization are skipped.
    if MAINORG_NER_LABEL in group and group[MAINORG_NER_LABEL]:
      organization['main'] = group[MAINORG_NER_LABEL]
      # Get sub-organizations if present.
      organization['suborg'] = group[SUBORG_NER_LABEL] if SUBORG_NER_LABEL in group and group[SUBORG_NER_LABEL] else ''
      # Get city and country if present
      location = [group[CITY_NER_LABEL]] if CITY_NER_LABEL in group and group[CITY_NER_LABEL] else []
      if COUNTRY_NER_LABEL in group:
        location.append(group[COUNTRY_NER_LABEL])
      organization['location'] = ', '.join(location)
      if organization not in organizations:
        organizations.append(organization)
  return organizations


# Get S2AFF predictions for one input "organization" (main organization possibly with children).
def get_s2aff_single_prediction(ror_index, pairwise_model, organization):
#---------------------------------------
  predicted_id = None
  predicted_score = None
  # We combine the organization and one child organization if present to look it up in the cache.
  organization_string = ', '.join([organization['suborg'], organization['main'], organization['location']])
  if organization_string in CACHED_PREDICTED_ID:
    return CACHED_PREDICTED_ID[organization_string], CACHED_PREDICTED_ID_SCORE[organization_string]
  else:
    first_stage_candidates, first_stage_scores = ror_index.get_candidates_from_main_affiliation(
                                                            organization['main'],
                                                            organization['location'],
                                                            [organization['suborg']])
    # Discard all candidates/scores below the minimum score. When know scores is a descending sorted list.
    len_filtered_scores = len([s for s in first_stage_scores if s >= THRESHOLD_SCORE_FILTER_FIRSTAGE_EL])
    candidates = first_stage_candidates[:len_filtered_scores]
    scores = first_stage_scores[:len_filtered_scores]
    if candidates:
      reranked_candidates, reranked_scores = pairwise_model.predict(
                                                organization_string,
                                                candidates[:NUM_CANDIDATES_TO_RERANK],
                                                scores[:NUM_CANDIDATES_TO_RERANK])
      # Get the top-ranked one if above score.
      top_rr_score = reranked_scores[0]
      if top_rr_score >= THRESHOLD_SCORE_RERANKED_EL:
        top_rr_ror_idx = reranked_candidates[0]
        if top_rr_ror_idx in ror_index.ror_dict and ROR_ID_FIELD in ror_index.ror_dict[top_rr_ror_idx]:
          predicted_id = ror_index.ror_dict[top_rr_ror_idx][ROR_ID_FIELD]
          predicted_score = top_rr_score
          CACHED_PREDICTED_ID[organization_string] = predicted_id
          CACHED_PREDICTED_ID_SCORE[organization_string] = predicted_score
  return predicted_id, predicted_score
        

def get_predicted_labels(ror_index, pairwise_model, organizations):
#--------------------------------------------------
  predicted_names = {}
  predicted_scores = {}
  for organization in organizations:
    predicted_id, predicted_score = get_s2aff_single_prediction(ror_index, pairwise_model, organization)
    if predicted_id and predicted_id not in predicted_names and ROR_NAME_FIELD in ror_index.ror_dict[predicted_id]:
      predicted_names[predicted_id] = ror_index.ror_dict[predicted_id][ROR_NAME_FIELD]
    if predicted_id and (predicted_id not in predicted_scores or predicted_scores[predicted_id] < predicted_score):
      predicted_scores[predicted_id] = predicted_score
  predicted_names_ids = [f'{predicted_names[predicted_id]} {{{predicted_id}}}' for predicted_id in predicted_names]
  predicted_labels = '|'.join(predicted_names_ids)
  predicted_labels_scores = '|'.join([f'{predicted_names[predicted_id]} {{{predicted_id}}}:{predicted_scores[predicted_id]:.2f}' for predicted_id in predicted_names])
  return [predicted_labels, predicted_labels_scores]


def process_chunk_el(ror_index, pairwise_model, df_chunk, output_file_path_chunks='', overwrite_existing=False):
#---------------------------------------------------------
  output_path = output_file_path_chunks.format(f'{df_chunk.index[0]}_{df_chunk.index[-1]}')
  if os. path. exists(output_path) and not overwrite_existing:
    print(f'Skipping existing output {output_path} since OVERWRITE_FILES_EL=False')
    df_chunk = read_file(output_path)
  else:
    df_chunk['grouped_entities'] = df_chunk.apply(lambda row: get_entity_groupings(row[COL_NER_ENTITIES]), axis=1)
    df_chunk['el_input_organizations'] = df_chunk.apply(lambda row: get_el_input_organizations(row['grouped_entities']), axis=1)
    df_chunk[[COL_PREDICTIONS_EL, COL_PREDICTIONS_SCORES_EL]] =\
          df_chunk.progress_apply(lambda row: get_predicted_labels(ror_index, pairwise_model, row['el_input_organizations']), axis=1, result_type='expand')
    drop_columns = [COL_NER_ENTITIES, 'grouped_entities', 'el_input_organizations']
    if COL_POTENTIAL_ERROR_NER in df_chunk:
      drop_columns.append(COL_POTENTIAL_ERROR_NER)
    df_chunk = df_chunk.drop(columns=drop_columns)
    if output_file_path_chunks:
      write_output(output_path, df_chunk)
  return df_chunk
  
  
  
  
  
  
  
  


  
