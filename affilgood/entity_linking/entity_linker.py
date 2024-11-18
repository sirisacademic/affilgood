# entity_linker.py
#import download_s2aff
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

# Whether to process a few for testing.
TEST_EL = False
# S2AFF root path relative to AffilGoodEL
S2AFF_PATH = 'S2AFF'
# ROR dump link.
ROR_DUMP_LINK = 'https://zenodo.org/api/records/?communities=ror-data&sort=mostrecent'
# ROR dump path relative to AffilGoodEL. Leave empty to download the latest ROR dump.
ROR_DUMP_PATH = ''
# Update OpenAlex ROR counts if current file older than this number of days.
UPDATE_OPENALEX_WORK_COUNTS_OLDER_THAN = 7
# Substring of URLs to omit when downloading S2AFF data
OMIT_S2AFF = ['ner_model', 'training', 'ror-data.json']
# Chunk size.
CHUNK_SIZE_EL = 1000
# Max. parallel processes.
MAX_PARALLEL_EL = 20
# Save output by chunks.
SAVE_CHUNKS_EL = True
# This score is used if we want to run a system keeping first-stage candidates (NOT USED now).
THRESHOLD_SCORE_FIRSTSTAGE_EL = 0.65
# This score is used to determine which candidates are passed to the re-ranking stage. Now considering all positive values.
THRESHOLD_SCORE_FILTER_FIRSTAGE_EL = 0
# This score is used to determine which candidates are kept after the re-ranking stage.
THRESHOLD_SCORE_RERANKED_EL = 0.25
#THRESHOLD_SCORE_RERANKED_EL = 0.15
NUM_CANDIDATES_TO_RERANK = 10
# Names of EL output column for labels.
COL_PREDICTIONS_EL = 'predicted_label'
COL_POTENTIAL_ERROR_NER = 'potential_error_ner'

# Names of EL output columns for labels with scores.
COL_PREDICTIONS_SCORES_EL = 'predicted_label_score'
## ROR dictionary fields
ROR_NAME_FIELD = 'name'
ROR_ID_FIELD = 'id'

# NER entity labels
SUBORG_NER_LABEL = 'SUBORG'
MAINORG_NER_LABEL = 'ORG'
CITY_NER_LABEL = 'CITY'
COUNTRY_NER_LABEL = 'COUNTRY'
REGION_NER_LABEL = 'REGION'

# NER entity fields
NER_ENTITY_TYPE_FIELD = 'entity_group'
NER_ENTITY_TEXT_FIELD = 'word'
NER_ENTITY_SCORE_FIELD = 'score'
NER_ENTITY_START_FIELD = 'start'
NER_ENTITY_END_FIELD = 'end'
MIN_LENGTH_NER_ENTITY = 2
IGNORE_NER_ENTITY_PREFIX = '##'
COL_NER_ENTITIES = 'ner_raw'

S2AFF_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH))
sys.path.insert(0, S2AFF_ABS_PATH)
from s2aff.consts import PATHS
# Make sure that S2AFF_PATH is in the sys.path or add it if necessary.
from s2aff.ror import RORIndex
from s2aff.model import PairwiseRORLightGBMReranker
#ror_index = RORIndex()
# df_linked = process_chunk_el(ror_index, pairwise_model, df_ner)
from IPython.display import clear_output
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class EntityLinker:
    def __init__(self, method = ['S2AFF','LLM','ElasticSearch'], device="cpu"):
        # Initialize linking model (e.g., ElasticSearch index, S2AFF model, etc.)
        self.method = method
        self.CACHED_PREDICTED_ID = {}
        self.CACHED_PREDICTED_ID_SCORE = {}
        self.title_case = False

    def load_linker(self, model, device):
        # Load or initialize the entity linking model
        return None  # Placeholder

    def link_entities(self, ner_output):
        # Link NER entities to identifiers (e.g., ROR identifiers)
        linked_entities = [self.model(entity) for entity in ner_output]
        return linked_entities
    
    def get_local_ror_dumps(self, path_data):
        #---------------------------------
        # get a list of all of the json files in the directory that that have the text 'ror-data' in them
        # and sort it in order of most recent version to least recent version
        json_files = [os.path.join(path_data, f) for f in os.listdir(path_data) 
                    if os.path.isfile(os.path.join(path_data, f)) and "ror-data.json" in f]
        json_files.sort(reverse=True)
        return json_files
    
    def get_latest_ror(self):
        #---------------------------------
        ROR_DUMP_FILE = None
        path_data = f'{S2AFF_ABS_PATH}/data'
        local_ror_dumps = self.get_local_ror_dumps(path_data)
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
                    local_ror_dumps = self.get_local_ror_dumps(path_data)
        except:
            pass
        if len(local_ror_dumps) > 0:
            ROR_DUMP_FILE = local_ror_dumps[0]
        return f'{ROR_DUMP_FILE}'
    
    def update_openalex_works_counts(self,):
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

    def ignore_entity(self, entity):
        """
        Entities predicted by the NER that are to be ignored.
        """
        if not isinstance(entity, dict):
            #print(f"Invalid entity format: {entity}")
            return True  # Ignore invalid entities
        
        ignore = len(entity[NER_ENTITY_TEXT_FIELD]) < MIN_LENGTH_NER_ENTITY\
            or entity[NER_ENTITY_TEXT_FIELD].startswith(IGNORE_NER_ENTITY_PREFIX) #entity[NER_ENTITY_SCORE_FIELD] < THRESHOLD_SCORE_NER\ or
        return ignore
    

    def get_first_entity(self, entities, entity_type, search_right=True):
        """
        Find the first entity of a certain type in a list of entities, starting from a given position.
        Used, for instance, to find the first ORG to the right or left of a SUBORG.
        """
        if search_right:
            for pos, entity in enumerate(entities):
                if entity[NER_ENTITY_TYPE_FIELD] == entity_type and not self.ignore_entity(entity):
                    return entity, pos
        else:
            for pos, entity in enumerate(reversed(entities)):
                if entity[NER_ENTITY_TYPE_FIELD] == entity_type and not self.ignore_entity(entity):
                    return entity, len(entities) - pos - 1
        return None, 0
    
    def get_entity_groupings(self, entities):
        """
        Generate groupings from the NER output, used to generate candidate affiliations to be linked with S2AFF.
        """
        entity_groupings = []
        for pos, entity in enumerate(entities):
            if not self.ignore_entity(entity):
                parsed_entity = {}
                search_pos = pos + 1
                if entity[NER_ENTITY_TYPE_FIELD] == SUBORG_NER_LABEL:
                    # If it is a sub-organization, get parent organization -> first ORG found to the right.
                    parent, pos_parent = self.get_first_entity(entities[search_pos:], MAINORG_NER_LABEL, search_right=True)
                    if parent:
                        search_pos += pos_parent + 1
                    else:
                        # If no parent found to the right, get the first ORG to the left.
                        parent, pos_parent = self.get_first_entity(entities[:pos], MAINORG_NER_LABEL, search_right=False)
                        if parent:
                            parsed_entity[SUBORG_NER_LABEL] = entity[NER_ENTITY_TEXT_FIELD].strip()
                            parsed_entity[MAINORG_NER_LABEL] = parent[NER_ENTITY_TEXT_FIELD].strip()
                        else:
                            # If no parent is found we ignore it as it is most likely an error.
                            continue
                elif entity[NER_ENTITY_TYPE_FIELD] == MAINORG_NER_LABEL:
                    parsed_entity[MAINORG_NER_LABEL] = entity[NER_ENTITY_TEXT_FIELD].strip()
                if entity[NER_ENTITY_TYPE_FIELD] == SUBORG_NER_LABEL or entity[NER_ENTITY_TYPE_FIELD] == MAINORG_NER_LABEL:
                    # Look for city/region/country to the right of the MAINORG_NER_LABEL.
                    # City
                    city, pos_city = self.get_first_entity(entities[search_pos:], CITY_NER_LABEL)
                    if city:
                        search_pos += pos_city + 1
                        parsed_entity[CITY_NER_LABEL] = city[NER_ENTITY_TEXT_FIELD].strip()
                    # Region
                    region, pos_region = self.get_first_entity(entities[search_pos:], REGION_NER_LABEL)
                    if region:
                        search_pos += pos_region + 1
                        parsed_entity[REGION_NER_LABEL] = region[NER_ENTITY_TEXT_FIELD].strip()
                    # Country
                    country, pos_country = self.get_first_entity(entities[search_pos:], COUNTRY_NER_LABEL)
                    if country:
                        parsed_entity[COUNTRY_NER_LABEL] = country[NER_ENTITY_TEXT_FIELD].strip()
                if parsed_entity and parsed_entity not in entity_groupings:
                    entity_groupings.append(parsed_entity)
        return entity_groupings

    def get_el_input_organizations(self, grouped_entities, osm):
        """
        Get the input about organizations, locations and sub-organizations necessary for S2AFF.
        """
        organizations = []
        for group in grouped_entities:
            organization = {}
            # Groupings without a main organization are skipped.
            if MAINORG_NER_LABEL in group and group[MAINORG_NER_LABEL]:
                organization['main'] = group[MAINORG_NER_LABEL]
                # Get sub-organizations if present.
                organization['suborg'] = group[SUBORG_NER_LABEL] if SUBORG_NER_LABEL in group and group[SUBORG_NER_LABEL] else ''
                # Get city and country if present
                if osm == None:
                    location = [group[CITY_NER_LABEL]] if CITY_NER_LABEL in group and group[CITY_NER_LABEL] else []
                    if COUNTRY_NER_LABEL in group:
                        location.append(group[COUNTRY_NER_LABEL])
                    organization['location'] = ', '.join(location)
                if osm != None: 
                    organization['location'] = ', '.join([value for value in [osm.get('CITY', None), osm.get('PROVINCE', None), osm.get('STATE', None), osm.get('COUNTRY', None)] if value is not None])
                if organization not in organizations:
                    organizations.append(organization)
        return organizations
    
    def get_s2aff_single_prediction(self, ror_index, pairwise_model, organization):
        """
        Get S2AFF predictions for one input "organization" (main organization possibly with children).
        """
        predicted_id = None
        predicted_score = None
        # We combine the organization and one child organization if present to look it up in the cache.
        organization_string = ', '.join([organization['suborg'], organization['main'], organization['location']])
        if organization_string in self.CACHED_PREDICTED_ID:
            return self.CACHED_PREDICTED_ID[organization_string], self.CACHED_PREDICTED_ID_SCORE[organization_string]
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
                        self.CACHED_PREDICTED_ID[organization_string] = predicted_id
                        self.CACHED_PREDICTED_ID_SCORE[organization_string] = predicted_score
        return predicted_id, predicted_score
    
    def get_predicted_labels(self, ror_index, pairwise_model, organizations):
        #--------------------------------------------------
            predicted_names = {}
            predicted_scores = {}
            for organization in organizations:
                predicted_id, predicted_score = self.get_s2aff_single_prediction(ror_index, pairwise_model, organization)
                if predicted_id and predicted_id not in predicted_names and ROR_NAME_FIELD in ror_index.ror_dict[predicted_id]:
                    predicted_names[predicted_id] = ror_index.ror_dict[predicted_id][ROR_NAME_FIELD]
                if predicted_id and (predicted_id not in predicted_scores or predicted_scores[predicted_id] < predicted_score):
                    predicted_scores[predicted_id] = predicted_score
            predicted_names_ids = [f'{predicted_names[predicted_id]} {{{predicted_id}}}' for predicted_id in predicted_names]
            predicted_labels = '|'.join(predicted_names_ids)
            predicted_labels_scores = '|'.join([f'{predicted_names[predicted_id]} {{{predicted_id}}}:{predicted_scores[predicted_id]:.2f}' for predicted_id in predicted_names])
            return [predicted_labels, predicted_labels_scores]
    
    def process_chunk_el(self, chunk, ror_index, pairwise_model, output_file_path_chunks='', overwrite_existing=False):
            
            #output_path = output_file_path_chunks.format(f'{df_chunk.index[0]}_{df_chunk.index[-1]}')
            chunk_to_process = []
            chunk_index_map = []  # Track the position in text_list and span_entities index

            for idx, item in enumerate(chunk):
                span_entities = item.get("ner_raw", [])
                osm_entities = item.get("osm", [])
                for span_idx, tupla in enumerate(list(zip(span_entities,osm_entities))):
                    span, osm = tupla
                    # Clean and optionally apply title case to the span text
                    chunk_to_process.append((span,osm))
                    chunk_index_map.append((idx, span_idx))  # Record which item and span this belongs to

            processed_list = []
            for ner, osm in chunk_to_process:

                result = {}
                # Apply entity groupings
                result['grouped_entities'] = self.get_entity_groupings(ner)

                # Generate entity linking inputs
                result['el_input_organizations'] = self.get_el_input_organizations(result['grouped_entities'], osm)

                # Generate predictions
                predictions, scores = self.get_predicted_labels(
                    ror_index, pairwise_model, result['el_input_organizations']
                )
                result[COL_PREDICTIONS_EL] = predictions
                result[COL_PREDICTIONS_SCORES_EL] = scores

                # Add processed item to the list
                processed_list.append(result)
            
            results = [{"raw_text": item["raw_text"], "span_entities": item["span_entities"], "ner": item['ner'], "osm": item['osm'],"ror":[]} for item in chunk]

            for idx, ner in enumerate(chunk_to_process):
                # Map each output back to the corresponding text_list item and span_entities index
                entities = processed_list[idx]

                # Append ner entities for the current span to the correct entry in results
                item_idx, ror_idx = chunk_index_map[idx]
                # Ensure that each item in "ner" corresponds to each span in "span_entities"
                if len(results[item_idx]["ror"]) <= ror_idx:
                    results[item_idx]["ror"].append({})
                results[item_idx]["ror"][ror_idx] = entities[COL_PREDICTIONS_EL]

            return results





