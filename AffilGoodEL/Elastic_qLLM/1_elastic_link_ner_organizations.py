#!/usr/bin/env python3

import sys
sys.path.append('functions')

import ner
import os
import json
import country_converter as coco
import elastic as es
import pandas as pd
import tqdm
from functions.utils import get_stopwords, get_legal_entities
from config import ES_HOST_PORT, ES_AUTHENTICATION, ES_MAX_HITS
from config import DATA_DIR, DATASETS, INCLUDE_GOLD_ANNOTATIONS_IN_OUTPUT, SKIP_FILE_NAMES, VERBOSE_EL
from config import INCLUDE_GENERAL_QUERY, SORT_BY_MATCHING_SUBQUERIES, INPUT_FORMAT, ES_LINK_TEST, ES_LINK_TEST_SIZE
from elasticsearch import Elasticsearch

# Returned fields.
SCORE_FIELD_ELASTIC = 'score'
ROR_ID_FIELD_ELASTIC = 'ror_id'
ROR_NAME_FIELD_ELASTIC = 'ror_name'
CITY_FIELD_ELASTIC = 'city'
REGION_FIELD_ELASTIC = 'region'
COUNTRY_FIELD_ELASTIC = 'country_name'

### Functions

# Not using csv.writer because it has problems.
def write_output(header, output, output_file_path):
#-----------------------------------------
  df_output = pd.DataFrame(output, columns=header)
  df_output.to_csv(output_file_path, sep='\t', index=False)

# Move up those results obtained with more or more relevant matches.
def sort_results_by_matches(results, matches_queries):
#----------------------------------------------------
  length_results = [len(matches) for matches in matches_queries]
  final_results = [r for _, r in sorted(zip(length_results, results),\
                    key=lambda pair: pair[0], reverse=True)] 
  return final_results
  
# Generate subqueries with the organization name.  
def queries_organization_name(organization_name, is_parent=False):
#-------------------------------
  org_name_tokens = organization_name.split()
  # Exact match with the organization name
  # We also consider an exact match without the first word if it is a stopword ("The British Council = British Council").
  # ROR name
  org_name_queries = []
  if not is_parent:
    ror_name_phrase_query = es.match_query('match_phrase', 'ror_name', organization_name)
    ror_name_length_query = es.term_query('ror_name.length', len(org_name_tokens))
    ror_name_query = es.bool_query('must', [ror_name_phrase_query, ror_name_length_query], 'ror_name_query', 10)
    org_name_queries.append(ror_name_query)
  # All names
  base_field = 'parent' if is_parent else 'name'
  fuzziness = 2 if len(organization_name) > 10 else 0
  org_name_query = es.fuzzy_match_query(f'{base_field}.keyword', organization_name, fuzziness, 'org_keyword', 10)
  org_name_queries.append(org_name_query)
  if org_name_tokens[0].lower() in STOPWORDS:
    org_stripped_query = es.match_query('match', f'{base_field}.keyword', ' '.join(org_name_tokens[1:]), 'org_stripped_keyword', 5)
    org_name_queries.append(org_stripped_query)
  org_exact_query = es.bool_query('should', org_name_queries)
  # Other types of matches with the name
  org_partial_queries = []
  org_phrase_query = es.match_query('match_phrase', base_field, organization_name, 'org_phrase', 1.2)
  org_partial_queries.append(org_phrase_query)
  # If the organization name includes legal entity types we score higher if the name matches without them.
  if set(org_name_tokens) & set(LIST_LEGAL_TYPES):
    organization_name_shorten = ' '.join([token for token in org_name_tokens if token not in LIST_LEGAL_TYPES])
    org_shorten_shingles_query = es.match_query('match', f'{base_field}.shingles', organization_name_shorten, 'org_shorten_shingles')
    org_shorten_aprox_query = es.match_query('match', base_field, organization_name_shorten, 'org_shorten_aprox')
    org_partial_queries.append(org_shorten_shingles_query)
    org_partial_queries.append(org_shorten_aprox_query)
  org_name_tokens_no_stop = [token for token in org_name_tokens if token.lower() not in STOPWORDS]
  first_two_tokens = ' '.join(org_name_tokens_no_stop[:2])
  org_first_two_exact_phrase_query = es.match_query('match_phrase', base_field, first_two_tokens, 'org_first_two', 1.1)
  org_partial_queries.append(org_first_two_exact_phrase_query)
  org_full_shingles_query = es.match_query('match', f'{base_field}.shingles', organization_name, 'org_full_shingles', 1.1)
  org_full_aprox_query = es.match_query('match', base_field, organization_name, 'org_full_aprox')
  org_partial_queries.append(org_full_shingles_query)
  org_partial_queries.append(org_full_aprox_query)
  org_partial_query = es.bool_query('should', org_partial_queries)
  return org_exact_query, org_partial_query
 
############################### MAIN ###############################

# Elasticsearch client.
es_client = Elasticsearch(
    ES_HOST_PORT,
    http_auth=ES_AUTHENTICATION
)

# Create country-converter object to convert country codes to country names.
cc = coco.CountryConverter()

# Some stopwords common in organization names in multiple languages.
STOPWORDS = get_stopwords()

# Legal entity types (such as SRL, LTD, etc.) in different languages.
LIST_LEGAL_TYPES = get_legal_entities()

RETRIEVE_FIELDS_ELASTIC = [
  ROR_ID_FIELD_ELASTIC,
  ROR_NAME_FIELD_ELASTIC,
  CITY_FIELD_ELASTIC,
  REGION_FIELD_ELASTIC,
  COUNTRY_FIELD_ELASTIC
]

# Number of results to retrieve based on whether we are getting a final prediction.
MAX_RESULTS = ES_MAX_HITS

# Generate output header.
OUTPUT_HEADER = ['fidx', 'idx', 'orig_raw_affiliation_string', 'raw_affiliation_string']

if INCLUDE_GOLD_ANNOTATIONS_IN_OUTPUT:
  OUTPUT_HEADER = OUTPUT_HEADER + ['gold_label']

OUTPUT_HEADER = OUTPUT_HEADER + ['predicted_id', 'predicted_label', 'predicted_score', 'type']

for DATASET in DATASETS:

  SUBDIR_NER_OUTPUT = f'{DATA_DIR}/{DATASET}'

  NER_OUTPUT_PATH = f'input/{SUBDIR_NER_OUTPUT}'
  # NER output - input files for entity linking.
  NER_OUTPUT_FILES = os.listdir(NER_OUTPUT_PATH) 

  # Output path.
  OUTPUT_DIR = f'output/{SUBDIR_NER_OUTPUT}'
  OUTPUT_FILE_NAME = f'predictions_elastic_{SUBDIR_NER_OUTPUT.replace("/", "_").lower()}.tsv'
  if ES_LINK_TEST and ES_LINK_TEST_SIZE > 0:
    OUTPUT_FILE_NAME = f'test_{OUTPUT_FILE_NAME}'
  PATH_OUTPUT_EL = f'{OUTPUT_DIR}/{OUTPUT_FILE_NAME}'

  # Create output directory if it does not exist.
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

  # Read and process NER output files.
  grouped_entities = {}
  raw_affiliations = {}
  gold_labels = {}

  if not (ES_LINK_TEST and type(ES_LINK_TEST_SIZE)==int and ES_LINK_TEST_SIZE>0):
    ES_LINK_TEST_SIZE = 0
    
  # Group entities.
  for fidx, ner_output_file in enumerate(tqdm.tqdm(sorted(NER_OUTPUT_FILES), desc='Grouping NER entities')):
    if INPUT_FORMAT == 'json':
      if not ner_output_file.endswith('.json') or ner_output_file.split('/')[-1] in SKIP_FILE_NAMES:
        print(f'*** Skipped file {ner_output_file}')
        continue
      else:
        print(f'Processing {NER_OUTPUT_PATH}/{ner_output_file}')
        # Get NER entities from JSON files
        grouped_entities[fidx], raw_affiliations[fidx], gold_labels[fidx] = ner.get_ner_predictions_json(f'{NER_OUTPUT_PATH}/{ner_output_file}', ES_LINK_TEST_SIZE)
    elif INPUT_FORMAT == 'parquet':
      if not ner_output_file.endswith('.parquet') or ner_output_file.split('/')[-1] in SKIP_FILE_NAMES:
        print(f'*** Skipped file {ner_output_file}')
        continue
      else:
        print(f'Processing {NER_OUTPUT_PATH}/{ner_output_file}')
        # Get NER entities from Parquet files
        grouped_entities[fidx], raw_affiliations[fidx], gold_labels[fidx] = ner.get_ner_predictions_parquet(f'{NER_OUTPUT_PATH}/{ner_output_file}', ES_LINK_TEST_SIZE)

  # Output.
  output = []

  # Get linking candidates / predictions.
  for fidx in grouped_entities:
    for idx in tqdm.tqdm(grouped_entities[fidx], desc='Getting linking candidates'):
      # Entity groupings for affiliation idx in file fidx.
      results = []
      entity_groupings = grouped_entities[fidx][idx]
      if VERBOSE_EL:
        print()
        print(f'************* RAW: {raw_affiliations[fidx][idx]} ***********')
        if fidx in gold_labels and gold_labels[fidx]:
          print(f'GOLD ===> {gold_labels[fidx][idx]}\n')
      if not entity_groupings:
        out = [fidx, idx, raw_affiliations[fidx][idx], raw_affiliations[fidx][idx]]
        if INCLUDE_GOLD_ANNOTATIONS_IN_OUTPUT:
          if fidx in gold_labels and gold_labels[fidx]:
            out.append(gold_labels[fidx][idx])
          else:
            out.append('')
        out = out + ['', '', '', 'ORG']
        output.append(out)
      for entity_group in entity_groupings:
        affiliation = []
        if VERBOSE_EL:
          print(f'NER ===> {entity_group}')
        # We can consider matching subqueries in order to define different criteria.
        # For instance, we can require higher scores for matches resulting from queries that include a suborganization.
        search_all = []
        # ---------- Organization queries ----------
        org_query = ''
        if ner.SUBORG_LABEL in entity_group:
          # There is SUB. We know that there is also a parent otherwise we are discarding SUBs.
          INCLUDE_GENERAL_QUERY = False
          type_org = ner.SUBORG_LABEL
          suborg = entity_group[ner.SUBORG_LABEL].strip()
          parent = entity_group[ner.MAINORG_LABEL].strip()
          if suborg and parent:
            affiliation.extend([suborg, parent])
            suborg_exact_query, suborg_partial_query = queries_organization_name(suborg, is_parent=False)
            parent_exact_query, parent_partial_query = queries_organization_name(parent, is_parent=True)
            suborg_query = suborg_exact_query
            parent_query = es.bool_query('should', [parent_exact_query, parent_partial_query])
            org_query = es.bool_query('should', [suborg_query, parent_query])
        elif ner.MAINORG_LABEL in entity_group:
          # There is no SUB, only ORG.
          type_org = ner.MAINORG_LABEL
          organization = entity_group[ner.MAINORG_LABEL].strip()
          if organization:
            affiliation.append(organization)
            search_all.append(organization)
            org_exact_query, org_partial_query = queries_organization_name(organization, is_parent=False)
            org_query = es.bool_query('should', [org_exact_query, org_partial_query])
        if org_query:
          # ---------- Additional queries including the location ----------
          additional_queries = []
          # Sub-queries involving geographical entities.
          geo_queries = []
          location = []
          # --- Country ---
          country = entity_group[ner.COUNTRY_LABEL] if ner.COUNTRY_LABEL in entity_group else ''      
          is_country_name = country and len(country) > 3
          if country:
            # Now retrieving country name from code if necessary.
            country_name = country
            if not is_country_name:
              try:
                country_name = cc.convert(country, to='name_short')
                is_country_name = True
              except:
                pass
            location.append(country_name)
            search_all.append(country_name)
            country_query = es.match_query('match', 'country', country_name, 'country')
            geo_queries.append(country_query)
          # We are only including the city/region if the country name is found.
          # Otherwise spurious matchings can be produced in some cases.
          # --- Region ---
          region = entity_group[ner.REGION_LABEL] if ner.REGION_LABEL in entity_group else ''      
          if is_country_name and region:
            location = [region] + location
            search_all.append(region)
            region_query = es.match_query('match', 'region', region, 'region')
            geo_queries.append(region_query)
          # --- City ---
          city = entity_group[ner.CITY_LABEL] if ner.CITY_LABEL in entity_group else ''      
          if is_country_name and city:
            location = [city] + location
            search_all.append(city)
            city_query = es.match_query('match', 'city', city, 'city')
            geo_queries.append(city_query)
          # --- Location query ---
          if location:
            location_query = es.match_query('match', 'location', ', '.join(location), 'location', 1.5)
            geo_queries.append(location_query)
          # --- Combined geographical queries ---
          if geo_queries:
            geo_query = es.bool_query('should', geo_queries)
            additional_queries.append(geo_query)
          # Add locations to affiliation string.
          affiliation.extend(location)
          # ---------- General query with all entities ----------
          # Only if we are retrieving a set of candidates and it is not the final prediction.
          if INCLUDE_GENERAL_QUERY:
            all_entities = ', '.join(search_all)
            all_query = es.match_query('match', 'all_search', all_entities, 'all')
            additional_queries.append(all_query)
            additional_query = es.bool_query('should', additional_queries)
          else:
            additional_query = geo_query
          # ---------- Final query ----------
          if additional_queries:
            final_query = es.bool_query('must', [org_query, additional_query]) 
          else:
            final_query = org_query
          # ---------- Execute query ----------
          results, matches_queries = es.query_elastic(es_client, final_query, es.ES_INDEX, num_results=MAX_RESULTS, fields=RETRIEVE_FIELDS_ELASTIC)
          if SORT_BY_MATCHING_SUBQUERIES:
            results = sort_results_by_matches(results, matches_queries)
        # ---------- Process results and generate output ----------
        if results:
          predicted_ids = []
          predicted_labels = []
          predicted_scores = []
          for prediction in results:
            retrieved_locations = []
            retrieved_locations.extend([prediction[CITY_FIELD_ELASTIC][0]] if prediction[CITY_FIELD_ELASTIC] else [])
            retrieved_locations.extend([prediction[COUNTRY_FIELD_ELASTIC][0]] if prediction[COUNTRY_FIELD_ELASTIC] else [])
            predicted_ids.append(prediction[ROR_ID_FIELD_ELASTIC])
            predicted_labels.append(f'{prediction[ROR_NAME_FIELD_ELASTIC]} {retrieved_locations} {{{prediction[ROR_ID_FIELD_ELASTIC]}}}')
            predicted_scores.append(f'{prediction[SCORE_FIELD_ELASTIC]:.2f}')
          predicted_id = '|'.join(predicted_ids)
          predicted_label = '|'.join(predicted_labels)
          predicted_score = '|'.join(predicted_scores)
        else:
          predicted_id = ''
          predicted_label = ''
          predicted_score = ''
        # Generate output.
        out = [fidx, idx, raw_affiliations[fidx][idx], ', '.join(affiliation)]
        if INCLUDE_GOLD_ANNOTATIONS_IN_OUTPUT:
          if fidx in gold_labels and gold_labels[fidx]:
            out.append(gold_labels[fidx][idx])
          else:
            out.append('')
        out = out + [predicted_id, predicted_label, predicted_score, type_org]
        output.append(out)
        if VERBOSE_EL:
          print(f'PRED ===> {predicted_label}')
          print(f'SCORE ===> {predicted_score}')
          print()
          #print(out)
          #print(' | '.join(out))
          #print(matches_queries)
          #print(str(final_query).replace("'", '"'))
      # End original affiliation string.
      if VERBOSE_EL:
        print('************************************************************')
        print()
              
  # Write output file. 
  print(f'Saving output to file {PATH_OUTPUT_EL}')   
  write_output(OUTPUT_HEADER, output, f'{PATH_OUTPUT_EL}')








