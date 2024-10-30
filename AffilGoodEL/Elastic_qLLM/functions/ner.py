import sys
sys.path.insert(0, '..')

import json
import re
import pandas as pd
from config import FIELD_RAW_AFFILIATION_NER, FIELD_GOLD_ANNOTATIONS_NER, FIELD_PREDICTED_ENTITIES_NER

############################################
### NER parameters

# Now using the same threshold for all entity types.
THRESHOLD_SCORE_NER = 0.75
# NER entity labels
SUBORG_LABEL = 'SUB'
MAINORG_LABEL = 'ORG'
CITY_LABEL = 'CITY'
COUNTRY_LABEL = 'COUNTRY'
REGION_LABEL = 'REGION'
# NER entity fields
ENTITY_TYPE_FIELD = 'entity_group'
ENTITY_TEXT_FIELD = 'word'
ENTITY_SCORE_FIELD = 'score'
MIN_LENGTH_NER_ENTITY = 2
IGNORE_NER_ENTITY_PREFIX = '##'

############################################
### Functions

def find_replace_abbrevations(df_abbreviations, affiliation):
#-----------------------------------------------------------
  re_abbr = f'(?:^|[\s])({"|".join(df_abbreviations["abbreviation"].tolist())})(?:[;.,\s]|$)'
  span_abbr = re.search(re_abbr, affiliation)
  new_affiliation = None
  if span_abbr:
    replace = affiliation[span_abbr.start():span_abbr.end()]
    abbv = replace.strip(' -.,;()')
    replacement = df_abbreviations[df_abbreviations['abbreviation']==abbv]['full_name'].to_string(index=False).strip(' -.,;()')
    if not re.search(replacement, affiliation, re.IGNORECASE):
      new = replace.replace(abbv, replacement)
      new_affiliation = affiliation.replace(replace, new).replace('. ', ' ')
  return new_affiliation

# Entities predicted by the NER that are to be ignored.
def ignore_entity(entity):
#------------------------
  ignore = entity[ENTITY_SCORE_FIELD] < THRESHOLD_SCORE_NER\
    or len(entity[ENTITY_TEXT_FIELD]) < MIN_LENGTH_NER_ENTITY\
    or entity[ENTITY_TEXT_FIELD].startswith(IGNORE_NER_ENTITY_PREFIX)
  return ignore

# Find the first entity of a certain type in a list of entities.
# Used, for instance, to find the first ORG to the right of a SUBORG.
def get_first_entity(entities, entity_type):
#------------------------------------------
  for pos, entity in enumerate(entities):
    if entity[ENTITY_TYPE_FIELD] == entity_type and not ignore_entity(entity):
      return entity, pos
  return None, 0

# Get summary NER output to display.
def get_summary_ner(entities):
#----------------------------
  return '   '.join([f'[{entity[ENTITY_TEXT_FIELD]}][{entity[ENTITY_TYPE_FIELD]}:{entity[ENTITY_SCORE_FIELD]:.2f}]'\
                    for entity in entities if entity[ENTITY_SCORE_FIELD] > THRESHOLD_SCORE_NER])

# Generate groupings from the NER output, used to generate candidate affiliations to be linked with S2AFF.
def get_entity_groupings(entities):
#---------------------------------
  entity_groupings = []
  for pos, entity in enumerate(entities):
    if not ignore_entity(entity):
      parsed_entity = {}
      search_pos = pos+1
      if entity[ENTITY_TYPE_FIELD]==SUBORG_LABEL:
        # If it is a sub-organization, get parent organization -> first ORG found to the right.
        # If no parent found, the entity is ignored as it is most probably mis-predicted.
        # (See, for instance, "Univ. Grenoble Alpes, CEA, LITEN, DTS, LSEI, F-38000, Grenoble, France")
        # Note: It could be the case for French affilations that there are several parents preceeding it.
        # In general, those are not institutions that are included in ROR (to be confirmed).
        parent, pos_parent = get_first_entity(entities[search_pos:], MAINORG_LABEL)
        if parent:
          search_pos += pos_parent + 1
          parsed_entity[SUBORG_LABEL] = re.sub(r'[\n\r]', ' ', entity[ENTITY_TEXT_FIELD].strip())
          parsed_entity[MAINORG_LABEL] = re.sub(r'[\n\r]', ' ', parent[ENTITY_TEXT_FIELD].strip())
        else:
          continue
      elif entity[ENTITY_TYPE_FIELD]==MAINORG_LABEL:
        parsed_entity[MAINORG_LABEL] = re.sub(r'[\n\r]', ' ', entity[ENTITY_TEXT_FIELD].strip())
      if entity[ENTITY_TYPE_FIELD]==SUBORG_LABEL or entity[ENTITY_TYPE_FIELD]==MAINORG_LABEL:
        # Look for city/region/country to the right of the MAINORG_LABEL.
        # City
        city, pos_city = get_first_entity(entities[search_pos:], CITY_LABEL)
        if city:
          search_pos += pos_city + 1
          parsed_entity[CITY_LABEL] = re.sub(r'[\n\r]', ' ', city[ENTITY_TEXT_FIELD].strip())
        # Region
        region, pos_region = get_first_entity(entities[search_pos:], REGION_LABEL)
        if region:
          search_pos += pos_region + 1
          parsed_entity[REGION_LABEL] = re.sub(r'[\n\r]', ' ', region[ENTITY_TEXT_FIELD].strip())
        # Country
        country, pos_country = get_first_entity(entities[search_pos:], COUNTRY_LABEL)
        if country:
          parsed_entity[COUNTRY_LABEL] = re.sub(r'[\n\r]', ' ', country[ENTITY_TEXT_FIELD].strip())
      if parsed_entity and parsed_entity not in entity_groupings:
        entity_groupings.append(parsed_entity)
  return entity_groupings

# Generate groupings from NER predictions from a JSON file with NER output.
def get_ner_predictions_json(file_path, test_size=0):
#---------------------------------------
  raw_affiliations = {}
  gold_labels = {}
  grouped_entities = {}
  try:
    with open(file_path, 'r') as file_affiliations:
      ner_parsed_affiliations = json.load(file_affiliations)
    if test_size:
      keys_to_process = list(ner_parsed_affiliations.keys())[:test_size]
    else:
      keys_to_process = list(ner_parsed_affiliations.keys())
    # Process each parsed affiliation string
    for idx in keys_to_process:
      entities = ner_parsed_affiliations[idx].get(FIELD_PREDICTED_ENTITIES_NER, [])
      # Get groupings of entities that can form candidate affiliations
      grouped_entities[idx] = get_entity_groupings(entities)
      raw_affiliations[idx] = re.sub(r'[\n\r]', ' ', ner_parsed_affiliations[idx].get(FIELD_RAW_AFFILIATION_NER, '').strip())
      if FIELD_GOLD_ANNOTATIONS_NER and FIELD_GOLD_ANNOTATIONS_NER in ner_parsed_affiliations[idx]:
        gold_labels[idx] = ner_parsed_affiliations[idx][FIELD_GOLD_ANNOTATIONS_NER]
    return grouped_entities, raw_affiliations, gold_labels
  except FileNotFoundError:
    print(f"File {file_path} not found.")
    return None, None, None
  except json.JSONDecodeError:
    print(f"Error decoding JSON from file {file_path}.")
    return None, None, None
  except Exception as e:
    print(f"An error occurred: {e}")
    return None, None, None

# Generate groupings from NER predictions from a parquet file with NER output.
def get_ner_predictions_parquet(file_path, test_size=0):
#------------------------------------------
  raw_affiliations = {}
  gold_labels = {}
  grouped_entities = {}
  try:
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)
    if test_size:
      df = df.head(test_size)
    # Process each row in the DataFrame
    for idx, row in df.iterrows():
      entities = row.get(FIELD_PREDICTED_ENTITIES_NER, [])
      # Get groupings of entities that can form candidate affiliations
      grouped_entities[idx] = get_entity_groupings(entities)
      raw_affiliations[idx] = re.sub(r'[\n\r]', ' ', row.get(FIELD_RAW_AFFILIATION_NER, '').strip())
      if FIELD_GOLD_ANNOTATIONS_NER and FIELD_GOLD_ANNOTATIONS_NER in row:
        gold_labels[idx] = row[FIELD_GOLD_ANNOTATIONS_NER]
    return grouped_entities, raw_affiliations, gold_labels
  except FileNotFoundError:
    print(f"File {file_path} not found.")
    return None, None, None
  except ValueError as e:
    print(f"Error reading Parquet file {file_path}: {e}")
    return None, None, None
  except Exception as e:
    print(f"An error occurred: {e}")
    return None, None, None


