#!/usr/bin/env python3

import sys
sys.path.append('functions')

import torch
import json
import re
import ner
import codecs
import pandas as pd
import tqdm

from transformers import pipeline
from config import MODEL, QUANTISED_MODEL, MODEL_REVISION, MAX_NEW_TOKENS, TEMPERATURE, FORMAT_INPUT_OUTPUT
from config import DATA_DIR, DATASETS, VERBOSE_RERANKING, RERANK_TEST, RERANK_TEST_SIZE

#======================================== SETTINGS =======================================

# We consider Elasticsearch predictions for which the relation between the first and second scores is above this threshold as correct.
# And do not re-rank them.
THRESHOLD_RELATION_SCORES = 3

ROR_URL = 'https://ror.org/'
# Regular expression pattern to extract name, location, and ror_id
PATTERN_ORG = r'^(.*?) \[(.*?)\] \{(.*?)\}$'
# Pattern to extract the response.
PATTERN_RESPONSE_TEXT = r'\(\s*\"(.+?)\"\s*,\s*\"(.+?)\"\s*\)'

# Only the columns present are considered.
OUTPUT_COLUMNS = ['fidx', 'idx', 'orig_raw_affiliation_string', 'raw_affiliation_string']

"""Model settings"""

MODEL = 'TheBloke/neural-chat-7B-v3-2-GPTQ'
QUANTISED_MODEL = True
MODEL_REVISION = 'main'

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.1

### TODO: Move prompts to another file.
#========================================= PROMPTS =======================================

TASK_DESCRIPTION_TEXT = """
You are assistant helping to disambiguate research organization names.

You are provided with affiliation strings obtained from research publications and a list of candidate organizations.
Candidate organizations can include the organization name and location and are identified by their ROR IDs.
Your task is to evaluate these inputs and determine the most likely match between each affiliation string and a candidate organization.
When assessing the likeliness, consider that affiliation strings and candidates might be in different languages and there might be errors or omissions in the affiliation strings and/or the candidate organizations' names or locations.
When in doubt, select the most specific institution.

The output should be a tuple with the format: (most_likely_organization, most_likely_ror)

VERY IMPORTANT: Please output only the tuple WITHOUT ANY EXPLANATION.

If none of the candidates is a likely match the returned tuple should be ("None", "None")

Please consider the following three examples that show the user input and your expected response.

* Example 1:

Affiliation: Whitehead Institute for Biomedical Research, MIT, MA
Candidates:
- Massachusetts Institute of Technology, Cambridge, United States (ROR: 01h3p4f56)
- Harvard University, Cambridge, MA, USA (ROR: 02t5q7k89)
- Whitehead Institute, Boston, US (ROR: 03z8s9r07)

Expected response:
("Whitehead Institute, Boston, US", "03z8s9r07")

* Example 2:

Affiliation: UNIVERSITÀ DI TORINO, TORINO, Italy
Candidates:
- A. O. Ordine Mauriziano di Torino, Turin, Italy (ROR: 03efxpx82)
- Turin Institute for the Deaf, Pianezza, Italy (ROR: 00f21ds03)
- Azienda Ospedaliera Citta' della Salute e della Scienza di Torino, Turin, Italy (ROR: 001f7a930)
- INFN Sezione di Torino, Turin, Italy (ROR: 01vj6ck58)
- Osservatorio Astrofisico di Torino, Pino Torinese, Italy (ROR: 00yrf4e35)

Expected response:
("None", "None")

* Example 3:

Affiliation: Institute of Chemistry, University of Graz, Graz, Austria
Candidates:
- Institute for European Tort Law, Vienna, 'Österreich (ROR: 00n0zsa28)
- Universitätsklinik für Frauenheilkunde und Geburtshilfe, Graz, Österreich (ROR: 02d9e4911)
- Institute for Interdisciplinary Studies of Austrian Universities, Vienna, Österreich (ROR: 001zf8v27)

Expected response:
("None", "None")

Please make sure that your response ONLY contains a tuple as shown above.
"""

#======================================== PIPELINE =======================================

if QUANTISED_MODEL:
  # For quantized models.
  PIPELINE = pipeline('text-generation', model=MODEL, revision=MODEL_REVISION, device_map='auto')
else:
  # For large models - use Accelerator.
  from accelerate import Accelerator
  accelerator = Accelerator()
  # Load the pipeline with manual device placement.
  PIPELINE = pipeline("text-generation", model=MODEL, revision=MODEL_REVISION, device=accelerator.device)
  PIPELINE.model, PIPELINE.tokenizer = accelerator.prepare(PIPELINE.model, PIPELINE.tokenizer)


#======================================= FUNCTIONS =======================================

# Function used to send prompts to the model.
def prompt_model_text(organizations_text):
#----------------------------------------
  prompt = f'<|system|>{TASK_DESCRIPTION_TEXT}\n<|user|>{organizations_text}\n<|assistant|>'
  outputs = PIPELINE(prompt, max_new_tokens=len(prompt)+MAX_NEW_TOKENS, temperature=TEMPERATURE, do_sample=True)
  output_without_prompt = outputs[0]['generated_text'].replace(prompt, '')
  return output_without_prompt

def rerank_text(organizations):
#-----------------------------
  response = prompt_model_text(organizations)
  matches = re.findall(PATTERN_RESPONSE_TEXT, response, re.DOTALL)
  return matches
  
def apply_get_rel_score(row):
#---------------------------
  if isinstance(row, list):
    return get_rel_score(row)
  else:
    return row

def get_rel_score(row):
#---------------------
  rel_score = THRESHOLD_RELATION_SCORES + 1
  try:
    if len(row)>1:
      return float(row[0])/float(row[1])
    else:
      return rel_score
  except:
    return rel_score

def extract_parts_label(label):
#-----------------------------
  match = re.match(PATTERN_ORG, label)
  name = match.group(1).strip() if match else ''
  ror_id = match.group(3).strip().replace(ROR_URL, '') if match else ''
  list_location = match.group(2).strip().split(',') if match else []
  location = ', '.join([loc.strip(" '") for loc in list_location])
  return name, location, ror_id


def handle_bad_lines(bad_line):
#-----------------------------
  print(f"Bad line: {bad_line}")
  return bad_line

#========================================= MAIN #=========================================

for DATASET in DATASETS:

  PATH_ROOT_OUTPUT = f'output/{DATA_DIR}/{DATASET}'

  PATH_OUTPUT_ELASTIC = f'{PATH_ROOT_OUTPUT}/predictions_elastic_{DATA_DIR}_{DATASET.lower()}.tsv'
  OUTPUT_FILE_NAME = f'reranked_predictions_elastic_{DATA_DIR}_{DATASET.lower()}.tsv'
  if RERANK_TEST and RERANK_TEST_SIZE > 0:
    OUTPUT_FILE_NAME = f'test_{OUTPUT_FILE_NAME}'
  PATH_OUTPUT_LLM = f'{PATH_ROOT_OUTPUT}/{OUTPUT_FILE_NAME}'

  reranked_organizations = []
  organizations = []
  processed = 0
  ### TODO: Output the content of errors.
  reprocess_output_with_errors = []
  errors = []

  # Read output from first step.
  print(f'Reading candidates from {PATH_OUTPUT_ELASTIC}')

  # Input dataframe.
  df_output_elastic = pd.read_csv(
    PATH_OUTPUT_ELASTIC,
    sep='\t',
    on_bad_lines=handle_bad_lines,
    encoding='utf-8',
    engine='python'
  )
  
  if RERANK_TEST and RERANK_TEST_SIZE > 0:
    df_output_elastic = df_output_elastic.head(RERANK_TEST_SIZE)
  
  df_output_elastic.fillna('', inplace=True)
  df_output_elastic['predicted_label_list'] = df_output_elastic['predicted_label'].str.split('|')
  # If we only have one float value in the predicted_score column we keep all rows.
  if df_output_elastic.dtypes.predicted_score == 'float64':
    df_orgs_to_process = df_output_elastic
  else:
    df_output_elastic['predicted_score_list'] = df_output_elastic['predicted_score'].str.split('|')
    df_output_elastic['diff_score'] = df_output_elastic['predicted_score_list'].apply(apply_get_rel_score)
    # Keep organizations to re-rank.
    df_orgs_to_process = df_output_elastic[df_output_elastic['diff_score'] < THRESHOLD_RELATION_SCORES]

  # Output dataframe.
  keep_columns = [col for col in OUTPUT_COLUMNS if col in df_orgs_to_process]

  df_output_llm = pd.DataFrame()
  df_output_llm[keep_columns] = df_orgs_to_process[keep_columns]

  # TODO: ADD fidx, idx for combining later!!!

  ### Process in TEXT format.
  if FORMAT_INPUT_OUTPUT == 'text':
    num_orgs_to_process = len(df_orgs_to_process)
    records = df_orgs_to_process.to_dict('records')
    for row in tqdm.tqdm(records, desc=f'Processing {num_orgs_to_process} organizations'):
      response = ''
      if len(row['predicted_label_list']) > 0:
        ror_candidates = {}
        processed += 1
        prompt = f'Affiliation: {row["raw_affiliation_string"]}\n'
        prompt += 'Candidates:\n'
        for predicted_label in row['predicted_label_list']:
          name, location, ror_id = extract_parts_label(predicted_label)
          prompt += f'- {name}, {location} (ROR: {ror_id})\n'
          ror_candidates[ror_id] = f'{name} ({location})'
        matches = rerank_text(prompt)
        for match in matches:
          if len(match) == 2:
            selected_ror_id = match[1]
            if selected_ror_id in ror_candidates or selected_ror_id == 'None':
              response = selected_ror_id
              if VERBOSE_RERANKING:
                selected_ror_name = ror_candidates[selected_ror_id] if selected_ror_id in ror_candidates else 'None'
                print(f'{row["raw_affiliation_string"]} ===> {selected_ror_name} ({selected_ror_id})')
                print()
            else:
              print(f'=> Returned ROR id {selected_ror_id} not present in source.')
              reprocess_output_with_errors.append(prompt)
              errors.append(match)
          else:
            print(f'=> Returned ill-formatted match: {match}')
            reprocess_output_with_errors.append(prompt)
            errors.append(match)
      reranked_organizations.append(response)

  ### Write final output.
  df_output_llm['ror_id'] = reranked_organizations

  print(f'Writing output to {PATH_OUTPUT_LLM}')
  df_output_llm.to_csv(PATH_OUTPUT_LLM, sep='\t', index=False)

print('=== Finished processing ===')

