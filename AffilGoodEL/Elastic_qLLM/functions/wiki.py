import sys
sys.path.insert(0, '..')

from config import ROOT_PROJECT, WIKIDATA_USER_AGENT, FILE_ROR_ID_WIKI_ID_LABEL_LANG
import requests
import time
import csv
import os

NO_WIKI_ID = ''
ROR_ID_URL = 'https://ror.org/'
WIKI_ID_URL = 'http://www.wikidata.org/wiki/'
WIKIDATA_PROP_ROR_ID = 'P6782'
WIKIDATA_SPARQL_ENDPOINT = 'https://query.wikidata.org/sparql'
SPARQL_QUERY_ID = """
  PREFIX wdt: <http://www.wikidata.org/prop/direct/>
  SELECT ?item WHERE {{
    ?item wdt:{} "{}".
  }}
"""

SPARQL_QUERY_LABELS = """
  SELECT ?langCode ?label
  WHERE {{
    VALUES ?entity {{ wd:{} }}  
    {{?entity rdfs:label ?label}}
    UNION
    {{?entity skos:altLabel ?label}}
    BIND(LANG(?label) AS ?langCode)
  }}
"""
 
WIKIDATA_CACHED_IDS = {}
WIKIDATA_CACHED_LABELS = {}

MAX_RETRIES = 5

HEADERS = {'User-Agent': WIKIDATA_USER_AGENT}

def load_wikidata_caches_from_dump():
#-----------------------------------
  if FILE_ROR_ID_WIKI_ID_LABEL_LANG:
    # Open and read the TSV file
    with open(f'{ROOT_PROJECT}/{FILE_ROR_ID_WIKI_ID_LABEL_LANG}', newline='', encoding='utf-8') as tsvfile:
      reader = csv.DictReader(tsvfile, delimiter='\t')  # use '\t' as the delimiter
      # Iterate through each row in the TSV
      for row in reader:
        ror_id = row['ror_id']
        wiki_id = row['wiki_id']
        lang = row['lang']
        label = row['label']
        if lang and label:
          if wiki_id not in WIKIDATA_CACHED_LABELS:
            WIKIDATA_CACHED_LABELS[wiki_id] = {}
          if lang not in WIKIDATA_CACHED_LABELS[wiki_id]:
            WIKIDATA_CACHED_LABELS[wiki_id][lang] = []
          if label not in WIKIDATA_CACHED_LABELS[wiki_id][lang]:
            WIKIDATA_CACHED_LABELS[wiki_id][lang].append(label)
          if (WIKIDATA_PROP_ROR_ID, ror_id) not in WIKIDATA_CACHED_IDS:
            WIKIDATA_CACHED_IDS[(WIKIDATA_PROP_ROR_ID, ror_id)] = wiki_id

def query_wikidata_sparql(params, max_retries=MAX_RETRIES):
#---------------------------------------------------------
  retries = 0   
  while retries < max_retries:
    try:
      response = requests.get(WIKIDATA_SPARQL_ENDPOINT, headers=HEADERS, params=params)
      response.raise_for_status()  # Raise an exception for 4XX or 5XX status codes
      return response
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
      print(f'An error occurred: {e}')
      retries += 1
      print(f'Retrying... Attempt {retries} of {max_retries}')
      time.sleep(2**retries)  # Exponential backoff
  return None  # Return None if max retries reached without success


def get_wiki_id_by_property(property_id, property_value):
#-------------------------------------------------------
  if (property_id, property_value) in WIKIDATA_CACHED_IDS:
    wiki_id = WIKIDATA_CACHED_IDS[(property_id, property_value)]
  else:
    wiki_id = NO_WIKI_ID
    # Define parameters of query.
    params={
      'query': SPARQL_QUERY_ID.format(property_id, property_value),
      'format': 'json'
    }
    # Send the SPARQL query request.
    response = query_wikidata_sparql(params)
    if response and response.status_code == 200:
      data = response.json()
      if 'results' in data and 'bindings' in data['results'] and data['results']['bindings']:
        item = data['results']['bindings'][0]
        if item['item']['type'] == 'uri':
          wiki_id = item['item']['value'].replace('/entity/', '/wiki/')
    # Cache result.
    WIKIDATA_CACHED_IDS[(property_id, property_value)] = wiki_id
  return wiki_id.replace(WIKI_ID_URL, '')
  
def get_wiki_id_by_ror_id(ror_id):
#--------------------------------  
  return get_wiki_id_by_property(WIKIDATA_PROP_ROR_ID, ror_id.replace(ROR_ID_URL, ''))
  
def get_wiki_labels(wiki_id, list_lang):
#--------------------------------------
  list_lang = [l.strip().lower() for l in list_lang]
  if wiki_id in WIKIDATA_CACHED_LABELS:
    wiki_labels = WIKIDATA_CACHED_LABELS[wiki_id]
  else:
    wiki_labels = {}
    # Define parameters of query.
    params={
      'query': SPARQL_QUERY_LABELS.format(wiki_id),
      'format': 'json'
    }
    # Send the SPARQL query request.
    response = query_wikidata_sparql(params)
    if response and response.status_code == 200:
      data = response.json()
      if 'results' in data and 'bindings' in data['results']:
        for item in data['results']['bindings']:
            lang_code = item['langCode']['value'].lower()
            label = item['label']['value']
            if lang_code not in wiki_labels:
              wiki_labels[lang_code] = []
            wiki_labels[lang_code].append(label)
    # Cache result.
    WIKIDATA_CACHED_LABELS[wiki_id] = wiki_labels
  wiki_labels_languages = {lang_code: wiki_labels[lang_code] for lang_code in wiki_labels if lang_code in list_lang}
  return wiki_labels_languages

# ========================== MAIN =========================

# Initialize WikiData caches.
load_wikidata_caches_from_dump()





  
