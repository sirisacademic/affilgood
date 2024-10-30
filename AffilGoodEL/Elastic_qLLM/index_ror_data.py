#!/usr/bin/env python3

import json
import functions.elastic as es
from config import ROOT_PROJECT, JSON_INPUT_FILE, SKIP_ALREADY_INDEXED, ONLY_CREATE_INDEX, ADD_WIKIDATA, SHOW_PROGRESS_EVERY, INDEX_TEST, INDEX_TEST_SIZE
from config import ES_HOST_PORT, ES_AUTHENTICATION, ES_INDEX, ES_INDEXING_CHUNK_SIZE
from functions.utils import get_variants_text, get_variants_list, get_variants_country, get_languages_country
from functions.wiki import get_wiki_id_by_ror_id, get_wiki_labels
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk

# Create Elasticsearch index if not exists.
def create_index(client, index):
#------------------------------
  index_settings = {
    'settings': {
      'number_of_shards': 1,
      'number_of_replicas': 0,
      'analysis': {
        'normalizer': {
          'keyword_normalizer': {
            'type': 'custom',
            'char_filter': [],
            'filter': ['lowercase', 'trim']
          }
        },
        'analyzer': {
          'shingle_analyzer': {
            'tokenizer': 'standard',
            'filter': ['lowercase', 'snowball', 'shingle_filter']
          },
          'stemming_analyzer': {
            'tokenizer': 'standard',
            'filter': ['lowercase', 'snowball']
          },
          'whitespace_analyzer': {
            'tokenizer': 'whitespace'
          }
        },
        'filter': {
          'shingle_filter': {
            'type': 'shingle',
            'min_shingle_size': 2,
            'max_shingle_size': 5,
            'output_unigrams': False
          }
        }
      }
    },
    'mappings': {
      'properties': {
        'ror_id': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'alias': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'acronym': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'label': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'wikidata_label': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'country_name': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'country_code': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'city': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'postcode': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'street_address': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'ror_status': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'wikidata_id': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'source': {'type': 'keyword', 'normalizer': 'keyword_normalizer'},
        'ror_name': {
          'type': 'text',
          'analyzer': 'stemming_analyzer',
          'fields': {
            'length': { 
              'type': 'token_count',
              'analyzer': 'whitespace_analyzer'
            },
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            },
            'shingles': {
              'type': 'text',
              'analyzer': 'shingle_analyzer'
            }
          }
        },
        'country': {
          'type': 'text',
          'fields': {
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            }
          }
        },
        'region': {
          'type': 'text',
          'fields': {
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            }
          }
        },
        'address': {
          'type': 'text',
          'fields': {
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            }
          }
        },
        'parent': {
          'type': 'text',
          'analyzer': 'stemming_analyzer',
          'fields': {
            'length': { 
              'type': 'token_count',
              'analyzer': 'whitespace_analyzer'
            },
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            },
            'shingles': {
              'type': 'text',
              'analyzer': 'shingle_analyzer'
            }
          }
        },
        'location': {
          'type': 'text',
          'fields': {
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            }
          }
        },
        'name': {
          'type': 'text',
          'analyzer': 'stemming_analyzer',
          'fields': {
            'length': { 
              'type': 'token_count',
              'analyzer': 'whitespace_analyzer'
            },
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            },
            'shingles': {
              'type': 'text',
              'analyzer': 'shingle_analyzer'
            }
          }
        },
        'all_search': {
          'type': 'text',
          'fields': {
            'keyword': {
              'type': 'keyword',
              'normalizer': 'keyword_normalizer'
            }
          }
        }
      }
    }
  }
  if not client.indices.exists(index=index):
    res = client.indices.create(index=index, **index_settings)
    print(res)

# Yield elements to be indexed.
def index_data(data, es_client):
#------------------------------
  display_first = True
  pos_first = 0
  if SKIP_ALREADY_INDEXED:
    already_indexed_ror_ids = es.get_all_indexed_ror_ids(es_client)
    print(f'Skipping {len(already_indexed_ror_ids)} already indexed records.')
    print(f'Indexing {len(data)-len(already_indexed_ror_ids)} new records.')
  for ror_org in data:
    if SKIP_ALREADY_INDEXED and ror_org['id'] in already_indexed_ror_ids:
      pos_first += 1
      continue
    if display_first:
      pos_first += 1
      print(f'Starting indexing at item {pos_first} with ROR id {ror_org["id"]}')
      display_first = False
    indexed_org = {}
    indexed_org['source'] = ['ror']
    # ROR info.
    indexed_org['ror_id'] = ror_org['id']
    indexed_org['ror_status'] = ror_org['status']
    indexed_org['ror_name'] = ror_org['name']
    # Names and aliases
    indexed_org['acronym'] = get_variants_list(ror_org['acronyms'])
    indexed_org['label'] = get_variants_list([label['label'] for label in ror_org['labels']]+[indexed_org['ror_name']])
    indexed_org['alias'] = get_variants_list(ror_org['aliases'])
    try:
      indexed_org['alias'].extend(get_variants_list(ror_org['external_ids']['CNRS']['all']))
    except KeyError:
      pass
    # Country name and code. TODO: Add country names in native languages.
    try:
      indexed_org['country_code'] = [ror_org['country']['country_code']]
    except KeyError:
      indexed_org['country_code'] = []
    country_names = get_variants_country(indexed_org['country_code'][0]) if indexed_org['country_code'] else []
    try:
      indexed_org['country_name'] = list(set(country_names + get_variants_text(ror_org['country']['country_name'])))
    except KeyError:
      indexed_org['country_name'] = country_names   
    # Currently there are no ROR entries with more than one address, so considering the first one.
    # City
    try:
      indexed_org['city'] = get_variants_text(ror_org['addresses'][0]['city'])
    except KeyError:
      indexed_org['city'] = []
    try:
      if ror_org['addresses'][0]['geonames_city']['city'] not in indexed_org['city']:
        indexed_org['city'].extend(get_variants_text(ror_org['addresses'][0]['geonames_city']['city']))
    except KeyError:
      pass
    # Region
    regions = []
    try:
      regions.append(ror_org['addresses'][0]['geonames_city']['geonames_admin1']['name'])
      regions.append(ror_org['addresses'][0]['geonames_city']['geonames_admin1']['ascii_name'])
      regions.append(ror_org['addresses'][0]['geonames_city']['geonames_admin2']['name'])
      regions.append(ror_org['addresses'][0]['geonames_city']['geonames_admin2']['ascii_name'])
    except KeyError:
      pass
    indexed_org['region'] = get_variants_list(regions)
    # Street address / postal code
    try:
      indexed_org['postcode'] = get_variants_text(ror_org['addresses'][0]['postcode'])
    except KeyError:
      indexed_org['postcode'] = []
    try:
      indexed_org['street_address'] = get_variants_text(ror_org['addresses'][0]['line'])
    except KeyError:
      indexed_org['street_address'] = [] 
    # WikiData
    try:
      indexed_org['wikidata_id'] = ror_org['external_ids']['Wikidata']['all']
    except KeyError:
      indexed_org['wikidata_id'] = []
    if ADD_WIKIDATA:
      # If the WikiData ID is not in ROR we try to get it from WikiData.
      if not indexed_org['wikidata_id']:
        wiki_id = get_wiki_id_by_ror_id(ror_org['id'])
        if wiki_id:
          indexed_org['wikidata_id'] = [wiki_id]
      # Wiki names in English and native languages.
      wiki_labels = []
      if indexed_org['wikidata_id']:
        if indexed_org['country_name']:
          country_languages = get_languages_country(indexed_org['country_name'][0]) + ['en']
        for wiki_id in indexed_org['wikidata_id']:
          for lang_code, lang_labels in get_wiki_labels(wiki_id, country_languages).items():
            wiki_labels.extend(lang_labels)
      indexed_org['wikidata_label'] = list(set(wiki_labels))
    else:
      indexed_org['wikidata_label'] = []
    # Parents 
    indexed_org['parent'] = get_variants_list([relationship['label'] \
                                              for relationship in ror_org['relationships'] \
                                              if relationship['type'] == 'Parent'])
    # Accumulated fields.
    # Country
    indexed_org['country'] = list(set(indexed_org['country_name'] + \
                                      indexed_org['country_code']))
    # Address
    indexed_org['address'] = list(set(indexed_org['postcode'] + \
                                      indexed_org['street_address']))
    # Names
    indexed_org['name'] =  list(set(indexed_org['alias'] + \
                                    indexed_org['acronym'] + \
                                    indexed_org['label'] + \
                                    indexed_org['wikidata_label']))
    # Locations
    indexed_org['location'] = list(set(indexed_org['country'] + \
                                       indexed_org['city'] + \
                                       indexed_org['region']))
    # All
    indexed_org['all_search'] = list(set(indexed_org['name'] + \
                                         indexed_org['location'] + \
                                         indexed_org['parent']))
    # Create and yield index action
    index_action = {
      "_op_type": "index",
      "_index": es.ES_INDEX,
      "_source": indexed_org
    }
    yield index_action



# ========================== MAIN =========================

if __name__ == '__main__':

  es_client = Elasticsearch(
      ES_HOST_PORT,
      http_auth=ES_AUTHENTICATION
  )

  # Create Elasticsearch index
  print(f'Creating index {ES_INDEX} if it does not exist.')
  create_index(es_client, ES_INDEX)

  if ONLY_CREATE_INDEX:
    print(f'Exiting after index creation.')
  else:
    # Read data.
    print(f'Processing data from {ROOT_PROJECT}/{JSON_INPUT_FILE} - indexing in {ES_INDEX}.')
    with open(f'{ROOT_PROJECT}/{JSON_INPUT_FILE}', encoding='utf-8') as file:
      data = json.load(file)
      # !!!! TO TEST !!!!
      if INDEX_TEST:
        print('Warning: Indexing a subset of data for testing purposes. To index the full ROR dump set INDEX_TEST to False.')
        data = data[:INDEX_TEST_SIZE]

    # Tag and index data.
    bulk_response = parallel_bulk(client=es_client, actions=index_data(data, es_client), chunk_size=ES_INDEXING_CHUNK_SIZE)

    # Consume the generator to execute the bulk indexing.
    # Omit this when indexing large volumes of data.
    # See https://elasticsearch-py.readthedocs.io/en/7.x/helpers.html#elasticsearch.helpers.bulk
    processed = 0
    for success, info in bulk_response:
      processed += 1
      if processed % SHOW_PROGRESS_EVERY == 0:
        print(f'Processed {processed} records.')
      if not success:
        print(f"A document failed when indexing: {info}")

    print(f'Finished indexing process.')


