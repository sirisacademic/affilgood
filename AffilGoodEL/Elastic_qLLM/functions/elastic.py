import sys
sys.path.insert(0, '..')

from config import ES_DISTANCE_SCORE, ES_INDEX, ES_MAX_HITS
from elasticsearch.helpers import scan

# Query the Elasticsearch server.
def query_elastic(es_client, query, index, num_results=ES_MAX_HITS, fields=['ror_id', 'ror_name'], dist_score=ES_DISTANCE_SCORE):
#-------------------------------------------------------------------------------------------------------------
  results = []
  response = []
  matched_queries = []
  try:
    response = es_client.search(index=index, query=query, _source=fields, size=num_results)
  except Exception as e:
    print(e)
    print(query)
  if 'hits' in response:
    hits = response['hits']
    max_score = hits['max_score']
    for hit in hits['hits']:
      if 'matched_queries' in hit:
        matched_queries.append(hit['matched_queries'])
      score = hit['_score']
      source = hit['_source']
      if dist_score==0 or score >= max_score-dist_score:
        source['score'] = score
        results.append(source)
      else:
        break
  return (results, matched_queries)

# Generate Elasticsearch 'match' query.
def fuzzy_match_query(field, text, fuzziness=0, name='', boost=0):
#------------------------------------------------
  query = {'match': {field: {'query': text, 'fuzziness': fuzziness}}}
  if boost > 0:
    query['match'][field]['boost'] = boost
  if name:
    query['match'][field]['_name'] = name
  return query

def match_query(type, field, text, name='', boost=0):
#------------------------------------------------
  query = {type: {field: {'query': text}}}
  if boost > 0:
    query[type][field]['boost'] = boost
  if name:
    query[type][field]['_name'] = name
  return query

# Generate Elasticsearch 'term' query.
def term_query(field, value, name='', boost=0):
#---------------------------------------------
  query = {'term': {field: {'value': value}}}
  if boost > 0:
    query['term'][field]['boost'] = boost
  if name:
    query['term'][field]['_name'] = name
  return query

# Generate Elasticsearch 'bool' query.
def bool_query(type, subqueries, name='', boost=0):
#----------------------------------------
  query = {'bool': {type: subqueries}}
  if boost > 0:
    query['bool']['boost'] = boost
  if name:
    query['bool']['_name'] = name
  return query

# Get ROR ids for already indexed elements.
def get_all_indexed_ror_ids(es_client):
#----------------------------
  try:
    query = {'query': {'match_all': {}}}
    ror_ids = [doc['_source']['ror_id'] for doc in scan(es_client, index=ES_INDEX, query=query, _source='ror_id')]
  except:
    print('An error occurred when retrieving the indexed ROR ids.')
    ror_ids = []
  return ror_ids
      


  


