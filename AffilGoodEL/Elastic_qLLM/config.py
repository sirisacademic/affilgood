############################################
### Base paths

# Full path to the root of the project.
ROOT_PROJECT = '/home/pablo/ror_entity_linking'

############################################
### Elasticsearch parameters

# Options: "local", "siris", "cloud"
ES_SERVER = "siris"

# Index name.
ES_INDEX = 'orgs_ror_v1.47-2024-05-30'

# Host/port.
ES_HOST_PORT = "http://192.168.0.180:9200"
  
# Authentication
ES_AUTHENTICATION = None

# Default size of chunks to index.
ES_INDEXING_CHUNK_SIZE = 1000

# Default number of maximum results to retrieve.
ES_MAX_HITS = 5

# We retrieve all results with a score within this distance from the maximum score. Set as 0 to retrieve all.
ES_DISTANCE_SCORE = 0

############################################
### Parameters for index_ror_data.py

# Input file. Relative to ROOT_PROJECT.
JSON_INPUT_FILE = 'ror_dump/ror-data.json'

# Whether to query WikiData to enrich labels with WikiData's.
ADD_WIKIDATA = True

# User-Agent header used in WikiData requests.
# Only considered if ADD_WIKIDATA=True
WIKIDATA_USER_AGENT = 'RorIndexing/1.0 (pablo.accuosto@sirisacademic.com)'

# This file is used to populate a cache with mappings of ROR ids, WikiData ids and labels for multiple languages. Relative to ROOT_PROJECT.
# If a mapping is not found it is retrieved using WikiData's SPARQL endpoint.
# If the path is set to None the file is not loaded and all the mappings are retrieved from WikiData.
# Only considered if ADD_WIKIDATA=True
FILE_ROR_ID_WIKI_ID_LABEL_LANG = 'ror_wiki/ror_id_wiki_id_label_lang.tsv'

# Skipped indexed ROR ids?
SKIP_ALREADY_INDEXED = False

# Only create index and quit?
ONLY_CREATE_INDEX = False

# Show progress every.
SHOW_PROGRESS_EVERY = 100

# Index only a subset for testing purposes.
INDEX_TEST = False

# INDEX_TEST_SIZE is used only if INDEX_TEST = True.
INDEX_TEST_SIZE = 50

############################################
### Parameters for 1_elastic_link_ner_organizations.py

# Subdirectory with data.
DATA_DIR = 'affilgood-ner-test-multilingual-v5'

# Subdirectories where data for specific datasets is read from / written to. Relative to DATA_DIR.
# Input directory is assumed to be ROOT_PROJECT/input/DATA_DIR/DATASET[0], ...
# Output directory is assumed to be ROOT_PROJECT/output/DATA_DIR/DATASET[0], ...
DATASETS = ['NL_AFFILS']

# Input format. Can be 'json' or 'parquet'.
INPUT_FORMAT = 'parquet'

# Whether to include gold annotations to evaluate (if present in input).
INCLUDE_GOLD_ANNOTATIONS_IN_OUTPUT = True

# File names to ignore in input directory.
SKIP_FILE_NAMES = ['potential_errors.json']

# Whether to output detailed data for debugging purposes.
VERBOSE_EL = False

# Include a query with the full string against all fields (can match organizations in other locations).
# Note: This is overriden as False when matching suborganizations.
INCLUDE_GENERAL_QUERY = False

# Sort results by number of matching subqueries (instead of score).
SORT_BY_MATCHING_SUBQUERIES = False

# Process only a subset for testing purposes.
ES_LINK_TEST = True

# ES_LINK_TEST_SIZE is used only if INDEX_TEST = True.
ES_LINK_TEST_SIZE = 50

# Names of NER output fields / columns.
# JSON format.
#FIELD_RAW_AFFILIATION_NER = 'raw_affil_string'
#FIELD_GOLD_ANNOTATIONS_NER = 'label'
#FIELD_PREDICTED_ENTITIES_NER = 'entities'
# Parquet format
FIELD_RAW_AFFILIATION_NER = 'raw_affiliation_string'
FIELD_GOLD_ANNOTATIONS_NER = ''
FIELD_PREDICTED_ENTITIES_NER = 'affilGood-NER-en'

############################################
### Parameters for 1.1_final_predictions_elastic_no_reranked.py

# Optional.
# These settings are only used when re-ranking is not performed, but predictions are
# obtained directly based on the Elasticsearch scores.

THRESHOLD_MAIN_ORG = 70
THRESHOLD_SUB_ORG = 200

############################################
### Parameters for 2_llm_reranking.py

MODEL = 'TheBloke/neural-chat-7B-v3-2-GPTQ'
QUANTISED_MODEL = True
MODEL_REVISION = 'main'

MAX_NEW_TOKENS = 500
TEMPERATURE = 0.1

# Format of interaction. Can be 'text' or 'json'. JSON works for larger models only.
FORMAT_INPUT_OUTPUT = 'text'

VERBOSE_RERANKING = False

# Process only a subset for testing purposes.
RERANK_TEST = True

# ES_LINK_TEST_SIZE is used only if INDEX_TEST = True.
RERANK_TEST_SIZE = 50

# Batch size.
RERANKING_BATCH_SIZE = 4


