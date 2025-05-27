# constants.py

# Whether to process a few for testing
TEST_EL = False

# Substring of URLs to omit when downloading S2AFF data.
OMIT_S2AFF = [
    'ner',
    'ror-data',
    'annotations',
    'features'
]

# This score is used if we want to run a system keeping first-stage candidates (NOT USED now)
THRESHOLD_SCORE_FIRSTSTAGE_EL = 0.65

# NER entity fields
NER_ENTITY_TYPE_FIELD = 'entity_group'
NER_ENTITY_TEXT_FIELD = 'word'
NER_ENTITY_SCORE_FIELD = 'score'
NER_ENTITY_START_FIELD = 'start'
NER_ENTITY_END_FIELD = 'end'

# Names of NER output columns
COL_NER_ENTITIES = 'ner_raw'

# NER entity Labels
MAINORG_NER_LABEL = 'ORG'
SUBORG_NER_LABEL = 'SUB'
CITY_NER_LABEL = 'CITY'
REGION_NER_LABEL = 'REGION'
COUNTRY_NER_LABEL = 'COUNTRY'

# OpenStreeMap Labels
CITY_OSM_LABEL = 'CITY'
PROVINCE_OSM_LABEL = 'PROVINCE'
STATE_OSM_LABEL = 'STATE'
COUNTRY_OSM_LABEL = 'COUNTRY'

# ROR Fields
ROR_ID_FIELD = 'id'
ROR_NAME_FIELD = 'name'

# NER Configuration
MIN_LENGTH_NER_ENTITY = 2
IGNORE_NER_ENTITY_PREFIX = '##'

# Scoring Thresholds
THRESHOLD_SCORE_FILTER_FIRSTAGE_EL = 0
THRESHOLD_SCORE_RERANKED_EL = 0.20
NUM_CANDIDATES_TO_RERANK = 10

# Output Column Names
COL_PREDICTIONS_EL = 'predicted_label'
COL_PREDICTIONS_SCORES_EL = 'predicted_label_score'
COL_POTENTIAL_ERROR_NER = 'potential_error_ner'

# Processing Configuration
CHUNK_SIZE_EL = 1000
MAX_PARALLEL_EL = 20
SAVE_CHUNKS_EL = True

# S2AFF.
S2AFF_PATH = 'S2AFF'
UPDATE_OPENALEX_WORK_COUNTS_OLDER_THAN = 30

# ROR
ROR_URL = 'https://ror.org/'

# If path left empty download always the latest one.
# Name file in S2AFF/data or full path.
ROR_DUMP_PATH = ''
ROR_DUMP_LINK = 'https://zenodo.org/api/records/?communities=ror-data&sort=mostrecent'

# File chunking.
CHUNK_SIZE_EL = 1000
MAX_PARALLEL_EL = 20
SAVE_CHUNKS_EL = True
OUTPUT_PARTIAL_CHUNKS = 'output'

