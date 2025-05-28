# constants.py
import os

EL_PATH = os.path.dirname(os.path.abspath(__file__))
EL_DATA_PATH = os.path.join(EL_PATH, 'data')

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
SUBORG_NER_LABEL = 'SUBORG'
SUB_NER_LABEL = 'SUB'
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

# Scoring Thresholds - **** used in S2AFF***
THRESHOLD_SCORE_FILTER_FIRSTAGE_EL = 0
THRESHOLD_SCORE_RERANKED_EL = 0.20

# Number of candidates to consider for reranking
NUM_CANDIDATES_TO_RERANK = 10

# Number of candidates to return in final results
NUM_CANDIDATES_TO_RETURN = 5

# Output Column Names
COL_PREDICTIONS_EL = 'predicted_label'
COL_PREDICTIONS_SCORES_EL = 'predicted_label_score'
COL_POTENTIAL_ERROR_NER = 'potential_error_ner'

# Processing Configuration
CHUNK_SIZE_EL = 1000
MAX_PARALLEL_EL = 20
SAVE_CHUNKS_EL = True

# File with alternative names for countries and language codes.
COUNTRY_LANGS_FILE = f'{EL_DATA_PATH}/countries_languages.tsv'
COUNTRY_ENG_NAME_COL = 'country_exonym'
COUNTRY_ALT_NAMES_COL = 'country_alternative'
COUNTRY_LANG_CODES_COL = 'lang_codes'
COUNTRY_COL_SEPARATOR = '|'

# File with abbreviations.
ABBREVIATIONS_FILE = f'{EL_DATA_PATH}/abbreviations.tsv'

# WikiData URL (to use for prefix, etc)
WIKIDATA_URL = 'https://www.wikidata.org/wiki/'

# File with organizations both in ROR and WikiData ids with WikiData labels and all languages.
WIKIDATA_LABELS_FILE = f'{EL_DATA_PATH}/ror_id_wiki_id_label_lang.tsv.gz'

# WikiData dump configuration
WIKIDATA_ORG_TYPES_SHORT = f'{EL_DATA_PATH}/wikidata_org_types_short.json'
WIKIDATA_ORG_TYPES_EXTENDED = f'{EL_DATA_PATH}/wikidata_org_types_extended.json'

# Special tokens for dense entity linking
SPECIAL_TOKENS = ['[MENTION]', '[ACRONYM]', '[PARENT]', '[CITY]', '[COUNTRY]']

# File with positive/negative samples to fine-tune the model used to generate the embeddings in the FAISS index.
CONTRASTIVE_DATASET = f'{EL_DATA_PATH}/contrastive/contrastive_dataset.json'

# Base the model for computing similarity (before fine-tuning)
# TODO: Test "jinaai/jina-embeddings-v3"
ENCODER_BASE_MODEL = "intfloat/multilingual-e5-large"

# Fine-tuned model for computing similarity.
ENCODER_MODEL_DIR = f'{EL_DATA_PATH}/contrastive/model'
#ENCODER_DEFAULT_MODEL = f'{ENCODER_MODEL_DIR}/finetuned_with_iter1_thres60_5neg_more-labels_100percROR_special_tokens'
ENCODER_DEFAULT_MODEL = "SIRIS-Lab/affilgood-dense-retriever"

# Cross-encoder model.
CROSS_ENCODER_BASE_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

# Fine-tuned model for re-ranking.
CROSS_ENCODER_MODEL_DIR = f'{EL_DATA_PATH}/cross_encoder/model'
CROSS_ENCODER_DEFAULT_MODEL = f'{CROSS_ENCODER_MODEL_DIR}/cross_enc_e5_finetuned_with_iter1_thres50_6neg_more-labels_100percROR_special_tokens'

# Direct pair reranker model.
DIRECT_PAIR_RERANKER_BASE_MODEL = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

DIRECT_PAIR_RERANKER_MODEL_DIR = f'{EL_DATA_PATH}/direct_pair_reranker/model'
#DIRECT_PAIR_RERANKER_DEFAULT_MODEL = f'{DIRECT_PAIR_RERANKER_MODEL_DIR}/model'

DIRECT_PAIR_RERANKER_DEFAULT_MODEL="jinaai/jina-reranker-v2-base-multilingual"

# Annoy configuration
USE_ANNOY_BY_DEFAULT = False
ANNOY_TREES = 100
ANNOY_INDEX_PATH = f'{EL_DATA_PATH}/annoy_index'

# Whoosh indices
WHOOSH_INDICES_PATH = f'{EL_PATH}/whoosh_indices'

# HNSW configuration
USE_HNSW_BY_DEFAULT = True
HNSW_M = 16            # Number of connections per element (higher = better accuracy but more memory)
HNSW_EF_CONSTRUCTION = 200  # Construction time/accuracy trade-off
HNSW_EF_SEARCH = 50    # Search time/accuracy trade-off
HNSW_INDICES_PATH = f'{EL_PATH}/hnsw_indices'

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



