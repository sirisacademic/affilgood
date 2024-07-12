############################################
### AffilGoodNER/config.py
### NER parameters

# TEST NER
TEST_NER = False

# NER Model
NER_MODEL = f'nicolauduran45/affilgood-ner-test-multilingual-v5'

# Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which the NER pipeline will be allocated.
NER_DEVICE = 0

# Chunk size.
CHUNK_SIZE_NER = 10000

# Max. parallel processes.
MAX_PARALLEL_NER = 10

# batch_size parameter for the pipeline.
BATCH_SIZE_PIPELINE_NER = 64

# Now using the same threshold for all entity types.
THRESHOLD_SCORE_NER = 0.75

# Whether to change case to title-cased before processing (affilgood-ner-test-v3 has problems with all upper-cased strings).
TITLE_CASE_NER = False

# Whether to post-process "word" field to match offsets.
FIX_PREDICTED_WORDS_NER = True

# Column name with text in input data.
COL_RAW_AFFILIATION = 'raw_affiliation_string'

# Output column with entities.
COL_NER_ENTITIES = 'ner_entities'

# Output column to mark rows with potential errors in NER output. If left blank the column is not added.
COL_POTENTIAL_ERROR_NER = 'potential_error_ner'

# Columns/values below to filter some rows that we do not want to process.
COL_FILTER_NER = 'raw_affiliation_string'

# Filter values to skip.
FILTER_VALUE_NER = ''

# NER entity labels
SUBORG_NER_LABEL = 'SUB'
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


