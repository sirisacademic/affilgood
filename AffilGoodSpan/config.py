############################################
### AffilGoodSpan/config.py
### Configuration for AffilGoodSpan

# TEST mode
TEST_SPAN = False

# Span model
SPAN_MODEL = 'nicolauduran45/affilgood-affiliation-span'

# Device configuration
SPAN_DEVICE = 0

# Chunk size for processing
CHUNK_SIZE_SPAN = 10000

# Maximum parallel processes
MAX_PARALLEL_SPAN = 10

# Batch size for the pipeline
BATCH_SIZE_PIPELINE_SPAN = 64

# Threshold for span detection
THRESHOLD_SCORE_SPAN = 0.75

# Whether to post-process "word" field to match offsets.
FIX_PREDICTED_WORDS_SPAN = True

# Change case to title-case before processing
TITLE_CASE_SPAN = False

# Column name with text in input data
COL_RAW_TEXT = 'raw_text'

# Output column with spans
COL_SPAN_ENTITIES = 'raw_affiliation_string'

# Output column to mark rows with potential errors
COL_POTENTIAL_ERROR_SPAN = 'potential_error_span'

# Columns/values to filter rows not to be processed
COL_FILTER_SPAN = 'raw_text'
FILTER_VALUE_SPAN = ''

# Span entity fields
SPAN_ENTITY_TYPE_FIELD = 'entity_group'
SPAN_ENTITY_TEXT_FIELD = 'word'
SPAN_ENTITY_SCORE_FIELD = 'score'
SPAN_ENTITY_START_FIELD = 'start'
SPAN_ENTITY_END_FIELD = 'end'

