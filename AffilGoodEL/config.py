############################################
### AffilGoodEL/config.py
### Entity linking parameters

# Whether to process a few for testing.
TEST_EL = False

# S2AFF root path relative to AffilGoodEL
S2AFF_PATH = 'S2AFF'

# ROR dump link.
ROR_DUMP_LINK = 'https://zenodo.org/api/records/?communities=ror-data&sort=mostrecent'

# ROR dump path relative to AffilGoodEL. Leave empty to download the latest ROR dump.
ROR_DUMP_PATH = ''

# Update OpenAlex ROR counts if current file older than this number of days.
UPDATE_OPENALEX_WORK_COUNTS_OLDER_THAN = 7

# Substring of URLs to omit when downloading S2AFF data
OMIT_S2AFF = ['ner_model', 'training', 'ror-data.json']

# Chunk size.
CHUNK_SIZE_EL = 1000

# Max. parallel processes.
MAX_PARALLEL_EL = 20

# Save output by chunks.
SAVE_CHUNKS_EL = True

# This score is used if we want to run a system keeping first-stage candidates (NOT USED now).
THRESHOLD_SCORE_FIRSTSTAGE_EL = 0.65

# This score is used to determine which candidates are passed to the re-ranking stage. Now considering all positive values.
THRESHOLD_SCORE_FILTER_FIRSTAGE_EL = 0

# This score is used to determine which candidates are kept after the re-ranking stage.
THRESHOLD_SCORE_RERANKED_EL = 0.20

#THRESHOLD_SCORE_RERANKED_EL = 0.15
NUM_CANDIDATES_TO_RERANK = 10

# Names of EL output column for labels.
COL_PREDICTIONS_EL = 'predicted_label'

# Names of EL output columns for labels with scores.
COL_PREDICTIONS_SCORES_EL = 'predicted_label_score'

## ROR dictionary fields
ROR_NAME_FIELD = 'name'
ROR_ID_FIELD = 'id'



