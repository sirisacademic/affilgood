############################################
### utils/config.py
### Pipeline parameters

# Full path to the root of the project.
ROOT_PROJECT = '/home/pablo/affilgood_pipeline'

# Subdirectories with each dataset to be processed.
# Output directories with the same names will be created under OUTPUT_PATH_[TASK] if they don't exist.
DATASETS = ['Test']

# Define which modules to run in the pipeline.
# It is a dictionary with the name of the module (directory) and the specific process to run.
# Now we only have two possibilities in the case of entity linking. For span detection and NER there is only one.
RUN_MODULES = {
  'AffilGoodSpan': 'span',
  'AffilGoodNER': 'ner',
  'AffilGoodEL': 's2aff_el'
}

#=========== AffilGoodSpan paths/formats ===========

# Input extension/format ('csv', 'tsv', 'pkl', 'parquet')
INPUT_FILES_EXTENSION_SPAN = 'tsv'

# Output extension/format ('csv', 'tsv', 'pkl', 'parquet')
# This is the input format for NER.
OUTPUT_FILES_EXTENSION_SPAN = 'tsv'

# Overwrite existing files.
OVERWRITE_FILES_SPAN = False

# Path to input directory for AffilGoodSpan relative to project's root directory (e.g. affilgood_pipeline).
INPUT_PATH_SPAN = 'data/input_span'

# Output path relative to project's root directory (e.g. affilgood_pipeline)
# This is the input path for NER.
OUTPUT_PATH_SPAN = 'data/output_span'

#=========== AffilGoodNER paths/formats ===========

# By default the input format of AffilGoodNER is the output format of AffilGoodSpan.
INPUT_FILES_EXTENSION_NER = OUTPUT_FILES_EXTENSION_SPAN

# Output extension/format ('csv', 'tsv', 'pkl', 'parquet')
# This is the input format for EL.
OUTPUT_FILES_EXTENSION_NER = 'tsv'

# Overwrite existing files.
OVERWRITE_FILES_NER = False

# By default the input of AffilGoodNER is the output of AffilGoodSpan. Change if necessary.
INPUT_PATH_NER = OUTPUT_PATH_SPAN

# Output path relative to project's root directory (e.g. affilgood_pipeline)
# This is the input path for EL.
OUTPUT_PATH_NER = 'data/output_ner'

#=========== AffilGoodEL paths/formats ===========

# By default the input format of AffilGoodEL is the output format of AffilGoodNER.
INPUT_FILES_EXTENSION_EL = OUTPUT_FILES_EXTENSION_NER

# Output extension/format ('csv', 'tsv', 'pkl', 'parquet')
OUTPUT_FILES_EXTENSION_EL = 'tsv'

# Overwrite existing files.
OVERWRITE_FILES_EL = False

# By default the input of AffilGoodEL is the output of AffilGoodNER. Change if necessary.
INPUT_PATH_EL = OUTPUT_PATH_NER

# Output path relative to project's root directory (e.g. affilgood_pipeline)
OUTPUT_PATH_EL = 'data/output_el'



