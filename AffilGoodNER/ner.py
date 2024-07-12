#!/usr/bin/env python3

############################################
### AffilGoodNER/ner.py
### Main script for AffilGoodNER

import os
import sys
import concurrent.futures
import glob
import pandas as pd

# Add parent to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AffilGoodNER.config import *
from AffilGoodNER.functions import *

from utils.functions import *
from utils.config import *

def main():
#-------------
  # Load NER pipeline.   
  ner_pipe = ner_pipeline(model_name_path=NER_MODEL, device=NER_DEVICE)

  # Read input data.
  for DATASET in DATASETS:
    input_path_ner = f'{ROOT_PROJECT}/{INPUT_PATH_NER}/{DATASET}'
    output_path_ner = f'{ROOT_PROJECT}/{OUTPUT_PATH_NER}/{DATASET}'
    os.makedirs(output_path_ner, exist_ok=True)
    # Define the pattern to search for files
    input_file_paths = glob.glob(f'{input_path_ner}/*.{INPUT_FILES_EXTENSION_NER}')  
    for input_file_path in input_file_paths:
      # Process one input file.
      print(f'Processing file {input_file_path}')
      input_file_base_name = os.path.splitext(os.path.basename(input_file_path))[0]
      output_file_path = f'{output_path_ner}/{input_file_base_name}.{OUTPUT_FILES_EXTENSION_NER}'
      if os.path.exists(output_file_path) and not OVERWRITE_FILES_NER:
        print(f'Skipped file {input_file_path} as output file {output_file_path} exists and OVERWRITE_FILES_NER=False.')
      else:
        # Read file in chunks.
        input_chunks_ner = read_file_chunks(input_file_path, CHUNK_SIZE_NER, COL_FILTER, FILTER_VALUE)
        if TEST_NER and input_chunks_ner:
          input_chunks_ner = [input_chunks_ner[0].head(10)]
        # Process chunks in parallel if the output file does not already exist
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(input_chunks_ner), MAX_PARALLEL_NER)) as executor:
          futures = []
          for df_chunk in input_chunks_ner:
            futures.append(executor.submit(process_chunk_ner, ner_pipe, df_chunk))
          results_ner = pd.concat([future.result() for future in concurrent.futures.as_completed(futures)])
        # Write output. 
        print(f'Writing NER output to {output_file_path}')
        write_output(output_file_path, results_ner)       
      
      
#-------------------- MAIN --------------------

if __name__ == '__main__':
  main()
      
      

