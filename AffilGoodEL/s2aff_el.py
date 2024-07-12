#!/usr/bin/env python3

############################################
### AffilGoodEL/s2aff_el.py
### Main script for AffilGoodEL

import os
import sys
import concurrent.futures
import glob
import pandas as pd
import ast

# Add parent to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AffilGoodEL.config import *
from AffilGoodEL.functions import *
from AffilGoodEL.download_s2aff import download_s2aff_data

from utils.functions import *
from utils.config import *

S2AFF_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH))
sys.path.insert(1, S2AFF_ABS_PATH)
from s2aff.ror import RORIndex
from s2aff.model import PairwiseRORLightGBMReranker
from s2aff.consts import PATHS

def main():
#------------------
  if not os.path.exists(PATHS['lightgbm_model']):
    print(f'Downloading S2AFF data...')
    download_s2aff_data()

  ror_dump_path = ROR_DUMP_PATH
  if not ror_dump_path:
    ror_dump_path = get_latest_ror()
  if ror_dump_path:
    print(f'Using ROR dump: {ror_dump_path}')
  else:
    sys.exit(f'ROR dump is not available. Exiting.')

  if days_since_last_update(PATHS['openalex_works_counts']) > UPDATE_OPENALEX_WORK_COUNTS_OLDER_THAN:
    print(f'Updating OpenAlex work counts: {PATHS["openalex_works_counts"]}')
    update_openalex_works_counts()

  # Load ROR dump and pairwise model.
  print(f'Loading ROR index and pairwise model')
  ror_index = RORIndex(ror_data_path=ror_dump_path)
  pairwise_model = PairwiseRORLightGBMReranker(ror_index)

  # Read input data.
  for DATASET in DATASETS:
    output_path_ner = f'{ROOT_PROJECT}/{OUTPUT_PATH_NER}/{DATASET}'
    output_path_el = f'{ROOT_PROJECT}/{OUTPUT_PATH_EL}/{DATASET}'
    os.makedirs(output_path_el, exist_ok=True)
    if SAVE_CHUNKS_EL:
      output_path_chunks_el = f'{output_path_el}/chunks'
      os.makedirs(output_path_chunks_el, exist_ok=True)
    # Define the pattern to search for files
    input_file_paths = glob.glob(f'{output_path_ner}/*.{OUTPUT_FILES_EXTENSION_NER}')  
    for input_file_path in input_file_paths:
      # Process one input file.
      print(f'Processing file {input_file_path}')
      input_file_base_name = os.path.splitext(os.path.basename(input_file_path))[0]
      output_file_path = f'{output_path_el}/{input_file_base_name}.{OUTPUT_FILES_EXTENSION_EL}'
      output_file_path_chunks = f'{output_path_chunks_el}/{input_file_base_name}_{{}}.{OUTPUT_FILES_EXTENSION_EL}' if SAVE_CHUNKS_EL else ''
      if os.path.exists(output_file_path) and not OVERWRITE_FILES_EL:
        print(f'Skipped file {input_file_path} as output file {output_file_path} exists and OVERWRITE_FILES_EL=False.')
      else:
        # Read file in chunks.
        input_chunks_el = read_file_chunks(input_file_path, CHUNK_SIZE_EL, converters={COL_NER_ENTITIES: ast.literal_eval})
        if TEST_EL and input_chunks_el:
          input_chunks_el = [input_chunks_el[0].head(10)]
        # Process chunks in parallel if the output file does not already exist
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(input_chunks_el), MAX_PARALLEL_EL)) as executor:
          futures = []
          for df_chunk in input_chunks_el:
            print(f'Submitting chunk with rows {df_chunk.index[0]} - {df_chunk.index[-1]} to executor')
            futures.append(executor.submit(process_chunk_el, ror_index, pairwise_model, df_chunk, output_file_path_chunks, OVERWRITE_FILES_EL))
          results_el = pd.concat([future.result() for future in concurrent.futures.as_completed(futures)])
        # Write output. 
        print(f'Writing entity linking output to {output_file_path}')
        write_output(output_file_path, results_el) 
        

#-------------------- MAIN --------------------

if __name__ == '__main__':
  main()


      
