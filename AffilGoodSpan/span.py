#!/usr/bin/env python3

############################################
### AffilGoodSpan/span.py
### Main script for AffilGoodSpan

import os
import sys
import concurrent.futures
import glob
import pandas as pd

# Add parent to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AffilGoodSpan.config import *
from AffilGoodSpan.functions import *

from utils.functions import *
from utils.config import *

def main():
#-------------
  # Load Span pipeline
  span_pipe = span_pipeline(model_name_path=SPAN_MODEL, device=SPAN_DEVICE)

  # Read input data
  for DATASET in DATASETS:
    input_path_span = f'{ROOT_PROJECT}/{INPUT_PATH_SPAN}/{DATASET}'
    output_path_span = f'{ROOT_PROJECT}/{OUTPUT_PATH_SPAN}/{DATASET}'
    os.makedirs(output_path_span, exist_ok=True)
    # Define the pattern to search for files
    input_file_paths = glob.glob(f'{input_path_span}/*.{INPUT_FILES_EXTENSION_SPAN}')  
    for input_file_path in input_file_paths:
      # Process one input file
      print(f'Processing file {input_file_path}')
      input_file_base_name = os.path.splitext(os.path.basename(input_file_path))[0]
      output_file_path = f'{output_path_span}/{input_file_base_name}.{OUTPUT_FILES_EXTENSION_SPAN}'
      if os.path.exists(output_file_path) and not OVERWRITE_FILES_SPAN:
        print(f'Skipped file {input_file_path} as output file {output_file_path} exists and OVERWRITE_FILES_SPAN=False.')
      else:
        # Read file in chunks
        input_chunks_span = read_file_chunks(input_file_path, CHUNK_SIZE_SPAN, COL_FILTER_SPAN, FILTER_VALUE_SPAN)
        if TEST_SPAN and input_chunks_span:
          input_chunks_span = [input_chunks_span[0].head(10)]
        # Process chunks in parallel if the output file does not already exist
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(input_chunks_span), MAX_PARALLEL_SPAN)) as executor:
          futures = []
          for df_chunk in input_chunks_span:
            futures.append(executor.submit(process_chunk_span, span_pipe, df_chunk))
          results_span = pd.concat([future.result() for future in concurrent.futures.as_completed(futures)])
        # Write output
        print(f'Writing Span output to {output_file_path}')
        write_output(output_file_path, results_span) 
      
#-------------------- MAIN --------------------

if __name__ == '__main__':
  main()
      
      
      

