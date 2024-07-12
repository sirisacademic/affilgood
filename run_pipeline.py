#!/usr/bin/env python3

############################################
### run_pipeline.py
### Main script to run the entire pipeline

import sys
import os
import importlib

# Ensure the project root is in the system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import RUN_MODULES

def run_module(module_name, script_name):
#------------------
  module_path = f'{module_name}.{script_name}'
  module = importlib.import_module(module_path)
  if hasattr(module, 'main'):
    module.main()
  else:
    print(f'Module {module_path} does not have a main() function.')

def main():
#------------------
  for module_name, script_name in RUN_MODULES.items():
    print(f'Running module: {module_name}')
    run_module(module_name, script_name)

#-------------------- MAIN --------------------

if __name__ == "__main__":
  main()


