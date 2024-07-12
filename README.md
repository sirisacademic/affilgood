# AffilGood 🕺🏿

AffilGood provides annotated datasets and tools to improve the accuracy of attributing scientific works to research organizations, especially in multilingual and complex contexts. The accurate attribution of scientific works to research organizations is hindered by the lack of openly available manually annotated data--in particular when multilingual and complex affiliation strings are considered. The AffilGood framework introduced in this paper addresses this gap. We identify three sub-tasks relevant for institution name disambiguation and make available annotated datasets and tools aimed at each of them, including i) a dataset annotated with affiliation spans in noisy automatically-extracted strings; ii) a dataset annotated with named entities for the identification of organizations and their locations; iii) seven datasets annotated with ROR identifiers for the evaluation of entity-linking systems. In addition, we describe, evaluate and make available newly developed tools that use these datasets to provide solutions for each of the identified sub-tasks. Our results confirm the value of the developed resources and methods in addressing key challenges in institution name disambiguation.

This is the official repository for the paper "AffilGood: Building reliable institution name disambiguation tools to improve scientific literature analysis", published in the Scholarly Document Processing (SDP) 2024 Workshop at ACL 2024 Conference.

![Figure 1](figure1.png)

## 📝 Results
Lorem ipsum

## 📣 Citation
```
@inproceedings{Lorem Ipsum
}
```

## Project Structure

The repository is structured as follows:

```
affilgood_pipeline/
├── AffilGoodNER/
│   ├── config.py
│   ├── functions.py
│   ├── __init__.py
│   └── ner.py
├── AffilGoodSpan/
│   ├── config.py
│   ├── span.py
│   ├── functions.py
│   └── __init__.py
├── AffilGoodEL/
│   ├── s2aff_el.py
│   ├── config.py
│   ├── functions.py
│   ├── download_s2aff.py
│   ├── __init__.py
│   └── S2AFF/
├── data/
│   ├── input_span/
│   ├── output_el/
│   ├── output_ner/
│   └── output_span/
├── utils/
│   ├── config.py
│   ├── functions.py
│   └── __init__.py
└── run_pipeline.py
```

### Modules

1. **AffilGoodSpan**: Identifies affiliation spans within input data.
2. **AffilGoodNER**: Performs named entity recognition on the identified spans.
3. **AffilGoodEL**: Links the recognized entities using the S2AFF entity linking module.

### Configuration

The main configuration file is located at `utils/config.py`. Each module also has its own configuration file (`config.py`) to specify parameters specific to that module. Generally, it is not necessary to modify these module-specific configuration files, but they can be adapted if needed.

## How It Works

### AffilGoodSpan

The `AffilGoodSpan` module reads input data, identifies affiliation spans, and writes the results to the output directory.

### AffilGoodNER

The `AffilGoodNER` module reads the output from `AffilGoodSpan`, performs named entity recognition on the identified spans, and writes the results to the output directory.

### AffilGoodEL

The `AffilGoodEL` module reads the output from `AffilGoodNER`, links the recognized entities using the S2AFF entity linking model, and writes the final results to the output directory. This project uses the code from S2AFF, which can be found [here](https://github.com/allenai/S2AFF). The necessary data for entity linking will be downloaded the first time the entity linking module is executed.

## How to Use

### Running the Full Pipeline

To run the full pipeline, simply execute the `run_pipeline.py` script:

```bash
python run_pipeline.py
```

This script will automatically run each module in sequence based on the configuration specified in `utils/config.py`.

### Running Individual Modules

Each module can also be run independently if needed.

#### Running AffilGoodSpan

```bash
python AffilGoodSpan/span.py
```

#### Running AffilGoodNER

```bash
python AffilGoodNER/ner.py
```

#### Running AffilGoodEL

```bash
python AffilGoodEL/s2aff_el.py
```

### Configuration

Modify the configuration settings in `utils/config.py` to suit your specific setup and requirements. This includes setting paths, file formats, and parameters for each module.

```python
############################################
### utils/config.py
### Pipeline parameters

# Full path to the root of the project.
ROOT_PROJECT = '/path/to/your/project'

# Subdirectories with each dataset to be processed.
DATASETS = ['Test']

# Define which modules to run in the pipeline.
RUN_MODULES = {
  'AffilGoodSpan': 'span',
  'AffilGoodNER': 'ner',
  'AffilGoodEL': 's2aff_el'
}

# AffilGoodSpan paths/formats
INPUT_FILES_EXTENSION_SPAN = 'tsv'
OUTPUT_FILES_EXTENSION_SPAN = 'tsv'
OVERWRITE_FILES_SPAN = False
INPUT_PATH_SPAN = 'data/input_span'
OUTPUT_PATH_SPAN = 'data/output_span'

# AffilGoodNER paths/formats
INPUT_FILES_EXTENSION_NER = OUTPUT_FILES_EXTENSION_SPAN
OUTPUT_FILES_EXTENSION_NER = 'tsv'
OVERWRITE_FILES_NER = False
INPUT_PATH_NER = OUTPUT_PATH_SPAN
OUTPUT_PATH_NER = 'data/output_ner'

# AffilGoodEL paths/formats
INPUT_FILES_EXTENSION_EL = OUTPUT_FILES_EXTENSION_NER
OUTPUT_FILES_EXTENSION_EL = 'tsv'
OVERWRITE_FILES_EL = False
INPUT_PATH_EL = OUTPUT_PATH_NER
OUTPUT_PATH_EL = 'data/output_el'
```

## Dependencies

Ensure you have all necessary dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## 📫 Contact

For further information, please contact <nicolau.duransilva@sirisacademic.com>.

## ⚖️ License

This work is distributed under a [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
