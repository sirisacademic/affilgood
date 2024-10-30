
# ROR Elastic Entity Linking

This folder provides a set of scripts for entity linking using Elasticsearch and a quantized Large Language Model (LLM). The process includes creating and populating an Elasticsearch index with the contents of the ROR dump, optionally enriched with WikiData labels, and then linking entities either directly through Elasticsearch's scores or through a reranking process.

## Prerequisites

- Python 3.x
- Elasticsearch
- Required Python packages (listed in `requirements.txt`)

## Project Structure

- `1_elastic_link_ner_organizations.py`: Script to link entities using Elasticsearch.
- `1.1_final_predictions_elastic_no_reranked.py`: Script to get predictions based on Elasticsearch's scores without reranking.
- `2_llm_reranking.py`: Script to rerank Elasticsearch predictions using a quantized LLM.
- `2.1_final_reranked_predictions.py`: Script to get final predictions after reranking.
- `index_ror_data.py`: Script to create and populate the Elasticsearch index with the contents of the ROR dump.
- `config.py`: Configuration file containing parameters that need to be changed before creating the index and using the scripts for entity linking.
- `functions/`: Directory containing helper functions.
- `.gitignore`: Git ignore file.
- `README`: This file.

## Setup

1. Clone the repository to your local machine:

    ```bash
    git clone <repository_url>
    cd ror_elastic_entity_linking
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Configure Elasticsearch and other parameters by editing `config.py`:

    ```python
    # config.py

    ES_HOST = 'localhost'
    ES_PORT = 9200
    INDEX_NAME = 'ror'
    ROR_DUMP_PATH = 'path/to/ror/dump.json'
    WIKIDATA_LABELS_PATH = 'path/to/wikidata/labels.json'  # Optional
    # Add other configurations as needed
    ```

## Creating and Populating Elasticsearch Index

Before using the scripts for entity linking, you need to create and populate the Elasticsearch index:

```bash
python index_ror_data.py
```

This script will create an index in Elasticsearch and populate it with the contents of the ROR dump, optionally enriched with WikiData labels.

## Entity Linking Pipelines

### Pipeline 1: Direct Predictions from Elasticsearch Scores

1. Run the entity linking script using Elasticsearch:

    ```bash
    python 1_elastic_link_ner_organizations.py
    ```

2. Get the final predictions without reranking:

    ```bash
    python 1.1_final_predictions_elastic_no_reranked.py
    ```

### Pipeline 2: Reranked Predictions Using LLM

1. Run the entity linking script using Elasticsearch:

    ```bash
    python 1_elastic_link_ner_organizations.py
    ```

2. Rerank Elasticsearch predictions using the quantized LLM:

    ```bash
    python 2_llm_reranking.py
    ```

3. Get the final predictions after reranking:

    ```bash
    python 2.1_final_reranked_predictions.py
    ```

## Notes

- Ensure that the Elasticsearch service is running before executing the scripts.
- Modify the `config.py` file as needed to match your environment and requirements.


