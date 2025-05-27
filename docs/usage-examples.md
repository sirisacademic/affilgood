# AffilGood Usage Examples

This document provides detailed examples of how to use the AffilGood library for different scenarios.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Working with Different Entity Linkers](#working-with-different-entity-linkers)
3. [Working with Multiple Data Sources](#working-with-multiple-data-sources)
4. [Handling Multilingual Affiliations](#handling-multilingual-affiliations)
5. [Processing Large Datasets](#processing-large-datasets)
6. [Advanced Configurations](#advanced-configurations)

## Basic Usage

### Complete Pipeline

The simplest way to use AffilGood is to run the complete pipeline with default settings:

```python
from affilgood import AffilGood

# Initialize with default settings
affil_good = AffilGood()

# Example affiliation strings
affiliations = [
    "Department of Computer Science, University of California, Berkeley, CA 94720, USA",
    "Max Planck Institute for Intelligent Systems, 72076 Tübingen, Germany"
]

# Process affiliations through the complete pipeline
results = affil_good.process(affiliations)

# Print identified organizations and their ROR IDs
for item in results:
    print(f"Original: {item['raw_text']}")
    print(f"Spans: {item['span_entities']}")
    print(f"Organizations: {[ner.get('ORG', []) for ner in item['ner']]}")
    print(f"ROR: {item['ror']}")
    print("---")
```

### Step-by-Step Processing

You can also run each step of the pipeline separately:

```python
from affilgood import AffilGood

affil_good = AffilGood()

# Example affiliation strings
affiliations = [
    "Department of Computer Science, University of California, Berkeley, CA 94720, USA",
    "Max Planck Institute for Intelligent Systems, 72076 Tübingen, Germany"
]

# Step 1: Identify spans
spans = affil_good.get_span(affiliations)
print("After span identification:", spans[0]['span_entities'])

# Step 2: Perform named entity recognition
entities = affil_good.get_ner(spans)
print("Entities recognized:", entities[0]['ner'][0])

# Step 3: Normalize location metadata
normalized = affil_good.get_normalization(entities)
print("Normalized locations:", normalized[0]['osm'][0])

# Step 4: Link to ROR identifiers
linked = affil_good.get_entity_linking(normalized)
print("ROR IDs:", linked[0]['ror'][0])
```

### Custom Configuration

Customize AffilGood components for your specific needs:

```python
from affilgood import AffilGood

# Configure with specific models and settings
affil_good = AffilGood(
    span_separator=';',  # Split spans by semicolons instead of using model
    ner_model_path='SIRIS-Lab/affilgood-NER-multilingual',
    entity_linkers=['Whoosh', 'Dense'],  # Use multiple linkers
    return_scores=True,  # Include confidence scores in results
    metadata_normalization=True,
    verbose=True,  # Enable detailed logging
    device='cuda:0'  # Use specific GPU
)

# Process affiliations with custom configuration
affiliations = [
    "Department of Physics; Faculty of Engineering, University of XYZ, London, UK",
    "Institute of Quantum Computing, ETH Zürich, Switzerland"
]

results = affil_good.process(affiliations)
```

## Working with Different Entity Linkers

### Using Whoosh Linker

```python
from affilgood import AffilGood
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Initialize AffilGood with Whoosh linker
affil_good = AffilGood(entity_linkers='Whoosh')

# Or with custom Whoosh linker configuration
whoosh_linker = WhooshLinker(
    threshold_score=0.3,
    rerank=True,
    number_candidates_rerank=7,
    debug=False
)

affil_good = AffilGood(entity_linkers=whoosh_linker)

# Process affiliations
affiliations = ["Stanford University, Department of Computer Science, Stanford, CA, USA"]
results = affil_good.process(affiliations)
```

### Using Dense Linker

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker

# Initialize AffilGood with Dense linker
affil_good = AffilGood(entity_linkers='Dense')

# Or with custom Dense linker configuration
dense_linker = DenseLinker(
    threshold_score=0.35,
    use_hnsw=True,
    data_source="ror"
)

affil_good = AffilGood(entity_linkers=dense_linker)

# Process affiliations
affiliations = ["Stanford University, Department of Computer Science, Stanford, CA, USA"]
results = affil_good.process(affiliations)
```

### Using S2AFF Linker

```python
from affilgood import AffilGood
from affilgood.entity_linking.s2aff_linker import S2AFFLinker

# Initialize AffilGood with S2AFF linker
affil_good = AffilGood(entity_linkers='S2AFF')

# Or with custom S2AFF linker configuration
s2aff_linker = S2AFFLinker(
    debug=False
)

affil_good = AffilGood(entity_linkers=s2aff_linker)

# Process affiliations
affiliations = ["Stanford University, Department of Computer Science, Stanford, CA, USA"]
results = affil_good.process(affiliations)
```

### Using Multiple Linkers

```python
from affilgood import AffilGood
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.dense_linker import DenseLinker

# Initialize AffilGood with multiple linkers
affil_good = AffilGood(entity_linkers=['Whoosh', 'Dense'])

# Or with custom linker configurations
whoosh_linker = WhooshLinker(threshold_score=0.3)
dense_linker = DenseLinker(threshold_score=0.35)

affil_good = AffilGood(entity_linkers=[whoosh_linker, dense_linker])

# Process affiliations
affiliations = ["Stanford University, Department of Computer Science, Stanford, CA, USA"]
results = affil_good.process(affiliations)
```

### Using Rerankers

```python
from affilgood import AffilGood
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker

# Create a reranker
reranker = DirectPairReranker(model_name="jinaai/jina-reranker-v2-base-multilingual")

# Initialize AffilGood with the custom reranker
affil_good = AffilGood(reranker=reranker)

# Process affiliations
affiliations = ["Stanford University, Department of Computer Science, Stanford, CA, USA"]
results = affil_good.process(affiliations)
```

## Working with Multiple Data Sources

### Using ROR Data Source

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker

# Configure Dense linker with ROR data source
dense_linker = DenseLinker(data_source="ror")

# Initialize AffilGood with the custom linker
affil_good = AffilGood(entity_linkers=dense_linker)

# Process affiliations
affiliations = ["Stanford University, Department of Computer Science, Stanford, CA, USA"]
results = affil_good.process(affiliations)
```

### Using WikiData Data Source

```python
from affilgood import AffilGood

# Initialize AffilGood with the custom WikiData organization types/countries
affil_good = AffilGood(
                wikidata_countries=["Spain", "Germany"],
                wikidata_org_types=["foundation", "university"]
            )

# Process affiliations
affiliations = [
    "Universidad de Barcelona, Barcelona, Spain",
    "Charité - Universitätsmedizin Berlin, Berlin, Germany"
]
results = affil_good.process(affiliations)
```

### Using Spanish Hospitals Data Source

```python
from affilgood import AffilGood
# Import the plugin to register the data source
import affilgood.entity_linking.plugins.spanish_hospitals

# Use Spanish hospitals data source directly
affil_good = AffilGood(data_sources=["spanish_hospitals"])

# Process affiliations
affiliations = ["Hospital Universitario La Paz, Madrid, Spain"]
results = affil_good.process(affiliations)
```

### Using Multiple Data Sources

```python
from affilgood import AffilGood
# Import plugins to register additional data sources
import affilgood.entity_linking.plugins.spanish_hospitals
import affilgood.entity_linking.plugins.sicris_organizations

# Use multiple data sources
affil_good = AffilGood(data_sources=["ror", "spanish_hospitals", "sicris"])

# Process affiliations from different sources
affiliations = [
    "Stanford University, Department of Computer Science, Stanford, CA, USA",
    "Hospital Universitario La Paz, Madrid, Spain",
    "University of Ljubljana, Ljubljana, Slovenia"
]
results = affil_good.process(affiliations)
```

## Handling Multilingual Affiliations

### Direct Multilingual Processing (Recommended)

AffilGood's default multilingual models handle different languages automatically:

```python
from affilgood import AffilGood

# Default configuration uses multilingual models
affil_good = AffilGood()

# Process multilingual affiliations directly - no translation needed
affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Università degli Studi di Milano, Milano, Italia",
    "Universität Heidelberg, Heidelberg, Deutschland",
    "東京大学, 東京, 日本"
]

# Multilingual models handle different languages automatically
results = affil_good.process(affiliations)

for result in results:
    print(f"Original: {result['raw_text']}")
    print(f"ROR: {result['ror']}")
    print("---")
```

### Optional Translation (Advanced)

Translation is available but not recommended for most use cases:

```python
from affilgood import AffilGood
from affilgood.preprocessing.llm_translator import LLMTranslator

# Only use translation if specifically needed
translator = LLMTranslator(
    skip_english=True,
    use_external_api=False  # or True for API mode
)

# Translate before processing (optional)
affiliations = ["Universidad de Barcelona, Barcelona, España"]
translated = translator.translate_batch(affiliations)

# Process translated affiliations
affil_good = AffilGood()
results = affil_good.process(translated)
```

## Processing Large Datasets

AffilGood has built-in batch processing capabilities that handle large datasets efficiently.
The recommended approach is to use AffilGood's native batch processing.

### Using Built-in Batch Processing

```python
from affilgood import AffilGood
import pandas as pd

# Initialize AffilGood
affil_good = AffilGood()

# Load large dataset
df = pd.read_csv('large_affiliations.csv')
affiliations = df['affiliation'].tolist()

# AffilGood automatically handles batching internally
print(f"Processing {len(affiliations)} affiliations...")
results = affil_good.process(affiliations, batch_size=64)

# Extract ROR IDs
ror_ids = []
for result in results:
    ror_id = None
    if result['ror'] and result['ror'][0]:
        ror_parts = result['ror'][0].split('{https://ror.org/')
        if len(ror_parts) > 1:
            ror_id = ror_parts[1].split('}')[0]
    ror_ids.append(ror_id)

# Create results dataframe
results_df = pd.DataFrame({
    'affiliation': [r['raw_text'] for r in results],
    'ror_id': ror_ids
})

results_df.to_csv('processed_affiliations.csv', index=False)
print(f"Processed {len(results)} affiliations")
```

## Advanced Configurations

### Custom Pipeline with Reranking

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker
from affilgood.entity_linking.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Create custom reranker
reranker = DirectPairReranker(
    model_name="cross-encoder/stsb-roberta-base",
    reranking_strategy="max_score",
    use_cache=True,
    debug=True
)

# Create dense linker with reranker
dense_linker = DenseLinker(
    data_manager=data_manager,
    threshold_score=0.30,
    use_hnsw=True,
    data_source="ror"
)

# Attach reranker to linker
dense_linker.reranker = reranker

# Initialize AffilGood with custom configuration
affil_good = AffilGood(
    entity_linkers=dense_linker,
    return_scores=True,
    metadata_normalization=True,
    verbose=True
)

# Process affiliations
affiliations = [
    "Computer Science Department, Stanford University, Stanford, CA",
    "Department of Physics, Harvard University, Cambridge, MA"
]

results = affil_good.process(affiliations)
```

### Performance-Optimized Configuration

```python
from affilgood import AffilGood

affil_good = AffilGood(
    entity_linkers='Whoosh',
    rerank=False,  # Disable reranking for speed
    metadata_normalization=False,  # Disable metadata normalization
    verbose=False,
    device='cpu'  # Use CPU to save GPU memory
)

# Process large dataset quickly
import time

start_time = time.time()
results = affil_good.process(large_affiliation_list)
end_time = time.time()

print(f"Processed {len(large_affiliation_list)} affiliations in {end_time - start_time:.2f} seconds")
```

### Multi-Source Comprehensive Pipeline

AffilGood can use multiple data sources simultaneously:

```python
from affilgood import AffilGood
# Import plugins for additional data sources
import affilgood.entity_linking.plugins.spanish_hospitals
import affilgood.entity_linking.plugins.sicris_organizations

# Initialize with multiple data sources
affil_good = AffilGood(
    data_sources=["ror", "spanish_hospitals", "sicris"],
    return_scores=True,
    metadata_normalization=True,
    verbose=True
)

# Process multilingual affiliations directly
multilingual_affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Università degli Studi di Milano, Milano, Italia",
    "Max-Planck-Institut für Intelligente Systeme, Tübingen, Deutschland",
    "Stanford University, Department of Computer Science, Stanford, CA, USA"
]

# Process directly with multilingual models
results = affil_good.process(multilingual_affiliations)

# Display results
for result in results:
    print(f"Affiliation: {result['raw_text']}")
    print(f"ROR Matches: {result['ror']}")
    print("---")
```
