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

# Print identified organizations and their linked identifiers
for item in results:
    print(f"Original: {item['raw_text']}")
    print(f"Spans: {item['span_entities']}")
    print(f"Organizations: {[ner.get('ORG', []) for ner in item['ner']]}")
    print(f"Entity Linking: {item['entity_linking']}")
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

# Step 4: Link to organization identifiers
linked = affil_good.get_entity_linking(normalized)
print("Entity Linking Results:", linked[0]['entity_linking'])
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

### Extracting Results from New Format

With the new nested entity linking structure, here's how to extract organization identifiers:

```python
def extract_organization_results(result, data_source="ror"):
    """Extract organization results from the new entity linking format."""
    entity_linking = result.get('entity_linking', {})
    source_results = entity_linking.get(data_source, {})
    
    # Get results per span and combined results
    per_span = source_results.get('linked_orgs_spans', [])
    combined = source_results.get('linked_orgs', '')
    
    # Parse combined results
    organizations = []
    if combined:
        for org_entry in combined.split('|'):
            if '{' in org_entry and '}' in org_entry:
                # Extract name, URL, and score
                name = org_entry[:org_entry.find('{')].strip()
                url_start = org_entry.find('{') + 1
                url_end = org_entry.find('}')
                url = org_entry[url_start:url_end]
                
                # Extract score if present
                score = None
                if ':' in org_entry[url_end:]:
                    try:
                        score = float(org_entry.split(':')[-1])
                    except ValueError:
                        pass
                
                organizations.append({
                    'name': name,
                    'url': url,
                    'score': score
                })
    
    return {
        'per_span': per_span,
        'organizations': organizations
    }

# Example usage
results = affil_good.process(affiliations)
for result in results:
    ror_results = extract_organization_results(result, "ror")
    print(f"ROR organizations: {ror_results['organizations']}")
    print(f"Per span: {ror_results['per_span']}")
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

# Extract results
for result in results:
    entity_linking = result['entity_linking']
    if 'ror' in entity_linking:
        print(f"ROR matches: {entity_linking['ror']['linked_orgs']}")
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

# Extract results
for result in results:
    entity_linking = result['entity_linking']
    print(f"Entity linking results: {entity_linking}")
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

# Extract results
for result in results:
    entity_linking = result['entity_linking']
    print(f"Entity linking results: {entity_linking}")
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

# With multiple linkers, you may get results from different systems
for result in results:
    entity_linking = result['entity_linking']
    for data_source, source_results in entity_linking.items():
        print(f"{data_source} results: {source_results['linked_orgs']}")
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

# Extract ROR results
for result in results:
    ror_results = result['entity_linking'].get('ror', {})
    if ror_results:
        print(f"ROR matches: {ror_results['linked_orgs']}")
```

### Using Multiple Data Sources

```python
from affilgood import AffilGood

# Configure AffilGood to use multiple data sources
# Note: This example assumes you have configured the Spanish hospitals data source
affil_good = AffilGood(
    entity_linkers=['Dense'],  # Can use any linker
    data_sources=['ror', 'spanish_hospitals']  # Multiple data sources
)

# Process affiliations from different domains
affiliations = [
    "Stanford University, Department of Computer Science, Stanford, CA, USA",
    "Hospital del Mar, Barcelona, Spain"
]

results = affil_good.process(affiliations)

# Extract results from different data sources
for i, result in enumerate(results):
    print(f"\nAffiliation {i+1}: {result['raw_text']}")
    entity_linking = result['entity_linking']
    
    # Check ROR results
    if 'ror' in entity_linking and entity_linking['ror']['linked_orgs']:
        print(f"  ROR: {entity_linking['ror']['linked_orgs']}")
    
    # Check Spanish hospitals results
    if 'spanish_hospitals' in entity_linking and entity_linking['spanish_hospitals']['linked_orgs']:
        print(f"  Spanish Hospitals: {entity_linking['spanish_hospitals']['linked_orgs']}")
```

### Extracting IDs from Different Data Sources

```python
def extract_ids_by_source(result):
    """Extract organization IDs from all data sources."""
    entity_linking = result.get('entity_linking', {})
    extracted_ids = {}
    
    for data_source, source_results in entity_linking.items():
        linked_orgs = source_results.get('linked_orgs', '')
        ids = []
        
        if linked_orgs:
            for org_entry in linked_orgs.split('|'):
                if '{' in org_entry and '}' in org_entry:
                    url_start = org_entry.find('{') + 1
                    url_end = org_entry.find('}')
                    url = org_entry[url_start:url_end]
                    
                    # Extract ID from URL based on data source
                    if data_source == 'ror' and 'ror.org' in url:
                        org_id = url.split('ror.org/')[-1]
                    elif data_source == 'spanish_hospitals' and 'numero=' in url:
                        org_id = url.split('numero=')[-1]
                    else:
                        org_id = url  # Use full URL as ID
                    
                    ids.append(org_id)
        
        extracted_ids[data_source] = ids
    
    return extracted_ids

# Example usage
results = affil_good.process(affiliations)
for result in results:
    ids_by_source = extract_ids_by_source(result)
    print(f"Organization IDs: {ids_by_source}")
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
    entity_linking = result['entity_linking']
    
    # Print results from all data sources
    for data_source, source_results in entity_linking.items():
        if source_results['linked_orgs']:
            print(f"  {data_source}: {source_results['linked_orgs']}")
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

for result in results:
    print(f"Processed: {result['raw_text']}")
    entity_linking = result['entity_linking']
    print(f"Entity linking: {entity_linking}")
```

## Processing Large Datasets

AffilGood has built-in batch processing capabilities that handle large datasets efficiently.

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
results = affil_good.process(affiliations)

# Extract organization IDs from results
def extract_all_ids(result):
    """Extract all organization IDs from all data sources."""
    all_ids = []
    entity_linking = result.get('entity_linking', {})
    
    for data_source, source_results in entity_linking.items():
        linked_orgs = source_results.get('linked_orgs', '')
        if linked_orgs:
            for org_entry in linked_orgs.split('|'):
                if '{' in org_entry and '}' in org_entry:
                    url_start = org_entry.find('{') + 1
                    url_end = org_entry.find('}')
                    url = org_entry[url_start:url_end]
                    all_ids.append(url)
    
    return all_ids

# Create results dataframe
processed_data = []
for result in results:
    ids = extract_all_ids(result)
    primary_id = ids[0] if ids else None
    
    processed_data.append({
        'affiliation': result['raw_text'],
        'primary_id': primary_id,
        'all_ids': '|'.join(ids) if ids else None
    })

results_df = pd.DataFrame(processed_data)
results_df.to_csv('processed_affiliations.csv', index=False)
print(f"Processed {len(results)} affiliations")
```

### Processing with Progress Tracking

```python
from affilgood import AffilGood
import pandas as pd
from tqdm import tqdm

def process_in_batches(affiliations, batch_size=100):
    """Process affiliations in batches with progress tracking."""
    affil_good = AffilGood()
    all_results = []
    
    # Process in batches
    for i in tqdm(range(0, len(affiliations), batch_size), desc="Processing batches"):
        batch = affiliations[i:i+batch_size]
        batch_results = affil_good.process(batch)
        all_results.extend(batch_results)
    
    return all_results

# Load data
df = pd.read_csv('large_dataset.csv')
affiliations = df['affiliation'].tolist()

# Process with progress tracking
results = process_in_batches(affiliations, batch_size=50)
print(f"Completed processing {len(results)} affiliations")
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

# Display detailed results
for result in results:
    print(f"Affiliation: {result['raw_text']}")
    entity_linking = result['entity_linking']
    
    for data_source, source_results in entity_linking.items():
        if source_results['linked_orgs']:
            print(f"  {data_source.upper()}: {source_results['linked_orgs']}")
            print(f"  Per span: {source_results['linked_orgs_spans']}")
    print("---")
```

### Performance-Optimized Configuration

```python
from affilgood import AffilGood
import time

# Configure for maximum speed
affil_good = AffilGood(
    span_separator=';',  # Use simple span splitting
    entity_linkers='Whoosh',  # Fast full-text search
    metadata_normalization=False,  # Disable for speed
    verbose=False,
    device='cpu'  # Use CPU to save GPU memory
)

# Process large dataset quickly
large_affiliation_list = ["University of California, Berkeley"] * 1000

start_time = time.time()
results = affil_good.process(large_affiliation_list)
end_time = time.time()

print(f"Processed {len(large_affiliation_list)} affiliations in {end_time - start_time:.2f} seconds")

# Show sample results
for i, result in enumerate(results[:3]):
    entity_linking = result['entity_linking']
    print(f"Sample {i+1}: {entity_linking}")
```

### Multi-Source Comprehensive Pipeline

```python
from affilgood import AffilGood

# Initialize with comprehensive configuration
affil_good = AffilGood(
    entity_linkers=['Dense', 'Whoosh'],  # Multiple linkers for better coverage
    return_scores=True,
    metadata_normalization=True,
    verbose=True
)

# Process diverse multilingual affiliations
diverse_affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Università degli Studi di Milano, Milano, Italia", 
    "Max-Planck-Institut für Intelligente Systeme, Tübingen, Deutschland",
    "Stanford University, Department of Computer Science, Stanford, CA, USA",
    "Hospital del Mar, Barcelona, Spain"
]

# Process with multilingual models
results = affil_good.process(diverse_affiliations)

# Analyze results comprehensively
def analyze_results(results):
    """Analyze entity linking results across all data sources."""
    analysis = {
        'total_affiliations': len(results),
        'successful_links': 0,
        'by_data_source': {},
        'multilingual_support': {}
    }
    
    for result in results:
        entity_linking = result.get('entity_linking', {})
        has_links = False
        
        for data_source, source_results in entity_linking.items():
            if source_results.get('linked_orgs'):
                has_links = True
                if data_source not in analysis['by_data_source']:
                    analysis['by_data_source'][data_source] = 0
                analysis['by_data_source'][data_source] += 1
        
        if has_links:
            analysis['successful_links'] += 1
    
    return analysis

# Perform analysis
analysis = analyze_results(results)
print(f"Analysis: {analysis}")

# Display detailed results
for i, result in enumerate(results):
    print(f"\nAffiliation {i+1}: {result['raw_text']}")
    entity_linking = result['entity_linking']
    
    for data_source, source_results in entity_linking.items():
        if source_results['linked_orgs']:
            print(f"  {data_source.upper()}: {source_results['linked_orgs']}")
```