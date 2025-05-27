# Getting Started with AffilGood

This guide will help you install AffilGood, understand its basic configuration options, and run your first affiliation processing pipeline.

## Installation

### Basic Installation

The simplest way to install AffilGood is via pip:

```bash
pip install affilgood
```

This will install the core AffilGood package and its essential dependencies.

### Development Installation

For development or to access the latest features, you can install directly from the GitHub repository:

```bash
git clone https://github.com/sirisacademic/affilgood.git
cd affilgood
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Dependencies

AffilGood has several dependencies that will be automatically installed:

- **Core NLP**: transformers, datasets, torch
- **Search**: whoosh, requests
- **Geolocation**: geopy, country_converter
- **Data processing**: pandas, numpy
- **LLM integration**: (optional for LLM reranking features)

Some optional features may require additional dependencies:

```bash
# For LLM-based reranking
pip install transformers>=4.30.0 optimum auto-gptq

# For WikiData integration
pip install SPARQLWrapper pycountry requests_cache
```

## Basic Configuration

The main entry point to AffilGood is the `AffilGood` class, which can be configured with various parameters to customize the pipeline.

### Minimal Configuration

```python
from affilgood import AffilGood

# Initialize with default settings
affil_good = AffilGood()
```

This creates an AffilGood instance with default settings:
- Model-based span identification
- Multilingual NER
- Whoosh-based entity linking
- Enabled metadata normalization

### Customizing Components

You can customize each component of the pipeline:

```python
from affilgood import AffilGood

# Customize component behavior
affil_good = AffilGood(
    # Span Identification options
    span_separator=';',  # Use character-based separation instead of model
    span_model_path='SIRIS-Lab/affilgood-span-multilingual',  # Or specify a model
    
    # NER options
    ner_model_path='SIRIS-Lab/affilgood-NER-multilingual',
    
    # Entity Linking options
    entity_linkers=['Whoosh', 'Dense'],  # Use multiple linkers
    return_scores=True,  # Include confidence scores in results
    
    # Metadata normalization
    metadata_normalization=True,
    use_cache_metadata_normalization=True,
    
    # General options
    verbose=True,  # Detailed logging
    device='cuda:0'  # Specific GPU, 'cpu', or None for auto-detect
)
```

### Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `span_separator` | str | Character used to split text into spans. If provided, uses SimpleSpanIdentifier instead of model-based identification. | `''` (model-based) |
| `span_model_path` | str | Path to the span identification model. | `'SIRIS-Lab/affilgood-span-multilingual'` |
| `ner_model_path` | str | Path to the NER model. | `'SIRIS-Lab/affilgood-NER-multilingual'` |
| `entity_linkers` | str/list | Entity linker(s) to use. Can be a string ('Whoosh', 'S2AFF', 'Dense') or a list of strings for multiple linkers. | `'Whoosh'` |
| `return_scores` | bool | Whether to include confidence scores in results. | `False` |
| `metadata_normalization` | bool | Whether to normalize location metadata. | `True` |
| `use_cache_metadata_normalization` | bool | Whether to use cached normalization data. | `True` |
| `verbose` | bool | Whether to display detailed logs. | `True` |
| `device` | str | Device for model inference. | `None` (auto-detect) |

## Processing Your First Affiliations

Now that you have AffilGood installed and configured, let's process some affiliation strings.

### Single Affiliation

```python
from affilgood import AffilGood

# Initialize with default settings
affil_good = AffilGood()

# Process a single affiliation
affiliation = "Department of Computer Science, University of Oxford, Oxford, UK"
result = affil_good.process([affiliation])

# Print the result
print(f"Spans: {result[0]['span_entities']}")
print(f"Organizations: {[ner.get('ORG', []) for ner in result[0]['ner']]}")
print(f"Entity Linking: {result[0]['entity_linking']}")
```

### Multiple Affiliations

```python
from affilgood import AffilGood

# Initialize with default settings
affil_good = AffilGood()

# Process multiple affiliations
affiliations = [
    "Department of Computer Science, University of Oxford, Oxford, UK",
    "Max Planck Institute for Intelligent Systems, Tübingen, Germany",
    "Departamento de Informática, Universidad de Chile, Santiago, Chile"
]

# Process all affiliations
results = affil_good.process(affiliations)

# Print the results
for i, result in enumerate(results):
    print(f"\nAffiliation {i+1}: {result['raw_text']}")
    print(f"  Spans: {result['span_entities']}")
    print(f"  Organizations: {[ner.get('ORG', []) for ner in result['ner']]}")
    print(f"  Entity Linking: {result['entity_linking']}")
    print("---")
```

## Step-by-Step Processing

You can also run each step of the pipeline separately:

```python
from affilgood import AffilGood

affil_good = AffilGood()

# Input affiliations
affiliations = [
    "Department of Computer Science, University of Oxford, Oxford, UK",
    "Max Planck Institute for Intelligent Systems, Tübingen, Germany"
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

## Understanding the Output

The output of the `process()` method is a list of dictionaries, one for each input affiliation string. Each dictionary contains:

- `raw_text`: The original affiliation string
- `span_entities`: A list of identified spans
- `ner`: Named entities recognized in each span
- `ner_raw`: Raw NER model output
- `osm`: Normalized location information
- `entity_linking`: Organization identifiers from different data sources
- `language_info`: Language detection and processing information

### New Output Structure

The entity linking results are now organized by data source:

```python
{
    "entity_linking": {
        "ror": {
            "linked_orgs_spans": [
                "Organization Name {https://ror.org/XXXXXX}:0.95",
                ""  # Empty if no match for this span
            ],
            "linked_orgs": "Organization Name {https://ror.org/XXXXXX}:0.95"
        },
        "spanish_hospitals": {
            "linked_orgs_spans": [
                "",
                "Hospital Name {https://www.sanidad.gob.es/...}:0.90"
            ],
            "linked_orgs": "Hospital Name {https://www.sanidad.gob.es/...}:0.90"
        }
        # Additional data sources as configured
    }
}
```

Example complete output structure:

```python
[
    {
        "raw_text": "Hospital del Mar, Barcelona; Department of Medicine, University Health Network, Mount Sinai Hospital, Toronto",
        "span_entities": [
            "Hospital del Mar, Barcelona",
            "Department of Medicine, University Health Network, Mount Sinai Hospital, Toronto"
        ],
        "ner": [
            {
                "ORG": ["Hospital del Mar"],
                "CITY": ["Barcelona"]
            },
            {
                "SUB": ["Department of Medicine", "University Health Network"],
                "ORG": ["Mount Sinai Hospital"],
                "CITY": ["Toronto"]
            }
        ],
        "ner_raw": [
            # Raw NER output for each span
        ],
        "osm": [
            {
                "CITY": "Barcelona",
                "COUNTY": "Barcelonès",
                "PROVINCE": "Barcelona", 
                "STATE": "Catalonia",
                "COUNTRY": "Spain",
                "COORDS": "('41.3825802', '2.1770730')",
                "OSM_ID": "347950"
            },
            {
                "CITY": "Toronto",
                "STATE_DISTRICT": "Golden Horseshoe",
                "STATE": "Ontario",
                "COUNTRY": "Canada", 
                "COORDS": "('43.6534817', '-79.3839347')",
                "OSM_ID": "324211"
            }
        ],
        "entity_linking": {
            "ror": {
                "linked_orgs_spans": [
                    "Hospital Del Mar {https://ror.org/03a8gac78}:0.95",
                    "Mount Sinai Hospital {https://ror.org/05deks119}:0.90"
                ],
                "linked_orgs": "Hospital Del Mar {https://ror.org/03a8gac78}:0.95|Mount Sinai Hospital {https://ror.org/05deks119}:0.90"
            },
            "spanish_hospitals": {
                "linked_orgs_spans": [
                    "Hospital del Mar. {https://www.sanidad.gob.es/ciudadanos/centros.do?metodo=realizarDetalle&tipo=hospital&numero=080057}:0.96",
                    ""
                ],
                "linked_orgs": "Hospital del Mar. {https://www.sanidad.gob.es/ciudadanos/centros.do?metodo=realizarDetalle&tipo=hospital&numero=080057}:0.96"
            }
        },
        "language_info": {}
    }
]
```

### Extracting Results

To extract organization identifiers from the new format:

```python
def extract_organization_ids(result, data_source="ror"):
    """Extract organization IDs from entity linking results."""
    entity_linking = result.get('entity_linking', {})
    source_results = entity_linking.get(data_source, {})
    linked_orgs = source_results.get('linked_orgs', '')
    
    # Parse the pipe-separated results
    organizations = []
    if linked_orgs:
        for org_entry in linked_orgs.split('|'):
            if '{' in org_entry and '}' in org_entry:
                # Extract the URL from the entry
                url_start = org_entry.find('{') + 1
                url_end = org_entry.find('}')
                url = org_entry[url_start:url_end]
                
                # Extract score if present
                score = None
                if ':' in org_entry[url_end:]:
                    score = float(org_entry.split(':')[-1])
                
                organizations.append({
                    'url': url,
                    'score': score,
                    'name': org_entry[:url_start-1].strip()
                })
    
    return organizations

# Example usage
results = affil_good.process(affiliations)
for result in results:
    ror_orgs = extract_organization_ids(result, "ror")
    hospital_orgs = extract_organization_ids(result, "spanish_hospitals")
    
    print(f"ROR organizations: {ror_orgs}")
    print(f"Spanish hospitals: {hospital_orgs}")
```

## Next Steps

Now that you have AffilGood up and running, you can explore more advanced features:

- [Using different entity linkers](entity-linking.md)
- [Working with multiple data sources](data-sources.md)
- [Processing multilingual affiliations](language-processing.md)
- [Optimizing performance](performance.md)
- [Exploring usage examples](usage-examples.md)