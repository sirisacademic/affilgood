# AffilGood Modules

This document provides detailed information about the AffilGood library's major components, their configuration options, and usage examples.

## Table of Contents

1. [Main AffilGood Class](#main-affilgood-class)
2. [Span Identification](#span-identification)
3. [Named Entity Recognition](#named-entity-recognition)
4. [Entity Linking](#entity-linking)
5. [Metadata Normalization](#metadata-normalization)
6. [Data Structures](#data-structures)

## Main AffilGood Class

The `AffilGood` class is the primary entry point for using the library. It orchestrates the complete pipeline from span identification to entity linking.

### Initialization

```python
from affilgood import AffilGood

affil_good = AffilGood(
    span_separator='',         # Optional: Character to split spans (e.g., ';')
    span_model_path=None,      # Optional: Custom path for span identification model
    ner_model_path=None,       # Optional: Custom path for NER model
    entity_linkers=None,       # Optional: Entity linker type(s) or custom instance(s)
    return_scores=False,       # Optional: Return confidence scores with predictions
    metadata_normalization=True, # Optional: Enable location normalization
    use_cache_metadata_normalization=True, # Optional: Use cached normalization data
    verbose=False,             # Optional: Enable detailed logging
    device=None                # Optional: Device to use ('cpu', 'cuda', or None for auto-detect)
)
```

### Parameters

- **span_separator**: Character used to split spans if using `SimpleSpanIdentifier`. If provided, affiliation strings will be split on this character. Set to empty string to use model-based span identification.
- **span_model_path**: Path to a custom span identification model. Defaults to `'SIRIS-Lab/affilgood-span-multilingual'`.
- **ner_model_path**: Path to a custom NER model. Defaults to `'SIRIS-Lab/affilgood-NER-multilingual'`.
- **entity_linkers**: Entity linker(s) to use. Can be a string ('Whoosh', 'S2AFF', 'Dense'), a list of strings for multiple linkers, or a custom linker instance. Defaults to 'Whoosh'.
- **return_scores**: Whether to return confidence scores with predictions. Defaults to `False`.
- **metadata_normalization**: Whether to enable location normalization. Defaults to `True`.
- **use_cache_metadata_normalization**: Whether to use cached normalization data. Defaults to `True`.
- **verbose**: Whether to enable detailed logging. Defaults to `False`.
- **device**: Device to use for model inference. Defaults to auto-detect.

### Methods

#### process()

Executes the complete pipeline: span identification, NER, metadata normalization, and entity linking.

```python
results = affil_good.process(text)
```

- **text**: String or list of strings containing affiliation text.
- **Returns**: List of dictionaries containing processed results.

#### get_span()

Identifies spans within the input text.

```python
spans = affil_good.get_span(text)
```

- **text**: String or list of strings containing affiliation text.
- **Returns**: List of dictionaries with "raw_text" and "span_entities" fields.

#### get_ner()

Performs named entity recognition on identified spans.

```python
entities = affil_good.get_ner(spans)
```

- **spans**: Output from `get_span()`.
- **Returns**: List of dictionaries with "raw_text", "span_entities", and "ner" fields.

#### get_normalization()

Normalizes location metadata in recognized entities.

```python
normalized = affil_good.get_normalization(entities)
```

- **entities**: Output from `get_ner()`.
- **Returns**: List of dictionaries with normalized location information added.

#### get_entity_linking()

Links recognized entities to external identifiers (e.g., ROR IDs).

```python
linked = affil_good.get_entity_linking(normalized)
```

- **normalized**: Output from `get_normalization()`.
- **Returns**: List of dictionaries with entity linking information added.

## Span Identification

AffilGood provides three different span identification approaches:

### 1. SpanIdentifier

Model-based span identification using a transformer model.

```python
from affilgood.span_identification.span_identifier import SpanIdentifier

span_identifier = SpanIdentifier(
    model_path="SIRIS-Lab/affilgood-span-multilingual",
    device=0,  # GPU ID or 'cpu'
    chunk_size=10000,
    max_parallel=10,
    batch_size=64,
    threshold_score=0.75,
    fix_predicted_words=True,
    title_case=False
)

results = span_identifier.identify_spans(text_list)
```

### 2. SimpleSpanIdentifier

Character-based span splitting (e.g., splitting by semicolons).

```python
from affilgood.span_identification.simple_span_identifier import SimpleSpanIdentifier

span_identifier = SimpleSpanIdentifier(separator=";")
results = span_identifier.identify_spans(text_list)
```

### 3. NoopSpanIdentifier

Pass-through identifier for pre-segmented data.

```python
from affilgood.span_identification.noop_span_identifier import NoopSpanIdentifier

span_identifier = NoopSpanIdentifier()
results = span_identifier.identify_spans(text_list)
```

## Named Entity Recognition

AffilGood's NER module identifies organizations, sub-organizations, and locations in affiliation text.

```python
from affilgood.ner.ner import NER

ner = NER(
    model_path="SIRIS-Lab/affilgood-NER-multilingual",
    device=0,  # GPU ID or 'cpu'
    chunk_size=10000,
    max_parallel=10,
    batch_size=64,
    fix_predicted_words=True,
    title_case=False
)

results = ner.recognize_entities(spans)
```

### Entity Types

The NER model identifies the following entity types:

- **ORG**: Main organization (e.g., "University of California")
- **SUB**: Sub-organization or department (e.g., "Department of Chemistry")
- **CITY**: City name (e.g., "Berkeley")
- **REGION**: Region, state, or province (e.g., "California")
- **COUNTRY**: Country name (e.g., "USA")

## Entity Linking

AffilGood supports multiple entity linking strategies, orchestrated by the `EntityLinker` class.

```python
from affilgood.entity_linking.entity_linker import EntityLinker

# Initialize with multiple linkers
linker = EntityLinker(
    linkers=['Whoosh', 'Dense'],  # Use multiple linkers
    return_scores=True                  # Return confidence scores
)

results = linker.process_in_chunks(entities)
```

### Available Linkers

#### Whoosh Linker

Uses the Whoosh full-text search engine to match organizations against an index.

```python
from affilgood.entity_linking.whoosh_linker import WhooshLinker

whoosh_linker = WhooshLinker(
    data_manager=None,           # Optional: DataManager instance
    index_dir=None,              # Optional: Path to Whoosh index directory
    rebuild_index=False,         # Optional: Rebuild index if it exists
    threshold_score=0.25,        # Optional: Minimum score for matches
    rerank=True,                 # Optional: Use LLM for reranking
    rerank_model_name=None,      # Optional: LLM model for reranking
    number_candidates_rerank=5,  # Optional: Number of candidates to rerank
    max_hits=10,                 # Optional: Maximum number of search results
    debug=False                  # Optional: Enable debug output
)
```

#### S2AFF Linker

Integrates with the Semantic Scholar Affiliation Matching system.

```python
from affilgood.entity_linking.s2aff_linker import S2AFFLinker

s2aff_linker = S2AFFLinker(
    data_manager=None,     # Optional: DataManager instance
    ror_dump_path=None,    # Optional: Path to ROR dump file
    debug=False            # Optional: Enable debug output
)
```

#### Dense Linker

Uses dense vector representations and semantic similarity for matching.

```python
from affilgood.entity_linking.dense_linker import DenseLinker

dense_linker = DenseLinker(
    data_manager=None,           # Optional: DataManager instance
    encoder_path=None,           # Optional: Path to encoder model
    batch_size=32,               # Optional: Batch size for encoding
    scores_span_text=False,      # Optional: Score spans directly
    return_num_candidates=10,    # Optional: Number of candidates to return
    threshold_score=0.30,        # Optional: Minimum score threshold
    use_hnsw=True,               # Optional: Use HNSW for efficient search
    rebuild_index=False,         # Optional: Rebuild index if exists
    debug=False,                 # Optional: Enable debug output
    use_cache=True,              # Optional: Use caching for results
    data_source="ror"            # Optional: Data source to use
)
```

### Rerankers

#### Direct Pair Reranker

Compares affiliation text directly with candidate organizations.

```python
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker

reranker = DirectPairReranker(
    model_name="ENCODER_MODEL_NAME",  # Cross-encoder model for reranking
    batch_size=32,                    # Batch size for processing
    max_length=256,                   # Maximum sequence length
    device=None,                      # Device to use (auto-detect if None)
    hnsw_metadata_path=None,          # Path to HNSW metadata
    reranking_strategy="max_score",   # Strategy for combining scores
    use_special_tokens=False,         # Whether to use special tokens
    use_cache=True,                   # Whether to cache results
    debug=False                       # Enable debug output
)

reranked_results = reranker.rerank(organization, candidates)
```

#### LLM Reranker

Uses Large Language Models to select the best candidate match.

```python
from affilgood.entity_linking.llm_reranker import LLMReranker

reranker = LLMReranker(
    model_name="TheBloke/neural-chat-7B-v3-2-GPTQ",  # LLM model to use
    verbose=False                                     # Enable verbose output
)

best_match = reranker.rerank(affiliation, candidates)
```

### Data Sources

AffilGood supports multiple data sources through the `DataSourceRegistry`:

```python
from affilgood.entity_linking import DataSourceRegistry

# Get a handler for a specific data source
ror_handler = DataSourceRegistry.get_handler("ror")
wikidata_handler = DataSourceRegistry.get_handler("wikidata")
spanish_hospitals_handler = DataSourceRegistry.get_handler("spanish_hospitals")
sicris_handler = DataSourceRegistry.get_handler("sicris")

# Get all available handlers
all_handlers = DataSourceRegistry.get_all_handlers()
```

#### Creating Custom Data Sources

You can create custom data sources by implementing the `DataSourceHandler` interface:

```python
from affilgood.entity_linking import DataSourceHandler, DataSourceRegistry

@DataSourceRegistry.register
class CustomDataSourceHandler(DataSourceHandler):
    @property
    def source_id(self):
        return "custom_source"
    
    def load_data(self, config):
        # Implementation
        pass
    
    def get_data_for_indexing(self, config, indices_type='whoosh', **kwargs):
        # Implementation
        pass
    
    def map_organization(self, org):
        # Implementation
        pass
    
    def format_id_url(self, org_id):
        # Implementation
        pass
```

### Language Processing

AffilGood includes language detection and translation capabilities:

```python
from affilgood.entity_linking.llm_translator import LLMTranslator

translator = LLMTranslator(
    skip_english=True,         # Skip translation for English text
    model_name=None,           # Model to use (defaults to a suitable model)
    use_external_api=False,    # Whether to use an external API
    verbose=False,             # Enable verbose output
    use_cache=True             # Use caching for translations
)

translated_text = translator.translate("Universidad de Barcelona, Barcelona, España")
```

## Metadata Normalization

AffilGood's metadata normalization module standardizes location information.

```python
from affilgood.metadata_normalization.normalizer import GeoNormalizer

normalizer = GeoNormalizer(
    use_cache=True,
    cache_fname='affilgood/metadata_normalization/cache.csv'
)
normalized = normalizer.normalize(entities)
```

### Key Features

- **Country Normalization**: Maps country names to standard names and ISO codes
- **Location Resolution**: Uses OpenStreetMap to resolve and standardize location names
- **Geocoding**: Adds geographic coordinates to locations
- **Language Translation**: Translates location names to English for better matching
- **Caching**: Stores previous lookups for efficiency

## Data Structures

### Span Identification Output

```python
[
    {
        "raw_text": "Original affiliation string",
        "span_entities": ["Span 1", "Span 2", ...]
    },
    ...
]
```

### Named Entity Recognition Output

```python
[
    {
        "raw_text": "Original affiliation string",
        "span_entities": ["Span 1", "Span 2", ...],
        "ner": [
            {
                "ORG": ["Organization name"],
                "SUB": ["Department name"],
                "CITY": ["City name"],
                "REGION": ["Region name"],
                "COUNTRY": ["Country name"]
            },
            ...
        ],
        "ner_raw": [...]  # Raw NER results
    },
    ...
]
```

### Metadata Normalization Output

```python
[
    {
        "raw_text": "Original affiliation string",
        "span_entities": ["Span 1", "Span 2", ...],
        "ner": [...],
        "ner_raw": [...],
        "osm": [
            {
                "CITY": "Normalized city name",
                "STATE_DISTRICT": "State district",
                "COUNTY": "County name",
                "PROVINCE": "Province name",
                "STATE": "State name",
                "REGION": "Region name",
                "COUNTRY": "Normalized country name",
                "COORDS": [latitude, longitude],
                "OSM_ID": "OpenStreetMap ID"
            },
            ...
        ]
    },
    ...
]
```

### Entity Linking Output

```python
[
    {
        "raw_text": "Original affiliation string",
        "span_entities": ["Span 1", "Span 2", ...],
        "ner": [...],
        "ner_raw": [...],
        "osm": [...],
        "ror": [
            "Organization Name {https://ror.org/XXXXXX}:0.95",
            ...
        ]
    },
    ...
]
```

If `return_scores=False`, the ROR results will not include confidence scores:

```python
"ror": ["Organization Name {https://ror.org/XXXXXX}", ...]
```
