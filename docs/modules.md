# AffilGood Modules

This document provides detailed information about the AffilGood library's major components, their configuration options, and usage examples.

## Table of Contents

1. [Main AffilGood Class](#main-affilgood-class)
2. [Span Identification](#span-identification)
3. [Named Entity Recognition](#named-entity-recognition)
4. [Entity Linking](#entity-linking)

6. [Metadata Normalization](#metadata-normalization)
7. [Data Structures](#data-structures)

## Main AffilGood Class

The `AffilGood` class is the primary entry point for using the library. It orchestrates the complete pipeline from span identification to entity linking.

### Initialization

```python
from affilgood import AffilGood

affil_good = AffilGood(
    span_separator='',         # Optional: Character to split spans (e.g., ';')
    span_model_path=None,      # Optional: Custom path for span identification model
    ner_model_path=None,       # Optional: Custom path for NER model
    entity_linkers=None,       # Optional: Entity linker type ('Whoosh', 'S2AFF', or custom instance)
    return_scores=False,       # Optional: Return confidence scores with predictions
    metadata_normalization=True, # Optional: Enable location normalization
    verbose=False,             # Optional: Enable detailed logging
    device=None                # Optional: Device to use ('cpu', 'cuda', or None for auto-detect)
)
```

### Parameters

- **span_separator**: Character used to split spans if using `SimpleSpanIdentifier`. If provided, affiliation strings will be split on this character. Set to empty string to use model-based span identification.
- **span_model_path**: Path to a custom span identification model. Defaults to `nicolauduran45/affilgood-span-v2`.
- **ner_model_path**: Path to a custom NER model. Defaults to `nicolauduran45/affilgood-ner-multilingual-v2`.
- **entity_linkers**: Entity linker type to use. Can be a string ('Whoosh', 'S2AFF') or a custom linker instance. Defaults to 'Whoosh'.
- **return_scores**: Whether to return confidence scores with predictions. Defaults to `False`.
- **metadata_normalization**: Whether to enable location normalization. Defaults to `True`.
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
    model_path="nicolauduran45/affilgood-affiliation-span",
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
    model_path="nicolauduran45/affilgood-ner-multilingual-v2",
    device=0,  # GPU ID or 'cpu'
    chunk_size=10000,
    max_parallel=10,
    batch_size=64
)

results = ner.recognize_entities(spans)
```

### Entity Types

The NER model identifies the following entity types:

- **ORG**: Main organization (e.g., "University of California")
- **SUBORG**: Sub-organization or department (e.g., "Department of Chemistry")
- **CITY**: City name (e.g., "Berkeley")
- **REGION**: Region, state, or province (e.g., "California")
- **COUNTRY**: Country name (e.g., "USA")

## Entity Linking

AffilGood supports multiple entity linking strategies, orchestrated by the `EntityLinker` class. For detailed information about the entity linking architecture and how to extend it with custom implementations, see the [Entity Linking Module: Architecture and Extension Guide](entity-linking-extension.md).

```python
from affilgood.entity_linking.entity_linker import EntityLinker

# Initialize with multiple linkers
linker = EntityLinker(
    linkers=['Whoosh', 'S2AFF'],  # Use both Whoosh and S2AFF linkers
    return_scores=True            # Return confidence scores
)

results = linker.process_in_chunks(entities)
```

### Whoosh Linker

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

### S2AFF Linker

```python
from affilgood.entity_linking.s2aff_linker import S2AFFLinker

s2aff_linker = S2AFFLinker(
    data_manager=None,     # Optional: DataManager instance
    device="cpu",          # Optional: Device to use ('cpu' or 'cuda')
    ror_dump_path=None,    # Optional: Path to ROR dump file
    debug=False            # Optional: Enable debug output
)
```

### LLM Reranker

```python
from affilgood.entity_linking.llm_reranker import LLMReranker

reranker = LLMReranker(
    model_name="TheBloke/neural-chat-7B-v3-2-GPTQ",  # Optional: LLM model to use
    verbose=False                                     # Optional: Enable verbose output
)

best_match = reranker.rerank(affiliation, candidates)
```

## Metadata Normalization

AffilGood's metadata normalization module standardizes location information.

```python
from affilgood.metadata_normalization.normalizer import GeoNormalizer

normalizer = GeoNormalizer(cache_fname='affilgood/metadata_normalization/cache.csv')
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
                "SUBORG": ["Department name"],
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
