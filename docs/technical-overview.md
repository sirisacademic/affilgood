# AffilGood: Technical Overview

This document provides a technical overview of the AffilGood library architecture, workflows, and implementation details.

## System Architecture

AffilGood implements a modular pipeline architecture with four main components:

1. **Span Identification**: Segmenting affiliation strings into individual institution spans
2. **Named Entity Recognition (NER)**: Identifying organizations, sub-organizations, and locations
3. **Metadata Normalization**: Standardizing location information
4. **Entity Linking**: Connecting organizations to standard identifiers (e.g., ROR IDs)

Each component can be used independently or as part of the complete pipeline.

![Architecture Diagram](figure1.png)

## Workflow Details

### 1. Span Identification

Span identification is the process of segmenting affiliation strings into individual institution spans. For example, an author may list multiple affiliations in a single string, and this step separates them.

AffilGood provides three implementations:

1. **SpanIdentifier**: Uses a transformer model to identify spans. This is the most sophisticated approach and works well with complex, unstructured text.
2. **SimpleSpanIdentifier**: Splits text based on a separator character (e.g., semicolon). This is faster but less accurate for complex cases.
3. **NoopSpanIdentifier**: Treats each input text as a single span. Useful when data is already pre-segmented.

The transformer-based approach uses a sequence labeling model that identifies the boundaries of affiliation spans. It's implemented using the Hugging Face transformers library with a token classification head.

### 2. Named Entity Recognition (NER)

The NER component identifies key entities within each affiliation span:

- **ORG**: Main organization (e.g., university, company)
- **SUBORG**: Sub-organization or department
- **CITY**: City location
- **REGION**: State, province, or region
- **COUNTRY**: Country

AffilGood uses a fine-tuned transformer model for NER. The models are available in both English-specific and multilingual variants. The multilingual model can recognize entities across multiple languages, which is essential for international research affiliations.

Implementation details:
- Fine-tuned from RoBERTa (English) or XLM-RoBERTa (multilingual) base models
- Trained on manually annotated affiliation data
- Uses BIO tagging scheme to handle multi-token entities
- Processes batches of spans for efficiency

### 3. Metadata Normalization

The metadata normalization component standardizes location information using several techniques:

1. **Country Normalization**: Maps country names to standard names and ISO codes using country_converter and custom mappings
2. **Language Translation**: Translates non-English location names to English for better matching
3. **Geocoding**: Uses OpenStreetMap's Nominatim service to resolve locations and extract structured information
4. **Caching**: Maintains a cache of previous lookups to reduce API calls and improve performance

This step is crucial for handling variations in location names (e.g., "USA", "United States", "États-Unis") and multilingual texts.

### 4. Entity Linking

Entity linking connects the identified organizations to standard identifiers. AffilGood supports multiple linking strategies:

1. **WhooshLinker**: Uses Whoosh, a pure-Python search engine, to match organizations against the Research Organization Registry (ROR) database. Features include:
   - Full-text search with boosting for exact matches
   - Handling of name variants and translations
   - Location-aware matching
   - LLM-based reranking of candidate matches

2. **S2AFFLinker**: Integrates with the S2AFF (Semantic Scholar Affiliation Matching) system. Features include:
   - Two-stage candidate retrieval and ranking
   - Pairwise re-ranking using LightGBM
   - Integration with OpenAlex work counts for popularity-based scoring

The entity linking process generally follows these steps:
1. Extract main organization, sub-organization, and location information
2. Generate search queries based on this information
3. Retrieve candidate matches from the index
4. Score and rank candidates
5. Optional: Re-rank candidates using an LLM

## Implementation Details

### Data Management

The `DataManager` class handles data-related operations:

- Downloading and caching ROR data
- Building and maintaining search indices
- Syncing with S3 for S2AFF integration
- Updating OpenAlex work counts
- Creating and managing Whoosh indices

### Parallelization and Chunking

AffilGood processes data in chunks to enable efficient parallelization and memory management:

- Input data is divided into manageable chunks
- Chunks are processed in parallel using ThreadPoolExecutor
- Results are merged back together
- Optional intermediate saving of chunk results

This approach makes it possible to process large datasets efficiently on modest hardware.

### LLM-based Reranking

The `LLMReranker` class provides a novel approach to improving entity linking accuracy:

- Takes a list of candidate matches for an affiliation
- Prompts an LLM to select the best match based on contextual understanding
- Returns the ROR ID of the selected candidate
- Improves precision especially for ambiguous cases

By default, it uses a quantized LLM model ("TheBloke/neural-chat-7B-v3-2-GPTQ") that can run on consumer hardware.

### Multilingual Support

AffilGood has extensive multilingual support:

- The span identification and NER models are available in multilingual variants
- Location normalization includes translation capabilities
- Entity linking considers name variants across languages
- Institution name mappings include translations for academic terms

This makes AffilGood suitable for processing global scientific literature.

## Performance Considerations

### Memory Usage

- The transformer models used for span identification and NER can use significant memory (1-2GB each)
- The Whoosh index for ROR data is relatively lightweight (~100MB)
- The LLM reranker, if used, requires additional memory (4-10GB depending on the model)

### Processing Speed

Approximate processing times per affiliation string on modern hardware:

- Span identification: 5-20ms
- NER: 10-30ms
- Metadata normalization: 50-500ms (depends on cache hits)
- Entity linking: 100-500ms
- LLM reranking: 200-1000ms

For batch processing, parallelization significantly improves throughput.

### Storage Requirements

- Base library: ~50MB
- Downloaded models: ~1-2GB
- ROR data and indices: ~500MB
- Caches: Variable (grows with usage)

### Optimizations

- Caching of geocoding results
- Caching of entity linking results
- Parallelized chunk processing
- Efficient metadata lookup with dictionaries
- Translation caching to minimize API calls

## Extension Points

AffilGood is designed to be extensible. Key extension points include:

1. **Custom Span Identifiers**: Implement a class with an `identify_spans` method following the interface in BaseSpanIdentifier

2. **Custom Entity Linkers**: Extend the `BaseLinker` class and implement required methods:
   - `initialize()`
   - `get_single_prediction()`

3. **Custom Rerankers**: Extend the `BaseReranker` class and implement the `rerank()` method

4. **Alternative Models**: Use custom models by specifying paths during initialization:
   - `span_model_path`
   - `ner_model_path`
   - `rerank_model_name`

## Limitations and Future Work

### Current Limitations

- Dependency on external services for geocoding
- Limited support for non-academic organizations
- No support for historical organization changes
- Large memory requirements for full functionality with LLM reranking

### Planned Enhancements

- End-to-end fine-tuning of the complete pipeline
- Improved handling of multilingual affiliations
- Enhanced offline capabilities with bundled location data
- Better support for historical and temporal aspects of organizations
- Reduced model sizes through distillation and pruning
- Integration with additional organization identifier systems beyond ROR

## Acronyms and Terminology

- **ROR**: Research Organization Registry - a global, open registry of research organizations
- **NER**: Named Entity Recognition - identifying organizations, locations, etc. in text
- **S2AFF**: Semantic Scholar Affiliation Matching - a system for matching affiliations
- **OSM**: OpenStreetMap - source of geographic data
- **LLM**: Large Language Model - used for reranking in entity linking
- **Span**: A single affiliation string, potentially containing multiple entities
