# Entity Linking in AffilGood

The entity linking module is a core component of the AffilGood pipeline, responsible for matching identified organizations with standard identifiers from various data sources. This document provides a comprehensive guide to the entity linking architecture, available strategies, and configuration options.

## Table of Contents

1. [Entity Linking Architecture](#entity-linking-architecture)
2. [Available Linking Strategies](#available-linking-strategies)
3. [Combining Multiple Retrievers](#combining-multiple-retrievers)
4. [Reranking Mechanisms](#reranking-mechanisms)
5. [Supported Data Sources](#supported-data-sources)
6. [Configuration Options](#configuration-options)
7. [Performance Considerations](#performance-considerations)
8. [Advanced Usage](#advanced-usage)

## Entity Linking Architecture

The AffilGood entity linking system follows a modular architecture with several key components:

1. **EntityLinker**: Main orchestrator that manages the entity linking process
2. **BaseLinker**: Abstract base class for all entity linkers
3. **Retrievers**: Components that find candidate matches (e.g., Whoosh, S2AFF, Dense)
4. **Rerankers**: Components that improve ranking of candidate matches
5. **DataSourceHandlers**: Components that provide data from different sources

The system uses a multi-stage approach:

1. **Candidate Retrieval**: Finding potential matches for an organization
2. **Candidate Combination**: Combining results from multiple retrievers (optional)
3. **Reranking**: Refining the ranking of candidates
4. **Result Selection**: Selecting the best match based on scores

![Entity Linking Architecture](entity_linking_architecture.png)

## Available Linking Strategies

AffilGood provides several entity linking strategies, each with unique strengths:

### Dense Linker (default retriever)

The `DenseLinker` uses dense vector representations and semantic similarity for matching.

**Key Features:**
- Semantic understanding of organization names
- Better handling of variations and translations
- Support for multiple languages
- Efficient search using HNSW indices

**Example:**
```python
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.data_manager import DataManager

data_manager = DataManager()
dense_linker = DenseLinker(
    data_manager=data_manager,
    encoder_path=None,  # Uses default encoder
    threshold_score=0.35,
    use_hnsw=True,
    data_source="ror"  # Can also use "wikidata" or custom sources
)

# Use directly
results = dense_linker.get_single_prediction(organization)

# Or with AffilGood
from affilgood import AffilGood
affil_good = AffilGood(entity_linkers='Dense')
```

### Whoosh Linker

The `WhooshLinker` uses the Whoosh full-text search engine to match organizations against an index.

**Key Features:**
- Fast full-text search capabilities
- Support for fuzzy matching
- Handling of acronyms and name variations
- Location-aware matching
- Optional LLM-based reranking

**Example:**
```python
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.data_manager import DataManager

data_manager = DataManager()
whoosh_linker = WhooshLinker(
    data_manager=data_manager,
    threshold_score=0.3,
    rerank=True,
    number_candidates_rerank=7
)

# Use directly
results = whoosh_linker.get_single_prediction(organization)

# Or with AffilGood
from affilgood import AffilGood
affil_good = AffilGood(entity_linkers='Whoosh')
```

### S2AFF Linker

The `S2AFFLinker` integrates with the Semantic Scholar Affiliation Matching system.

**Key Features:**
- Two-stage candidate retrieval and ranking
- Integration with the S2AFF ecosystem
- Support for partial matches and variations
- Built-in reranking using LightGBM

**Example:**
```python
from affilgood.entity_linking.s2aff_linker import S2AFFLinker
from affilgood.entity_linking.data_manager import DataManager

data_manager = DataManager()
s2aff_linker = S2AFFLinker(
    data_manager=data_manager,
    debug=False
)

# Use directly
results = s2aff_linker.get_single_prediction(organization)

# Or with AffilGood
from affilgood import AffilGood
affil_good = AffilGood(entity_linkers='S2AFF')
```

## Combining Multiple Retrievers

A powerful feature of AffilGood is the ability to combine results from multiple retrievers to improve accuracy. This approach leverages the strengths of different matching strategies.

**Example:**
```python
from affilgood import AffilGood

# Use multiple retrievers
affil_good = AffilGood(
    entity_linkers=['Whoosh', 'Dense'],
    return_scores=True
)

# Process affiliations
results = affil_good.process(affiliations)
```

*This is currently possible only with WhooshLinker and DenseLinker (not with S2AFFLinker)* 

### How Combination Works

1. Each retriever independently finds candidate matches for an organization
2. Results are merged based on entity identifiers
3. Scores are combined using a weighted approach
4. The combined candidates are optionally reranked
5. The best match is selected based on the final scores

This approach is particularly effective for complex or ambiguous cases where a single retriever might miss the correct match.

## Reranking Mechanisms

AffilGood includes reranking mechanisms to improve the quality of entity linking results.

### Direct Pair Reranker (default reranker)

The `DirectPairReranker` compares affiliation strings directly with candidate organizations.

**Key Features:**
- Text normalization and variant generation
- Direct comparison of multiple text variants
- Score adjustment based on match quality
- Caching for performance

**Example:**
```python
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker

reranker = DirectPairReranker(
    model_name="cross-encoder/stsb-roberta-base",
    reranking_strategy="max_score",
    use_cache=True
)

reranked_results = reranker.rerank(organization, candidates)
```

### LLM Reranker

The `LLMReranker` uses large language models to select the best candidate organization.

**Key Features:**
- Context-aware selection using LLMs
- Handling of complex organizational structures
- Better disambiguation of similar organizations
- Support for multilingual affiliations

**Example:**
```python
from affilgood.entity_linking.llm_reranker import LLMReranker

reranker = LLMReranker(
    model_name="TheBloke/neural-chat-7B-v3-2-GPTQ"
)

best_match = reranker.rerank(affiliation, candidates)
```

## Supported Data Sources

AffilGood supports multiple data sources through the `DataSourceRegistry`:

### Research Organization Registry (ROR)

ROR is the default data source and provides a comprehensive registry of research organizations.

**Example:**
```python
from affilgood.entity_linking import DataSourceRegistry

# Get ROR handler
ror_handler = DataSourceRegistry.get_handler("ror")

# Configure a linker with ROR data
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(data_source="ror")
```

### WikiData

WikiData integration allows linking to a broader range of organizations with rich metadata.

**Example:**
```python
from affilgood.entity_linking import DataSourceRegistry
from affilgood.entity_linking.wikidata_dump_generator import WikidataDumpGenerator

# Initialize WikiData generator
wikidata_generator = WikidataDumpGenerator(verbose=True)

# Generate index for specific countries and organization types
df = wikidata_generator.get_index(
    countries=["Spain", "Germany"],
    org_types=["university", "hospital"]
)

# Use WikiData as a data source
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(data_source="wikidata")
```

### Spanish Hospitals

A specialized data source for Spanish hospitals.

**Example:**
```python
from affilgood.entity_linking import DataSourceRegistry

# Get Spanish hospitals handler
spanish_hospitals_handler = DataSourceRegistry.get_handler("spanish_hospitals")

# Configure with custom settings
spanish_hospitals_handler.initialize({
    'file_path': "path/to/spanish_hospitals.xlsx",
    'verbose': True
})
```

### SICRIS (Slovenian Research Organizations)

A specialized data source for Slovenian research organizations.

**Example:**
```python
from affilgood.entity_linking import DataSourceRegistry

# Get SICRIS handler
sicris_handler = DataSourceRegistry.get_handler("sicris")

# Configure with custom settings
sicris_handler.initialize({
    'file_path': "path/to/sicris_registry.tsv",
    'verbose': True
})
```

### Custom Data Sources

You can implement custom data sources by extending the `DataSourceHandler` interface:

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

## Configuration Options

### EntityLinker Configuration

```python
from affilgood.entity_linking.entity_linker import EntityLinker

linker = EntityLinker(
    linkers=['Whoosh', 'Dense'],  # Linkers to use
    return_scores=True,                # Whether to return scores
    output_dir='output_chunks'         # Directory for partial results
)
```

### WhooshLinker Configuration

```python
whoosh_linker = WhooshLinker(
    data_manager=data_manager,
    index_dir=None,              # Path to index directory
    rebuild_index=False,         # Whether to rebuild index
    threshold_score=0.25,        # Minimum score threshold
    rerank=True,                 # Whether to use reranking
    rerank_model_name=None,      # Model for reranking
    number_candidates_rerank=5,  # Candidates to rerank
    max_hits=10,                 # Maximum search results
    debug=False                  # Debug output
)
```

### DenseLinker Configuration

```python
dense_linker = DenseLinker(
    data_manager=data_manager,
    encoder_path=None,           # Path to encoder model
    batch_size=32,               # Batch size for encoding
    scores_span_text=False,      # Score spans directly
    return_num_candidates=10,    # Candidates to return
    threshold_score=0.30,        # Minimum score threshold
    use_hnsw=True,               # Use HNSW for search
    rebuild_index=False,         # Rebuild index if exists
    debug=False,                 # Debug output
    use_cache=True,              # Use caching
    data_source="ror"            # Data source to use
)
```

### DirectPairReranker Configuration

```python
reranker = DirectPairReranker(
    model_name="cross-encoder/stsb-roberta-base",  # Model for reranking
    batch_size=32,                    # Batch size
    max_length=256,                   # Maximum sequence length
    device=None,                      # Device to use
    hnsw_metadata_path=None,          # Path to HNSW metadata
    reranking_strategy="max_score",   # Strategy for scores
    use_special_tokens=False,         # Use special tokens
    use_cache=True,                   # Use caching
    debug=False                       # Debug output
)
```

## Performance Considerations

### Linker Strengths and Weaknesses

| Linker | Strengths | Weaknesses | Best For |
|--------|-----------|------------|----------|
| Whoosh | Fast, good for exact matches | Less effective for variations | Large datasets, English text |
| S2AFF | Good precision, handles variations | May miss some matches | Academic affiliations |
| Dense | Best for variations and translations | Computationally intensive | Multilingual data, complex names |

### Combination Strategies

| Combination | Strengths | Considerations |
|-------------|-----------|----------------|
| Whoosh + Dense | Balances speed and accuracy | Higher resource usage |
| All Linkers | Highest accuracy | Highest resource usage |
| Whoosh + LLM Reranker | Good balance of speed and quality | Requires LLM integration |
| Dense + Direct Reranker | Best for multilingual data | Higher computational cost |

### Memory and Performance

- **Whoosh Linker**: Low memory usage (~100-200MB)
- **S2AFF Linker**: Moderate memory usage (~500MB-1GB)
- **Dense Linker**: Higher memory usage (~1-2GB)
- **LLM Reranker**: Significant memory usage (4-10GB depending on model)

For best performance:
- Use caching (enabled by default)
- Process in batches for large datasets
- Consider hardware acceleration for model-based components

## Advanced Usage

### Custom Entity Linking Pipeline

You can create a custom entity linking pipeline by combining different components:

```python
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker
from affilgood.entity_linking.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Create linkers
whoosh_linker = WhooshLinker(
    data_manager=data_manager,
    threshold_score=0.25,
    rerank=False  # We'll use our own reranker
)

dense_linker = DenseLinker(
    data_manager=data_manager,
    threshold_score=0.30,
    data_source="ror"
)

# Create reranker
reranker = DirectPairReranker(
    model_name="cross-encoder/stsb-roberta-base",
    reranking_strategy="max_score"
)

# Attach reranker to dense linker
dense_linker.reranker = reranker

# Create entity linker with both linkers
entity_linker = EntityLinker(
    linkers=[whoosh_linker, dense_linker],
    return_scores=True
)

# Process entities
results = entity_linker.process_in_chunks(entities)
```

### Working with Multiple Data Sources

You can use different data sources for different linkers:

```python
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Create linkers with different data sources
ror_linker = DenseLinker(
    data_manager=data_manager,
    data_source="ror"
)

wikidata_linker = DenseLinker(
    data_manager=data_manager,
    data_source="wikidata"
)

spanish_hospitals_linker = DenseLinker(
    data_manager=data_manager,
    data_source="spanish_hospitals"
)

# Create entity linker with all linkers
entity_linker = EntityLinker(
    linkers=[ror_linker, wikidata_linker, spanish_hospitals_linker],
    return_scores=True
)

# Process entities
results = entity_linker.process_in_chunks(entities)
```

### Custom Reranking Logic

You can implement custom reranking logic by extending the `BaseReranker` class:

```python
from affilgood.entity_linking.base_reranker import BaseReranker

class CustomReranker(BaseReranker):
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
    
    def rerank(self, affiliation, candidates):
        # Your custom reranking logic
        # ...
        return reranked_candidates
```
