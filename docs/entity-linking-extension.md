# Entity Linking Module: Architecture and Extension Guide

This document explains the class structure of the AffilGood entity linking module and provides guidance on extending it with custom implementations.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Class Hierarchy](#class-hierarchy)
- [Core Classes](#core-classes)
  - [EntityLinker](#entitylinker)
  - [BaseLinker](#baselinker)
  - [BaseReranker](#basereranker)
  - [DataManager](#datamanager)
- [Default Implementations](#default-implementations)
  - [WhooshLinker](#whooshlinker)
  - [S2AFFLinker](#s2afflinker)
  - [LLMReranker](#llmreranker)
- [Extension Guide](#extension-guide)
  - [Creating a Custom Linker](#creating-a-custom-linker)
  - [Creating a Custom Reranker](#creating-a-custom-reranker)
  - [Integration with EntityLinker](#integration-with-entitylinker)
- [Example Extensions](#example-extensions)
  - [Example: Elastic-based Linker](#example-elastic-based-linker)
  - [Example: Rule-based Reranker](#example-rule-based-reranker)

## Architecture Overview

The entity linking module follows a modular design with the following key components:

1. **EntityLinker**: Main orchestrator that manages multiple linker instances
2. **BaseLinker**: Abstract base class for all entity linkers
3. **BaseReranker**: Abstract base class for all reranking mechanisms
4. **DataManager**: Utility class for managing data resources

This architecture allows for:
- Multiple linking strategies to be used simultaneously
- Extensibility through custom linker implementations
- Optional reranking mechanisms to improve matching accuracy
- Central data management for caching and resource sharing

## Class Hierarchy

```
EntityLinker
├── manages multiple BaseLinker instances
│
BaseLinker (abstract)
├── WhooshLinker
│   └── uses LLMReranker (optional)
└── S2AFFLinker
    └── uses built-in PairwiseRORLightGBMReranker

BaseReranker (abstract)
└── LLMReranker

DataManager
└── used by linkers for data access
```

## Core Classes

### EntityLinker

The `EntityLinker` class is the main orchestrator of the entity linking process.

**Key Responsibilities:**
- Managing multiple linker instances
- Dividing input into manageable chunks
- Parallelizing processing for efficiency
- Merging results from different linkers
- Saving intermediate results

**Key Methods:**
- `__init__(linkers=[], return_scores=True, output_dir=OUTPUT_PARTIAL_CHUNKS)`: Initialize with a list of linker instances
- `process_in_chunks(entities, output_dir=None)`: Process text in chunks with parallelization
- `process_chunk_el(chunk)`: Process a single chunk using all linkers
- `merge_results(results)`: Merge results from multiple entity linking methods

### BaseLinker

The `BaseLinker` abstract base class defines the interface for all entity linkers.

**Key Responsibilities:**
- Defining standard methods for entity linking
- Managing shared caching and initialization
- Processing chunks of entities

**Key Methods:**
- `initialize()`: Abstract method for initializing entity linking components
- `get_single_prediction(organization)`: Abstract method for predicting a single organization
- `process_chunk_el(chunk, return_scores=True)`: Process chunks using specific entity linkers
- `get_entity_groupings(entities, fallback=True)`: Generate groupings from NER output
- `get_el_input_organizations(grouped_entities, osm)`: Extract input organizations for linking

### BaseReranker

The `BaseReranker` abstract base class defines the interface for rerankers.

**Key Responsibilities:**
- Defining standard methods for reranking
- Refining entity linking results

**Key Methods:**
- `rerank(affiliation, candidates)`: Abstract method for reranking candidates

### DataManager

The `DataManager` class handles data-related operations.

**Key Responsibilities:**
- Managing data downloads and caching
- Creating and updating search indices
- Resolving file paths

**Key Methods:**
- `download_s2aff_data()`: Downloads S2AFF data
- `get_latest_ror()`: Fetches the latest ROR dump
- `create_whoosh_index()`: Creates a Whoosh index for entity linking

## Default Implementations

### WhooshLinker

The `WhooshLinker` class implements entity linking using the Whoosh search engine.

**Key Features:**
- Full-text search capabilities
- Support for fuzzy matching
- Integration with ROR data
- Optional LLM reranking

**Key Methods:**
- `initialize()`: Set up Whoosh index and components
- `get_single_prediction(organization)`: Get prediction for a single organization
- `get_candidate_matches(organization)`: Perform search using Whoosh

### S2AFFLinker

The `S2AFFLinker` class integrates with the S2AFF (Semantic Scholar Affiliation) system.

**Key Features:**
- Two-stage candidate retrieval
- Integration with the S2AFF ecosystem
- Efficient partial matching

**Key Methods:**
- `initialize()`: Set up S2AFF components
- `get_single_prediction(organization)`: Get prediction for a single organization

### LLMReranker

The `LLMReranker` class uses language models to improve entity matching accuracy.

**Key Features:**
- Leverages LLMs for contextual understanding
- Handles ambiguous cases
- Customizable model selection

**Key Methods:**
- `rerank(affiliation, candidates)`: Reranks candidates using LLM
- `_format_prompt(affiliation, candidates)`: Formats prompt for LLM input

## Extension Guide

### Creating a Custom Linker

To create a custom entity linker, follow these steps:

1. **Inherit from BaseLinker**:
```python
from affilgood.entity_linking.base_linker import BaseLinker

class CustomLinker(BaseLinker):
    def __init__(self, custom_param=None, debug=False):
        super().__init__()
        self.custom_param = custom_param
        self.debug = debug
        self.is_initialized = False
```

2. **Implement required abstract methods**:

```python
def initialize(self):
    """Initialize your linker components."""
    if self.is_initialized:
        return
        
    # Set up resources, load models, etc.
    self.setup_your_components()
    
    self.is_initialized = True

def get_single_prediction(self, organization):
    """Get predictions for one input organization."""
    # Check cache first
    affiliation_string = self._create_cache_key(organization)
    predicted_id, predicted_name, predicted_score = self._get_from_cache(affiliation_string)
    if predicted_id is not None:
        return predicted_id, predicted_name, predicted_score
    
    # Your linking logic here
    # ...
    
    # Update cache and return
    self._update_cache(affiliation_string, predicted_id, predicted_name, predicted_score)
    return predicted_id, predicted_name, predicted_score
```

3. **Add helper methods as needed**:

```python
def _create_cache_key(self, organization):
    """Create a cache key from organization details."""
    parts = []
    if organization.get('main', ''):
        parts.append(organization['main'])
    if organization.get('suborg', ''):
        parts.append(organization['suborg'])
    if organization.get('location', ''):
        parts.append(organization['location'])
    return ', '.join(parts)

def setup_your_components(self):
    """Set up the components needed for your linker."""
    # Your setup code here
```

### Creating a Custom Reranker

To create a custom reranker, follow these steps:

1. **Inherit from BaseReranker**:
```python
from affilgood.entity_linking.base_reranker import BaseReranker

class CustomReranker(BaseReranker):
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
```

2. **Implement the rerank method**:

```python
def rerank(self, affiliation, candidates):
    """
    Rerank the candidates based on your custom logic.
    
    Args:
        affiliation (str): Original affiliation string
        candidates (list): List of candidate organizations
        
    Returns:
        str: ROR ID of the best match, or None if no good match
    """
    # Your reranking logic here
    # ...
    
    return best_ror_id
```

### Integration with EntityLinker

Once you've created your custom linker or reranker, you can integrate it with the `EntityLinker`:

```python
from affilgood.entity_linking.entity_linker import EntityLinker
from your_module import CustomLinker, CustomReranker

# Create your custom components
custom_linker = CustomLinker(custom_param="value")
custom_reranker = CustomReranker()

# Set up your linker to use your reranker if applicable
custom_linker.reranker = custom_reranker

# Create the entity linker with your custom linker
entity_linker = EntityLinker(
    linkers=[custom_linker],
    return_scores=True
)

# Process entities with your custom linker
results = entity_linker.process_in_chunks(entities)
```

Or, to use multiple linkers including your custom one:

```python
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from your_module import CustomLinker

# Create an entity linker with multiple linkers
entity_linker = EntityLinker(
    linkers=[WhooshLinker(), CustomLinker()],
    return_scores=True
)

# Process entities using both linkers
results = entity_linker.process_in_chunks(entities)
```

## Example Extensions

### Example: Elastic-based Linker

Here's an example implementation of an Elasticsearch-based linker:

```python
from affilgood.entity_linking.base_linker import BaseLinker
from elasticsearch import Elasticsearch

class ElasticLinker(BaseLinker):
    def __init__(self, 
                 es_host="localhost", 
                 es_port=9200, 
                 index_name="ror_organizations",
                 threshold_score=0.3,
                 debug=False):
        super().__init__()
        self.es_host = es_host
        self.es_port = es_port
        self.index_name = index_name
        self.threshold_score = threshold_score
        self.debug = debug
        self.is_initialized = False
        
    def initialize(self):
        """Initialize Elasticsearch client and verify index exists."""
        if self.is_initialized:
            return
            
        # Initialize Elasticsearch client
        self.es = Elasticsearch([{'host': self.es_host, 'port': self.es_port}])
        
        # Check if index exists
        if not self.es.indices.exists(index=self.index_name):
            raise ValueError(f"Elasticsearch index {self.index_name} does not exist")
            
        self.is_initialized = True
        
    def get_single_prediction(self, organization):
        """Get predictions for one organization using Elasticsearch."""
        # Initialize if not already done
        if not self.is_initialized:
            self.initialize()
            
        # Build cache key and check cache
        affiliation = []
        suborg_ner = organization.get('suborg', '')
        org_ner = organization['main']
        location_ner = organization['location']
        if suborg_ner:
            affiliation.append(suborg_ner)
        affiliation.append(org_ner)
        if location_ner:
            affiliation.append(location_ner)
        affiliation_string_ner = ', '.join(affiliation)
        
        predicted_id, predicted_name, predicted_score = self._get_from_cache(affiliation_string_ner)
        if predicted_id is not None:
            return predicted_id, predicted_name, predicted_score
            
        # Build Elasticsearch query
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"name": {"query": org_ner, "boost": 3.0}}},
                        {"match": {"aliases": {"query": org_ner, "boost": 2.0}}}
                    ]
                }
            }
        }
        
        # Add location to query if available
        if location_ner:
            query["query"]["bool"]["should"].append(
                {"match": {"location": {"query": location_ner, "boost": 1.5}}}
            )
            
        # Add suborg to query if available
        if suborg_ner:
            query["query"]["bool"]["should"].append(
                {"match": {"departments": {"query": suborg_ner, "boost": 1.5}}}
            )
        
        # Execute search
        response = self.es.search(index=self.index_name, body=query, size=10)
        
        # Process results
        hits = response.get('hits', {}).get('hits', [])
        if hits and hits[0]['_score'] >= self.threshold_score:
            top_hit = hits[0]['_source']
            predicted_id = top_hit.get('id', '')
            predicted_name = top_hit.get('name', '')
            predicted_score = hits[0]['_score']
            
            # Update cache
            self._update_cache(affiliation_string_ner, predicted_id, predicted_name, predicted_score)
            return predicted_id, predicted_name, predicted_score
        
        # No good match found
        self._update_cache(affiliation_string_ner, None, None, 0)
        return None, None, 0
```

### Example: Rule-based Reranker

Here's an example implementation of a rule-based reranker:

```python
from affilgood.entity_linking.base_reranker import BaseReranker
import re

class RuleBasedReranker(BaseReranker):
    def __init__(self, 
                 location_boost=1.5, 
                 exact_name_boost=2.0,
                 min_name_tokens=3):
        self.location_boost = location_boost
        self.exact_name_boost = exact_name_boost
        self.min_name_tokens = min_name_tokens
        
    def rerank(self, affiliation, candidates):
        """
        Rerank candidates using rule-based methods.
        
        Args:
            affiliation (str): Original affiliation string
            candidates (list): List of candidate organizations
            
        Returns:
            str: ROR ID of the best match, or None if no good match
        """
        if not candidates:
            return None
            
        # Parse candidates to extract name, location, and ROR ID
        parsed_candidates = []
        for candidate in candidates:
            # Extract parts from format: "Name, Location (ROR:id)"
            match = re.match(r"(.*?)(?:, (.*?))? \((https://ror\.org/[0-9a-z]+)\)", candidate)
            if match:
                name = match.group(1).strip()
                location = match.group(2).strip() if match.group(2) else ""
                ror_id = match.group(3).replace("https://ror.org/", "")
                parsed_candidates.append({
                    "name": name,
                    "location": location,
                    "ror_id": ror_id,
                    "score": 1.0  # Base score
                })
                
        # Apply scoring rules
        affiliation_lower = affiliation.lower()
        
        for candidate in parsed_candidates:
            # Rule 1: Exact name match boost
            if candidate["name"].lower() in affiliation_lower:
                candidate["score"] *= self.exact_name_boost
                
            # Rule 2: Name token overlap
            name_tokens = set(candidate["name"].lower().split())
            affiliation_tokens = set(affiliation_lower.split())
            if len(name_tokens) >= self.min_name_tokens:
                overlap_ratio = len(name_tokens.intersection(affiliation_tokens)) / len(name_tokens)
                candidate["score"] *= (1.0 + overlap_ratio)
                
            # Rule 3: Location match boost
            if candidate["location"] and candidate["location"].lower() in affiliation_lower:
                candidate["score"] *= self.location_boost
        
        # Sort by score and return top result
        parsed_candidates.sort(key=lambda x: x["score"], reverse=True)
        return parsed_candidates[0]["ror_id"] if parsed_candidates else None
```

With these examples and the extension guide, you should be able to create your own custom entity linking components that integrate seamlessly with the AffilGood framework.
