# Technical Overview

This document provides an in-depth technical overview of the AffilGood architecture, component interactions, data flow, and extension points.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Processing Pipeline](#processing-pipeline)
5. [Extension Points](#extension-points)
6. [Data Structures](#data-structures)
7. [Caching and Performance](#caching-and-performance)
8. [Error Handling](#error-handling)
9. [Threading and Concurrency](#threading-and-concurrency)
10. [Dependencies and Requirements](#dependencies-and-requirements)

## Architecture Overview

AffilGood follows a modular pipeline architecture designed for flexibility, scalability, and extensibility. The system is built around the concept of processing stages, where each stage can be customized or replaced independently.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Raw Text Input │───▶│ Span             │───▶│ Named Entity    │───▶│ Metadata         │
│                 │    │ Identification   │    │ Recognition     │    │ Normalization    │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                                         │                        │
                                                         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Final Results   │◀───│ Entity           │◀───│ Candidate       │◀───│ Location         │
│                 │    │ Linking          │    │ Retrieval       │    │ Resolution       │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

### Key Architectural Principles

1. **Modularity**: Each component is self-contained and can be replaced independently
2. **Extensibility**: New components can be added through well-defined interfaces
3. **Configurability**: All components can be configured through parameters
4. **Caching**: Extensive caching at multiple levels for performance
5. **Error Resilience**: Graceful handling of errors with fallback mechanisms

## Component Architecture

### Core Components

#### 1. AffilGood (Main Orchestrator)

The `AffilGood` class serves as the main orchestrator that coordinates all pipeline components:

```python
class AffilGood:
    def __init__(self, ...):
        self.span_identifier = ...    # Span identification component
        self.ner = ...               # Named entity recognition component  
        self.entity_linker = ...     # Entity linking component
        self.normalizer = ...        # Metadata normalization component
    
    def process(self, text):
        """Main processing pipeline"""
        spans = self.get_span(text)
        entities = self.get_ner(spans)
        normalized = self.get_normalization(entities)
        linked = self.get_entity_linking(normalized)
        return linked
```

#### 2. Span Identification Layer

Three different span identification strategies:

- **SpanIdentifier**: Model-based approach using transformer models
- **SimpleSpanIdentifier**: Rule-based approach using character delimiters
- **NoopSpanIdentifier**: Pass-through for pre-segmented text

```python
class BaseSpanIdentifier(ABC):
    @abstractmethod
    def identify_spans(self, text_list):
        """Convert raw text into meaningful spans"""
        pass
```

#### 3. Named Entity Recognition Layer

Transformer-based NER with support for multiple entity types:

```python
class NER:
    def __init__(self, model_path, device, ...):
        self.model = pipeline("ner", model=model_path, ...)
    
    def recognize_entities(self, spans):
        """Extract organizations, sub-organizations, and locations"""
        # Entity types: ORG, SUB, CITY, REGION, COUNTRY
        pass
```

#### 4. Entity Linking Layer

Multi-strategy entity linking with support for multiple retrievers and rerankers:

```python
class EntityLinker:
    def __init__(self, linkers, ...):
        self.linkers = self._initialize_linkers(linkers)
    
    def process_in_chunks(self, entities):
        """Link entities using configured strategies"""
        pass
```

### Linker Architecture

Each entity linker follows the `BaseLinker` interface:

```python
class BaseLinker(ABC):
    def __init__(self):
        self.CACHED_PREDICTED_ID = {}      # ID cache
        self.CACHED_PREDICTED_NAME = {}    # Name cache
        self.CACHED_PREDICTED_ID_SCORE = {} # Score cache
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self):
        """Initialize linker components"""
        pass
    
    @abstractmethod
    def get_single_prediction(self, organization):
        """Get prediction for one organization"""
        pass
```

#### Available Linkers

1. **WhooshLinker**: Full-text search using Whoosh
2. **S2AFFLinker**: Integration with Semantic Scholar system
3. **DenseLinker**: Dense vector similarity matching

### Data Source Architecture

Pluggable data source system using the registry pattern:

```python
class DataSourceHandler(ABC):
    @property
    @abstractmethod
    def source_id(self):
        """Unique identifier for this data source"""
        pass
    
    @abstractmethod
    def load_data(self, config):
        """Load data from the source"""
        pass
    
    @abstractmethod
    def map_organization(self, org):
        """Map to standard fields"""
        pass

@DataSourceRegistry.register
class RORHandler(DataSourceHandler):
    # Implementation...
```

## Data Flow

### Input Processing Flow

1. **Text Input**: Raw affiliation strings
2. **Span Identification**: Break text into meaningful segments
3. **Entity Recognition**: Identify organizations and locations within spans
4. **Metadata Normalization**: Standardize location information
5. **Entity Linking**: Match organizations to standard identifiers
6. **Result Compilation**: Combine all processing results

### Data Transformation Pipeline

```python
# Stage 1: Raw Input
raw_text = "Department of Computer Science, Stanford University, CA, USA"

# Stage 2: Span Identification
spans = {
    "raw_text": raw_text,
    "span_entities": ["Department of Computer Science, Stanford University, CA, USA"]
}

# Stage 3: Named Entity Recognition
entities = {
    "raw_text": raw_text,
    "span_entities": [...],
    "ner": [{
        "SUB": ["Department of Computer Science"],
        "ORG": ["Stanford University"], 
        "REGION": ["CA"],
        "COUNTRY": ["USA"]
    }],
    "ner_raw": [...]  # Raw NER output
}

# Stage 4: Metadata Normalization
normalized = {
    # ... previous fields ...
    "osm": [{
        "CITY": "Stanford",
        "REGION": "California", 
        "COUNTRY": "United States",
        "COORDS": [37.4419, -122.1430],
        "OSM_ID": "123456"
    }]
}

# Stage 5: Entity Linking
linked = {
    # ... previous fields ...
    "ror": ["Stanford University {https://ror.org/00f54p054}:0.95"]
}
```

## Processing Pipeline

### Initialization Sequence

1. **Component Initialization**: Initialize all pipeline components
2. **Model Loading**: Load transformer models and embeddings
3. **Index Preparation**: Prepare search indices for entity linking
4. **Cache Setup**: Initialize caching mechanisms

### Runtime Processing

#### Single Text Processing

```python
def process_single(self, text):
    # Stage 1: Span identification
    spans = self.span_identifier.identify_spans([text])
    
    # Stage 2: Named entity recognition  
    entities = self.ner.recognize_entities(spans)
    
    # Stage 3: Metadata normalization
    if self.normalizer:
        normalized = self.normalizer.normalize(entities)
    else:
        normalized = entities
    
    # Stage 4: Entity linking
    linked = self.entity_linker.process_in_chunks(normalized)
    
    return linked[0]
```

#### Batch Processing

```python
def process_batch(self, texts, batch_size=32):
    # Process in batches for efficiency
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Each stage processes the entire batch
        spans = self.span_identifier.identify_spans(batch)
        entities = self.ner.recognize_entities(spans)
        normalized = self.normalizer.normalize(entities) if self.normalizer else entities
        linked = self.entity_linker.process_in_chunks(normalized)
        
        results.extend(linked)
    
    return results
```

### Parallel Processing Architecture

AffilGood supports parallel processing at multiple levels:

#### Component-Level Parallelism

```python
# Entity linking with multiple linkers in parallel
with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EL) as executor:
    futures = [
        executor.submit(self.process_chunk_el, chunk)
        for chunk in chunks
    ]
    
    results = [future.result() for future in futures]
```

#### Batch-Level Parallelism

```python
# Processing multiple batches in parallel
def process_large_dataset(self, texts, chunk_size=1000):
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EL) as executor:
        futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
        results = [future.result() for future in futures]
    
    return [item for sublist in results for item in sublist]  # Flatten
```

## Extension Points

### 1. Custom Span Identification

Implement the span identification interface:

```python
class CustomSpanIdentifier:
    def identify_spans(self, text_list):
        # Your custom logic here
        return results
```

### 2. Custom Entity Linking

Extend the `BaseLinker` class:

```python
class CustomLinker(BaseLinker):
    def initialize(self):
        # Initialize your linker
        self.is_initialized = True
    
    def get_single_prediction(self, organization):
        # Your linking logic here
        return predicted_id, predicted_name, predicted_score
```

### 3. Custom Data Sources

Implement the `DataSourceHandler` interface:

```python
@DataSourceRegistry.register
class CustomDataSource(DataSourceHandler):
    @property
    def source_id(self):
        return "custom_source"
    
    def load_data(self, config):
        # Load your data
        pass
    
    def map_organization(self, org):
        # Map to standard format
        pass
```

### 4. Custom Rerankers

Extend the `BaseReranker` class:

```python
class CustomReranker(BaseReranker):
    def rerank(self, affiliation, candidates):
        # Your reranking logic
        return best_match
```

## Data Structures

### Core Data Types

#### Span Data Structure

```python
span_data = {
    "raw_text": str,           # Original input text
    "span_entities": List[str] # Identified spans
}
```

#### NER Data Structure

```python
ner_data = {
    "raw_text": str,
    "span_entities": List[str],
    "ner": List[Dict[str, List[str]]],  # Grouped entities by type
    "ner_raw": List[List[Dict]]         # Raw NER output
}
```

#### Normalized Data Structure

```python
normalized_data = {
    # ... previous fields ...
    "osm": List[Optional[Dict[str, Any]]]  # Location metadata
}
```

#### Final Result Structure

```python
result_data = {
    # ... previous fields ...
    "ror": List[str]  # Entity linking results
}
```

### Entity Linking Candidate Structure

```python
candidate = {
    "id": str,              # Organization identifier
    "name": str,            # Organization name
    "city": str,            # City name
    "country": str,         # Country name
    "parent": str,          # Parent organization
    "enc_score": float,     # Encoder similarity score
    "orig_score": float,    # Original retrieval score
    "source": str,          # Retrieval method used
    "data_source": str,     # Data source (ror, wikidata, etc.)
    "explanation": str      # Matching explanation
}
```

## Caching and Performance

### Multi-Level Caching Architecture

#### 1. Component-Level Caching

Each component maintains its own cache:

```python
class BaseLinker:
    def __init__(self):
        self.CACHED_PREDICTED_ID = {}      # Results cache
        self.CACHED_PREDICTED_NAME = {}    # Name cache
        self.CACHED_PREDICTED_ID_SCORE = {} # Score cache
```

#### 2. Reranker Caching

```python
class DirectPairReranker:
    def __init__(self, use_cache=True, cache_dir=None):
        self.use_cache = use_cache
        self.reranking_cache = {}  # In-memory cache
        self.cache_file = cache_file_path  # Persistent cache
```

#### 3. Translation Caching

```python
class LLMTranslator:
    def __init__(self, use_cache=True):
        self.session = requests_cache.CachedSession(
            cache_name='translation_cache',
            expire_after=604800  # 7 days
        )
```

#### 4. Metadata Normalization Caching

```python
class GeoNormalizer:
    def __init__(self, use_cache=True, cache_fname=CACHE_FILE_PATH):
        self.cache = self.load_cache(cache_fname) if use_cache else {}
```

### Performance Optimization Strategies

#### Memory Management

```python
# Lazy loading of models
def initialize(self):
    if not self.is_initialized:
        self.model = load_model()  # Only load when needed
        self.is_initialized = True
```

#### Batch Processing Optimization

```python
# Process embeddings in batches
def encode_batch(self, texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = self.encoder.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings
```

## Error Handling

### Exception Hierarchy

```python
class AffilGoodError(Exception):
    """Base exception for AffilGood"""
    pass

class ModelLoadError(AffilGoodError):
    """Error loading models"""
    pass

class ProcessingError(AffilGoodError):
    """Error during processing"""
    pass

class DataSourceError(AffilGoodError):
    """Error with data sources"""
    pass
```

### Error Recovery Strategies

#### Graceful Degradation

```python
def process_with_fallback(self, text):
    try:
        # Try primary processing
        return self.primary_process(text)
    except Exception as e:
        logger.warning(f"Primary processing failed: {e}")
        # Fall back to simpler processing
        return self.fallback_process(text)
```

#### Component Isolation

```python
def safe_component_call(self, component, *args, **kwargs):
    try:
        return component(*args, **kwargs)
    except Exception as e:
        logger.error(f"Component {component.__class__.__name__} failed: {e}")
        return self.get_default_result()
```

## Threading and Concurrency

### Thread Safety

Most components are designed to be thread-safe for read operations:

```python
class ThreadSafeLinker(BaseLinker):
    def __init__(self):
        super().__init__()
        self._lock = threading.RLock()
    
    def get_single_prediction(self, organization):
        with self._lock:
            # Thread-safe prediction
            return super().get_single_prediction(organization)
```

### Parallel Processing Patterns

#### Producer-Consumer Pattern

```python
import queue
import threading

def parallel_process(self, texts, num_workers=4):
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # Add texts to input queue
    for text in texts:
        input_queue.put(text)
    
    # Worker function
    def worker():
        while True:
            try:
                text = input_queue.get(timeout=1)
                result = self.process_single(text)
                output_queue.put(result)
                input_queue.task_done()
            except queue.Empty:
                break
    
    # Start workers
    threads = [threading.Thread(target=worker) for _ in range(num_workers)]
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Collect results
    results = []
    while not output_queue.empty():
        results.append(output_queue.get())
    
    return results
```

## Dependencies and Requirements

### Core Dependencies

#### Deep Learning Framework
- **PyTorch**: Neural network computations
- **Transformers**: Pre-trained language models
- **Datasets**: Data handling for model training

#### Search and Indexing
- **Whoosh**: Full-text search engine
- **hnswlib**: Approximate nearest neighbor search
- **sentence-transformers**: Semantic similarity

#### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **requests**: HTTP requests

#### Geolocation
- **geopy**: Geocoding services
- **country_converter**: Country code conversion

### Optional Dependencies

#### Language Processing
- **langdetect**: Language detection
- **translators**: Translation services
- **text_unidecode**: Text normalization

#### Machine Learning
- **lightgbm**: Gradient boosting (for S2AFF)
- **scikit-learn**: Machine learning utilities
- **hyperopt**: Hyperparameter optimization

#### Specialized Features
- **SPARQLWrapper**: WikiData integration
- **pycountry**: Country information
- **boto3**: AWS services (for data downloads)

### System Requirements

#### Minimum Requirements
- **Python**: 3.9+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB for models and indices
- **CPU**: 2+ cores

#### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16GB+
- **Storage**: 10GB+ for full indices
- **GPU**: CUDA-compatible for acceleration
- **CPU**: 8+ cores for parallel processing

### Environment Configuration

```python
# Environment variables for configuration
import os

# Model paths
SPAN_MODEL_PATH = os.getenv('AFFILGOOD_SPAN_MODEL', 'SIRIS-Lab/affilgood-span-multilingual')
NER_MODEL_PATH = os.getenv('AFFILGOOD_NER_MODEL', 'SIRIS-Lab/affilgood-NER-multilingual')

# Cache directories
CACHE_DIR = os.getenv('AFFILGOOD_CACHE_DIR', '~/.affilgood_cache')
INDEX_DIR = os.getenv('AFFILGOOD_INDEX_DIR', '~/.affilgood_indices')

# Performance settings
MAX_WORKERS = int(os.getenv('AFFILGOOD_MAX_WORKERS', '4'))
BATCH_SIZE = int(os.getenv('AFFILGOOD_BATCH_SIZE', '32'))

# Device configuration
DEVICE = os.getenv('AFFILGOOD_DEVICE', 'auto')  # 'auto', 'cpu', 'cuda:0', etc.
```

This technical overview provides a comprehensive understanding of AffilGood's architecture, enabling developers to effectively use, extend, and contribute to the system.