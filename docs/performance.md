# Performance Optimization in AffilGood

This document covers performance considerations and optimization strategies for the AffilGood pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Memory Usage](#memory-usage)
3. [Processing Speed](#processing-speed)
4. [Caching Strategies](#caching-strategies)
5. [Batch Processing](#batch-processing)
6. [Hardware Acceleration](#hardware-acceleration)
7. [Scaling for Large Datasets](#scaling-for-large-datasets)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Optimization Strategies](#optimization-strategies)

## Overview

AffilGood is designed to balance accuracy and performance. Each component of the pipeline has different performance characteristics and can be optimized for specific use cases.

The primary performance considerations for AffilGood are:

1. **Memory Usage**: How much memory each component requires
2. **Processing Speed**: How long each component takes to process data
3. **Scalability**: How well the pipeline handles large datasets

## Memory Usage

### Component Memory Requirements

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Span Identification (Model) | 500MB-1GB | Varies by model size |
| Span Identification (Simple) | <10MB | Character-based splitting uses minimal memory |
| NER | 500MB-1GB | Varies by model size |
| Whoosh Linker | 100-200MB | Depends on index size |
| S2AFF Linker | 500MB-1GB | Uses multiple models |
| Dense Linker | 1-2GB | Encoder model + vector index |
| Direct Pair Reranker | 500MB-1GB | Cross-encoder model |
| LLM Reranker | 4-10GB | Depends on LLM size |
| Metadata Normalization | <100MB | Mostly lookup tables |

### Reducing Memory Usage

1. **Use SimpleSpanIdentifier**: Replace model-based span identification with character-based splitting:

```python
from affilgood import AffilGood

# Use SimpleSpanIdentifier by specifying a separator
affil_good = AffilGood(span_separator=';')
```

2. **Disable Components**: Skip components you don't need:

```python
from affilgood import AffilGood

# Disable metadata normalization to save memory
affil_good = AffilGood(metadata_normalization=False)
```

3. **Choose Lightweight Entity Linkers**: Use Whoosh instead of Dense or LLM-based options:

```python
from affilgood import AffilGood

# Use Whoosh linker without reranking
affil_good = AffilGood(entity_linkers='Whoosh')
```

4. **Use CPU Instead of GPU**: For lower memory usage at the cost of speed:

```python
from affilgood import AffilGood

# Force CPU usage
affil_good = AffilGood(device='cpu')
```

## Processing Speed

### Component Processing Times

Approximate processing times per affiliation string on modern hardware:

| Component | Processing Time | Notes |
|-----------|----------------|-------|
| Span Identification (Model) | 10-50ms | Depends on text length |
| Span Identification (Simple) | <1ms | Very fast |
| NER | 10-50ms | Depends on text length |
| Whoosh Linker | 10-50ms | Fast but less accurate |
| S2AFF Linker | 50-200ms | Two-stage approach |
| Dense Linker | 50-200ms | Encoding + search |
| Direct Pair Reranker | 50-200ms | Multiple comparisons |
| LLM Reranker | 200-1000ms | Depends on model size |
| Metadata Normalization | 50-500ms | Depends on cache hits |

### Improving Processing Speed

1. **Use Simple Components**: Replace model-based components with rule-based alternatives:

```python
from affilgood import AffilGood

# Use SimpleSpanIdentifier for faster processing
affil_good = AffilGood(span_separator=';')
```

2. **Enable Caching**: Ensure caching is enabled for all components:

```python
from affilgood import AffilGood

# Enable caching for metadata normalization
affil_good = AffilGood(use_cache_metadata_normalization=True)
```

3. **Use Hardware Acceleration**: Enable GPU acceleration when available:

```python
from affilgood import AffilGood

# Use GPU acceleration
affil_good = AffilGood(device='cuda:0')
```

4. **Batch Processing**: Process data in batches for better efficiency:

```python
from affilgood.entity_linking.entity_linker import EntityLinker

# Process in batches
entity_linker = EntityLinker(linkers=['Whoosh'])
results = entity_linker.process_in_chunks(entities, output_dir='output')
```

## Caching Strategies

AffilGood implements several caching mechanisms to improve performance:

### Entity Linking Cache

The entity linkers cache results to avoid redundant processing:

```python
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Entity linker with caching
linker = WhooshLinker()

# Get cache hit rate
print(f"Cache hits: {len(linker.CACHED_PREDICTED_ID)}")
```

### Reranker Cache

The rerankers cache results to avoid redundant reranking:

```python
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker

# Reranker with caching
reranker = DirectPairReranker(use_cache=True)

# Get cache statistics
stats = reranker.get_cache_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")

# Clear cache if needed
reranker.clear_cache(expired_only=True)
```

### Metadata Normalization Cache

The GeoNormalizer caches results to avoid redundant API calls:

```python
from affilgood.metadata_normalization.normalizer import GeoNormalizer

# GeoNormalizer with caching
normalizer = GeoNormalizer(use_cache=True)
```

### WikiData Cache

The WikiDataCache caches SPARQL query results:

```python
from affilgood.entity_linking.wikidata_dump_generator import WikiDataCache

# Configure cache
cache = WikiDataCache(
    cache_dir="path/to/cache",
    cache_expiration_days=30
)

# Clear expired cache entries
cleared_count = cache.clear_expired()
print(f"Cleared {cleared_count} expired cache entries")
```

### Translation Cache

The LLMTranslator caches translation results:

```python
from affilgood.entity_linking.llm_translator import LLMTranslator

# Translator with caching
translator = LLMTranslator(use_cache=True)

# Get cache statistics
stats = translator.get_stats()
print(f"Cache hits: {stats['cache_hits']}")
```

## Batch Processing

For processing large datasets efficiently, AffilGood supports batch processing:

### Entity Linker Batching

```python
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Create entity linker
entity_linker = EntityLinker(
    linkers=[WhooshLinker()],
    return_scores=True
)

# Process in chunks with automatic parallelization
results = entity_linker.process_in_chunks(
    entities,
    output_dir='output'  # Save intermediate results
)
```

### Custom Batch Processing

```python
import concurrent.futures
from affilgood import AffilGood

def process_chunk(chunk):
    """Process a chunk of affiliations."""
    affil_good = AffilGood()
    return affil_good.process(chunk)

def process_large_dataset(affiliations, chunk_size=100, max_workers=4):
    """Process a large dataset in parallel chunks."""
    # Split into chunks
    chunks = [affiliations[i:i+chunk_size] for i in range(0, len(affiliations), chunk_size)]
    
    # Process chunks in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        for future in concurrent.futures.as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)
    
    return results
```

## Hardware Acceleration

AffilGood supports hardware acceleration for model-based components:

### GPU Acceleration

```python
from affilgood import AffilGood

# Use specific GPU
affil_good = AffilGood(device='cuda:0')

# Or let AffilGood choose automatically
affil_good = AffilGood(device=None)  # Auto-detect
```

### CPU-Only Mode

```python
from affilgood import AffilGood

# Force CPU usage
affil_good = AffilGood(device='cpu')
```

### Multi-GPU Support

```python
import os
import torch
from affilgood import AffilGood

# Set environment variable for PyTorch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")

# Use specific GPU
device = f'cuda:{num_gpus - 1}'  # Use last GPU
affil_good = AffilGood(device=device)
```

## Scaling for Large Datasets

For processing very large datasets (millions of affiliations), consider these approaches:

### Distributed Processing

```python
import concurrent.futures
import pandas as pd
from affilgood import AffilGood

def process_file(file_path):
    """Process a single file of affiliations."""
    # Load data
    df = pd.read_csv(file_path)
    affiliations = df['affiliation'].tolist()
    
    # Process with AffilGood
    affil_good = AffilGood()
    results = affil_good.process(affiliations)
    
    # Save results
    output_path = file_path.replace('.csv', '_processed.csv')
    output_df = pd.DataFrame({
        'affiliation': affiliations,
        'ror_id': [r['ror'][0] if r['ror'] else None for r in results]
    })
    output_df.to_csv(output_path, index=False)
    
    return output_path

def process_dataset(file_paths, max_workers=4):
    """Process multiple files in parallel."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path) for file_path in file_paths]
        
        for future in concurrent.futures.as_completed(futures):
            output_path = future.result()
            print(f"Processed: {output_path}")
```

### Database Integration

```python
import sqlite3
from affilgood import AffilGood

def process_database(db_path, batch_size=1000):
    """Process affiliations stored in a database."""
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute('SELECT COUNT(*) FROM affiliations WHERE ror_id IS NULL')
    total_count = cursor.fetchone()[0]
    
    # Process in batches
    processed_count = 0
    affil_good = AffilGood()
    
    while processed_count < total_count:
        # Get batch
        cursor.execute(
            'SELECT id, affiliation FROM affiliations WHERE ror_id IS NULL LIMIT ? OFFSET ?',
            (batch_size, processed_count)
        )
        batch = cursor.fetchall()
        
        if not batch:
            break
        
        # Extract IDs and affiliations
        ids = [row[0] for row in batch]
        affiliations = [row[1] for row in batch]
        
        # Process with AffilGood
        results = affil_good.process(affiliations)
        
        # Update database
        for i, result in enumerate(results):
            ror_id = None
            if result['ror'] and result['ror'][0]:
                ror_parts = result['ror'][0].split('{https://ror.org/')
                if len(ror_parts) > 1:
                    ror_id = ror_parts[1].split('}')[0]
            
            cursor.execute(
                'UPDATE affiliations SET ror_id = ? WHERE id = ?',
                (ror_id, ids[i])
            )
        
        # Commit changes
        conn.commit()
        
        # Update progress
        processed_count += len(batch)
        print(f"Processed {processed_count}/{total_count} affiliations")
    
    # Close connection
    conn.close()
```

## Performance Benchmarks

Typical performance metrics for the AffilGood pipeline:

### Small-Scale Processing

Processing 100 affiliations:

| Configuration | Memory Usage | Processing Time |
|---------------|--------------|----------------|
| Default (Whoosh) | ~1GB | ~5-10 seconds |
| SimpleSpanIdentifier + Whoosh | ~500MB | ~3-5 seconds |
| Full Pipeline (Dense + LLM) | ~5-10GB | ~30-60 seconds |

### Medium-Scale Processing

Processing 10,000 affiliations:

| Configuration | Memory Usage | Processing Time |
|---------------|--------------|----------------|
| Default (Whoosh) | ~1GB | ~5-10 minutes |
| SimpleSpanIdentifier + Whoosh | ~500MB | ~3-5 minutes |
| Full Pipeline (Dense + LLM) | ~5-10GB | ~30-60 minutes |

### Large-Scale Processing

Processing 1,000,000 affiliations:

| Configuration | Memory Usage | Processing Time |
|---------------|--------------|----------------|
| Default (Whoosh) | ~1-2GB | ~8-16 hours |
| SimpleSpanIdentifier + Whoosh | ~500MB-1GB | ~5-10 hours |
| Full Pipeline (Dense + LLM) | ~10-20GB | ~50-100 hours |

## Optimization Strategies

### Strategy 1: Speed-Focused Configuration

For maximum speed with acceptable accuracy:

```python
from affilgood import AffilGood
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Configure Whoosh linker
whoosh_linker = WhooshLinker(
    rerank=False,  # Disable reranking
    threshold_score=0.2  # Lower threshold for more matches
)

# Configure AffilGood
affil_good = AffilGood(
    span_separator=';',  # Use SimpleSpanIdentifier
    entity_linkers=whoosh_linker,
    metadata_normalization=False,  # Disable metadata normalization
    verbose=False
)
```

### Strategy 2: Memory-Efficient Configuration

For minimal memory usage:

```python
from affilgood import AffilGood

# Configure AffilGood
affil_good = AffilGood(
    span_separator=';',  # Use SimpleSpanIdentifier
    entity_linkers='Whoosh',  # Use Whoosh linker
    metadata_normalization=False,  # Disable metadata normalization
    verbose=False,
    device='cpu'  # Force CPU usage
)
```

### Strategy 3: Accuracy-Focused Configuration

For maximum accuracy:

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.llm_reranker import LLMReranker

# Configure reranker
reranker = LLMReranker()

# Configure linkers
dense_linker = DenseLinker(use_hnsw=True)
whoosh_linker = WhooshLinker(rerank=False)  # We'll use our custom reranker

# Apply reranker to linkers
dense_linker.reranker = reranker
whoosh_linker.reranker = reranker

# Configure AffilGood
affil_good = AffilGood(
    entity_linkers=[dense_linker, whoosh_linker],
    return_scores=True,
    metadata_normalization=True,
    use_cache_metadata_normalization=True,
    verbose=True,
    device='cuda:0'  # Use GPU acceleration
)
```

### Strategy 4: Balanced Configuration

For a good balance of speed and accuracy:

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker

# Configure reranker
reranker = DirectPairReranker()

# Configure linker
dense_linker = DenseLinker(use_hnsw=True)
dense_linker.reranker = reranker

# Configure AffilGood
affil_good = AffilGood(
    entity_linkers=dense_linker,
    return_scores=True,
    metadata_normalization=True,
    use_cache_metadata_normalization=True,
    verbose=False,
    device=None  # Auto-detect
)
```

### Strategy 5: Incremental Processing

For processing large datasets incrementally:

```python
import pandas as pd
from affilgood import AffilGood

def process_incrementally(file_path, batch_size=1000, output_path=None):
    """Process a large CSV file incrementally."""
    # Configure AffilGood
    affil_good = AffilGood()
    
    # Initialize output
    if output_path is None:
        output_path = file_path.replace('.csv', '_processed.csv')
    
    # Create empty output file
    with open(output_path, 'w') as f:
        f.write('affiliation,ror_id\n')
    
    # Process in batches
    chunk_iter = pd.read_csv(file_path, chunksize=batch_size)
    
    for i, chunk in enumerate(chunk_iter):
        # Extract affiliations
        affiliations = chunk['affiliation'].tolist()
        
        # Process with AffilGood
        results = affil_good.process(affiliations)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'affiliation': affiliations,
            'ror_id': [
                r['ror'][0].split('{https://ror.org/')[1].split('}')[0] 
                if r['ror'] and r['ror'][0] and '{https://ror.org/' in r['ror'][0] 
                else None 
                for r in results
            ]
        })
        
        # Append to output file
        output_df.to_csv(output_path, mode='a', header=False, index=False)
        
        print(f"Processed batch {i+1}: {len(affiliations)} affiliations")
    
    return output_path
```
