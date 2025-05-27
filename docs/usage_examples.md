# AffilGood Usage Examples

This document provides detailed examples of how to use the AffilGood library for different scenarios.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Working with Different Entity Linkers](#working-with-different-entity-linkers)
3. [Working with Multiple Data Sources](#working-with-multiple-data-sources)
4. [Handling Multilingual Affiliations](#handling-multilingual-affiliations)
5. [Processing Large Datasets](#processing-large-datasets)
6. [Integration Examples](#integration-examples)
7. [Advanced Configurations](#advanced-configurations)

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
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.direct_pair_reranker import DirectPairReranker

# Create a reranker
reranker = DirectPairReranker()

# Configure Whoosh linker with reranker
whoosh_linker = WhooshLinker(rerank=False)  # Disable built-in reranking
whoosh_linker.reranker = reranker  # Set custom reranker

# Initialize AffilGood with the custom linker
affil_good = AffilGood(entity_linkers=whoosh_linker)

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
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.wikidata_dump_generator import WikidataDumpGenerator

# Initialize WikiData generator
wikidata_generator = WikidataDumpGenerator(verbose=True)

# Generate index for specific countries and organization types
df = wikidata_generator.get_index(
    countries=["Spain", "Germany"],
    org_types=["university", "hospital"]
)

# Configure Dense linker with WikiData data source
dense_linker = DenseLinker(data_source="wikidata")

# Initialize AffilGood with the custom linker
affil_good = AffilGood(entity_linkers=dense_linker)

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
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking import DataSourceRegistry

# Get Spanish hospitals handler
spanish_hospitals_handler = DataSourceRegistry.get_handler("spanish_hospitals")

# Configure with custom settings
spanish_hospitals_handler.initialize({
    'file_path': "path/to/spanish_hospitals.xlsx",
    'verbose': True
})

# Configure Dense linker with Spanish hospitals data source
dense_linker = DenseLinker(data_source="spanish_hospitals")

# Initialize AffilGood with the custom linker
affil_good = AffilGood(entity_linkers=dense_linker)

# Process affiliations
affiliations = ["Hospital Universitario La Paz, Madrid, Spain"]
results = affil_good.process(affiliations)
```

### Using Multiple Data Sources

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker

# Configure Dense linkers with different data sources
ror_linker = DenseLinker(data_source="ror")
wikidata_linker = DenseLinker(data_source="wikidata")
spanish_hospitals_linker = DenseLinker(data_source="spanish_hospitals")

# Initialize AffilGood with multiple linkers
affil_good = AffilGood(entity_linkers=[
    ror_linker,
    wikidata_linker,
    spanish_hospitals_linker
])

# Process affiliations
affiliations = [
    "Stanford University, Department of Computer Science, Stanford, CA, USA",
    "Hospital Universitario La Paz, Madrid, Spain",
    "Max Planck Institute for Intelligent Systems, Tübingen, Germany"
]
results = affil_good.process(affiliations)
```

## Handling Multilingual Affiliations

### Translating Affiliations

```python
from affilgood import AffilGood
from affilgood.entity_linking.llm_translator import LLMTranslator

# Initialize translator
translator = LLMTranslator(skip_english=True)

# Translate non-English affiliations
affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Università degli Studi di Milano, Milano, Italia",
    "Universität Heidelberg, Heidelberg, Deutschland"
]
translated = translator.translate_batch(affiliations)

# Process translated affiliations
affil_good = AffilGood()
results = affil_good.process(translated)
```

### Using Multilingual Models

```python
from affilgood import AffilGood

# Initialize AffilGood with multilingual models
affil_good = AffilGood(
    span_model_path="SIRIS-Lab/affilgood-span-multilingual",
    ner_model_path="SIRIS-Lab/affilgood-NER-multilingual"
)

# Process multilingual affiliations
affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Università degli Studi di Milano, Milano, Italia",
    "Universität Heidelberg, Heidelberg, Deutschland",
    "東京大学, 東京, 日本"
]
results = affil_good.process(affiliations)
```

### Language-Aware Pipeline

```python
from affilgood import AffilGood
from affilgood.entity_linking.language_detector import get_language_heur
from affilgood.entity_linking.llm_translator import LLMTranslator

def process_multilingual_affiliation(affiliation):
    # Detect language
    language = get_language_heur(affiliation)
    
    # Initialize translator for non-English text
    if language != 'en':
        translator = LLMTranslator(skip_english=True)
        affiliation = translator.translate(affiliation)
    
    # Process with AffilGood
    affil_good = AffilGood()
    results = affil_good.process([affiliation])
    
    return results

# Example usage
affiliations = [
    "Université Paris-Saclay, Paris, France",
    "Universität Wien, Wien, Österreich",
    "Università La Sapienza, Roma, Italia"
]

for affiliation in affiliations:
    results = process_multilingual_affiliation(affiliation)
    print(f"Original: {affiliation}")
    print(f"Result: {results[0]['ror']}")
    print("---")
```

## Processing Large Datasets

### Batch Processing with CSV Files

```python
import pandas as pd
from affilgood import AffilGood

def process_csv_file(input_file, output_file, batch_size=1000):
    """Process a CSV file with affiliations in batches."""
    # Initialize AffilGood
    affil_good = AffilGood()
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Process in batches
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        affiliations = batch['affiliation'].tolist()
        
        # Process batch
        batch_results = affil_good.process(affiliations)
        
        # Extract ROR IDs
        for j, result in enumerate(batch_results):
            row_idx = i + j
            ror_id = None
            if result['ror'] and result['ror'][0]:
                ror_parts = result['ror'][0].split('{https://ror.org/')
                if len(ror_parts) > 1:
                    ror_id = ror_parts[1].split('}')[0]
            
            results.append({
                'index': row_idx,
                'affiliation': result['raw_text'],
                'ror_id': ror_id
            })
        
        print(f"Processed {i + len(batch)}/{len(df)} affiliations")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Example usage
process_csv_file('affiliations.csv', 'affiliations_with_ror.csv')
```

### Parallel Processing

```python
import concurrent.futures
import pandas as pd
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

# Example usage
affiliations = [
    "Stanford University, CA, USA",
    "MIT, Cambridge, MA, USA",
    "Harvard University, Cambridge, MA, USA",
    # ... many more affiliations
]

results = process_large_dataset(affiliations, chunk_size=50, max_workers=2)
```

### Database Integration

```python
import sqlite3
from affilgood import AffilGood

def process_database_affiliations(db_path, batch_size=1000):
    """Process affiliations stored in a database."""
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create results table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS affiliation_results (
            id INTEGER PRIMARY KEY,
            affiliation TEXT,
            ror_id TEXT,
            organization TEXT,
            score REAL
        )
    ''')
    
    # Get total count of unprocessed affiliations
    cursor.execute('SELECT COUNT(*) FROM affiliations WHERE processed = 0')
    total_count = cursor.fetchone()[0]
    
    # Initialize AffilGood
    affil_good = AffilGood(return_scores=True)
    
    # Process in batches
    processed_count = 0
    
    while processed_count < total_count:
        # Get batch of unprocessed affiliations
        cursor.execute('''
            SELECT id, affiliation FROM affiliations 
            WHERE processed = 0 
            LIMIT ? OFFSET ?
        ''', (batch_size, processed_count))
        
        batch = cursor.fetchall()
        if not batch:
            break
        
        # Extract IDs and affiliations
        ids = [row[0] for row in batch]
        affiliations = [row[1] for row in batch]
        
        # Process with AffilGood
        results = affil_good.process(affiliations)
        
        # Insert results and update processed status
        for i, result in enumerate(results):
            affiliation_id = ids[i]
            
            # Extract organization and ROR ID
            organizations = []
            for ner in result['ner']:
                if 'ORG' in ner:
                    organizations.extend(ner['ORG'])
            
            organization = organizations[0] if organizations else None
            ror_id = None
            score = None
            
            if result['ror'] and result['ror'][0]:
                ror_parts = result['ror'][0].split('{https://ror.org/')
                if len(ror_parts) > 1:
                    ror_id = ror_parts[1].split('}')[0]
                    # Extract score if available
                    if '}:' in ror_parts[1]:
                        score_str = ror_parts[1].split('}:')[1]
                        try:
                            score = float(score_str)
                        except ValueError:
                            pass
            
            # Insert result
            cursor.execute('''
                INSERT INTO affiliation_results (id, affiliation, ror_id, organization, score)
                VALUES (?, ?, ?, ?, ?)
            ''', (affiliation_id, result['raw_text'], ror_id, organization, score))
            
            # Mark as processed
            cursor.execute('UPDATE affiliations SET processed = 1 WHERE id = ?', (affiliation_id,))
        
        # Commit changes
        conn.commit()
        
        # Update progress
        processed_count += len(batch)
        print(f"Processed {processed_count}/{total_count} affiliations")
    
    # Close connection
    conn.close()
    print("Processing complete!")

# Example usage
process_database_affiliations('affiliations.db')
```

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from affilgood import AffilGood
import logging

app = Flask(__name__)
affil_good = AffilGood()

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/process', methods=['POST'])
def process_affiliations():
    """Process affiliation strings."""
    try:
        data = request.json
        
        if not data or 'affiliations' not in data:
            return jsonify({'error': 'No affiliations provided'}), 400
        
        affiliations = data['affiliations']
        
        if not isinstance(affiliations, list):
            affiliations = [affiliations]
        
        # Process affiliations
        results = affil_good.process(affiliations)
        
        # Format results for API response
        formatted_results = []
        
        for result in results:
            # Extract organizations
            organizations = []
            for ner in result['ner']:
                if 'ORG' in ner:
                    organizations.extend(ner['ORG'])
            
            # Extract ROR IDs with scores
            ror_matches = []
            if result['ror']:
                for ror in result['ror']:
                    if '{https://ror.org/' in ror:
                        parts = ror.split('{https://ror.org/')
                        ror_id = parts[1].split('}')[0]
                        score = None
                        if '}:' in parts[1]:
                            try:
                                score = float(parts[1].split('}:')[1])
                            except ValueError:
                                pass
                        
                        ror_matches.append({
                            'id': ror_id,
                            'url': f'https://ror.org/{ror_id}',
                            'score': score
                        })
            
            formatted_results.append({
                'affiliation': result['raw_text'],
                'spans': result['span_entities'],
                'organizations': organizations,
                'ror_matches': ror_matches
            })
        
        return jsonify({
            'results': formatted_results,
            'count': len(formatted_results)
        })
    
    except Exception as e:
        logging.error(f"Error processing affiliations: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/process/batch', methods=['POST'])
def process_batch():
    """Process a large batch of affiliations."""
    try:
        data = request.json
        
        if not data or 'affiliations' not in data:
            return jsonify({'error': 'No affiliations provided'}), 400
        
        affiliations = data['affiliations']
        batch_size = data.get('batch_size', 100)
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(affiliations), batch_size):
            batch = affiliations[i:i+batch_size]
            batch_results = affil_good.process(batch)
            all_results.extend(batch_results)
        
        # Format results
        formatted_results = []
        for result in all_results:
            ror_id = None
            if result['ror'] and result['ror'][0]:
                ror_parts = result['ror'][0].split('{https://ror.org/')
                if len(ror_parts) > 1:
                    ror_id = ror_parts[1].split('}')[0]
            
            formatted_results.append({
                'affiliation': result['raw_text'],
                'ror_id': ror_id
            })
        
        return jsonify({
            'results': formatted_results,
            'count': len(formatted_results)
        })
    
    except Exception as e:
        logging.error(f"Error processing batch: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### Jupyter Notebook Integration

```python
# Cell 1: Setup
import pandas as pd
from affilgood import AffilGood
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialize AffilGood
affil_good = AffilGood(return_scores=True, verbose=True)

# Cell 2: Load and explore data
df = pd.read_csv('sample_affiliations.csv')
print(f"Loaded {len(df)} affiliations")
print(df.head())

# Cell 3: Process sample
sample_affiliations = df['affiliation'].head(10).tolist()
sample_results = affil_good.process(sample_affiliations)

# Display results
for i, result in enumerate(sample_results):
    print(f"\n{i+1}. {result['raw_text']}")
    if result['ror']:
        print(f"   ROR: {result['ror'][0]}")
    else:
        print("   No ROR match found")

# Cell 4: Process full dataset
def process_with_progress(affiliations, batch_size=100):
    results = []
    
    for i in tqdm(range(0, len(affiliations), batch_size), desc="Processing batches"):
        batch = affiliations[i:i+batch_size]
        batch_results = affil_good.process(batch)
        results.extend(batch_results)
    
    return results

all_results = process_with_progress(df['affiliation'].tolist())

# Cell 5: Analyze results
# Extract ROR IDs
ror_ids = []
for result in all_results:
    if result['ror'] and result['ror'][0]:
        ror_parts = result['ror'][0].split('{https://ror.org/')
        if len(ror_parts) > 1:
            ror_ids.append(ror_parts[1].split('}')[0])
        else:
            ror_ids.append(None)
    else:
        ror_ids.append(None)

# Create results dataframe
results_df = pd.DataFrame({
    'affiliation': [r['raw_text'] for r in all_results],
    'ror_id': ror_ids
})

# Calculate statistics
total_processed = len(results_df)
matched = len(results_df[results_df['ror_id'].notna()])
match_rate = matched / total_processed * 100

print(f"Total processed: {total_processed}")
print(f"Matched: {matched}")
print(f"Match rate: {match_rate:.1f}%")

# Cell 6: Visualize results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.pie([matched, total_processed - matched], 
        labels=['Matched', 'Not Matched'], 
        autopct='%1.1f%%')
plt.title('ROR Matching Results')

plt.subplot(1, 2, 2)
# Count organizations by country (if available)
countries = []
for result in all_results:
    for ner in result['ner']:
        if 'COUNTRY' in ner:
            countries.extend(ner['COUNTRY'])

if countries:
    country_counts = pd.Series(countries).value_counts().head(10)
    country_counts.plot(kind='bar')
    plt.title('Top 10 Countries')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Cell 7: Export results
results_df.to_csv('processed_affiliations.csv', index=False)
print("Results exported to processed_affiliations.csv")
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
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Speed-optimized configuration
whoosh_linker = WhooshLinker(
    rerank=False,  # Disable reranking for speed
    threshold_score=0.2,  # Lower threshold for more matches
    max_hits=5  # Limit search results
)

affil_good = AffilGood(
    span_separator=';',  # Use simple span splitting
    entity_linkers=whoosh_linker,
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

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.llm_translator import LLMTranslator

# Initialize translator for multilingual support
translator = LLMTranslator(skip_english=True, use_cache=True)

# Create linkers for different data sources
ror_dense_linker = DenseLinker(data_source="ror", threshold_score=0.35)
wikidata_dense_linker = DenseLinker(data_source="wikidata", threshold_score=0.35)
ror_whoosh_linker = WhooshLinker(data_source="ror", threshold_score=0.25)

# Initialize comprehensive AffilGood pipeline
affil_good = AffilGood(
    entity_linkers=[ror_dense_linker, wikidata_dense_linker, ror_whoosh_linker],
    return_scores=True,
    metadata_normalization=True,
    verbose=True
)

def comprehensive_process(affiliations):
    """Process affiliations with translation and multiple data sources."""
    # Translate non-English affiliations
    translated_affiliations = []
    
    for affiliation in affiliations:
        try:
            translated = translator.translate(affiliation)
            translated_affiliations.append(translated)
        except Exception as e:
            print(f"Translation error for '{affiliation}': {e}")
            translated_affiliations.append(affiliation)  # Use original if translation fails
    
    # Process with AffilGood
    results = affil_good.process(translated_affiliations)
    
    # Add original affiliations to results
    for i, result in enumerate(results):
        result['original_affiliation'] = affiliations[i]
        result['translated_affiliation'] = translated_affiliations[i]
    
    return results

# Example usage
multilingual_affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Università degli Studi di Milano, Milano, Italia",
    "Max-Planck-Institut für Intelligente Systeme, Tübingen, Deutschland",
    "Stanford University, Department of Computer Science, Stanford, CA, USA"
]

comprehensive_results = comprehensive_process(multilingual_affiliations)

# Display results
for result in comprehensive_results:
    print(f"Original: {result['original_affiliation']}")
    if result['original_affiliation'] != result['translated_affiliation']:
        print(f"Translated: {result['translated_affiliation']}")
    print(f"ROR Matches: {result['ror']}")
    print("---")
```

This completes the usage examples documentation with comprehensive examples covering basic usage, different entity linkers, multiple data sources, multilingual handling, large dataset processing, integration patterns, and advanced configurations.
