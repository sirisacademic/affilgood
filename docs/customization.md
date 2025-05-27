# Customizing AffilGood

AffilGood is designed to be highly customizable, allowing you to extend and adapt the pipeline to your specific needs. This document provides guidance on customizing different components of the AffilGood pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Custom Span Identification](#custom-span-identification)
3. [Custom Entity Linking](#custom-entity-linking)
4. [Custom Data Sources](#custom-data-sources)
5. [Custom Rerankers](#custom-rerankers)
6. [Custom Language Processing](#custom-language-processing)
7. [Integration with External Systems](#integration-with-external-systems)
8. [Best Practices](#best-practices)

## Overview

AffilGood follows a modular architecture with well-defined extension points:

1. **Span Identification**: Converting raw text into meaningful spans
2. **Named Entity Recognition (NER)**: Identifying organizations and locations
3. **Entity Linking**: Matching identified entities with standard identifiers
4. **Metadata Normalization**: Standardizing location information

Each of these components can be customized or replaced with your own implementation.

## Custom Span Identification

### Creating a Custom Span Identifier

To create a custom span identifier, implement a class with an `identify_spans` method:

```python
class CustomSpanIdentifier:
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
        
    def identify_spans(self, text_list):
        """
        Identify spans in a list of texts.
        
        Args:
            text_list (list of str): Input texts to process
            
        Returns:
            list of dict: Each dict contains 'raw_text' and 'span_entities' fields
        """
        results = []
        
        for text in text_list:
            # Your custom span identification logic here
            # For example, splitting based on a specific pattern
            spans = self._my_custom_split_function(text)
            
            results.append({
                "raw_text": text,
                "span_entities": spans
            })
            
        return results
    
    def _my_custom_split_function(self, text):
        # Custom logic to split text into spans
        # ...
        return spans
```

### Using a Custom Span Identifier with AffilGood

```python
from affilgood import AffilGood

# Initialize your custom span identifier
custom_span_identifier = CustomSpanIdentifier(custom_param="value")

# Initialize AffilGood with your custom span identifier
affil_good = AffilGood()
affil_good.span_identifier = custom_span_identifier

# Process text with your custom span identifier
results = affil_good.process(["Your text here"])
```

### Example: Pattern-Based Span Identifier

```python
import re

class PatternBasedSpanIdentifier:
    def __init__(self, patterns=None):
        self.patterns = patterns or [
            r'([^.;]+(?:University|College|Institute)[^.;]+[.;])',
            r'([^.;]+Department[^.;]+[.;])',
            r'([^.;]+Laboratory[^.;]+[.;])'
        ]
        
    def identify_spans(self, text_list):
        results = []
        
        for text in text_list:
            spans = []
            
            # Apply each pattern
            for pattern in self.patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                spans.extend(matches)
            
            # If no spans found, use the entire text
            if not spans:
                spans = [text]
                
            results.append({
                "raw_text": text,
                "span_entities": spans
            })
            
        return results
```

## Custom Entity Linking

### Creating a Custom Entity Linker

To create a custom entity linker, extend the `BaseLinker` class:

```python
from affilgood.entity_linking.base_linker import BaseLinker

class CustomLinker(BaseLinker):
    def __init__(self, custom_param=None, debug=False):
        super().__init__()
        self.custom_param = custom_param
        self.debug = debug
        self.is_initialized = False
    
    def initialize(self):
        """Initialize your linker components."""
        if self.is_initialized:
            return
        
        # Your initialization code here
        # ...
        
        self.is_initialized = True
    
    def get_single_prediction(self, organization):
        """
        Get predictions for one organization.
        
        Args:
            organization (dict): Organization to match
            
        Returns:
            tuple: (predicted_id, predicted_name, predicted_score)
        """
        # Ensure initialization
        if not self.is_initialized:
            self.initialize()
        
        # Create a cache key for this organization
        affiliation_string = self._create_cache_key(organization)
        
        # Check if we have a cached result
        predicted_id, predicted_name, predicted_score = self._get_from_cache(affiliation_string)
        if predicted_id is not None:
            return predicted_id, predicted_name, predicted_score
        
        # Your custom linking logic here
        # ...
        
        # Update cache and return
        self._update_cache(affiliation_string, predicted_id, predicted_name, predicted_score)
        return predicted_id, predicted_name, predicted_score
    
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
```

### Using a Custom Entity Linker with AffilGood

```python
from affilgood import AffilGood

# Initialize your custom entity linker
custom_linker = CustomLinker(custom_param="value")

# Initialize AffilGood with your custom entity linker
affil_good = AffilGood(entity_linkers=custom_linker)

# Process text with your custom entity linker
results = affil_good.process(["Your text here"])
```

### Example: Simple Dictionary-Based Linker

```python
from affilgood.entity_linking.base_linker import BaseLinker

class DictionaryLinker(BaseLinker):
    def __init__(self, organization_dict=None, threshold_score=0.6, debug=False):
        super().__init__()
        self.organization_dict = organization_dict or {}
        self.threshold_score = threshold_score
        self.debug = debug
        self.is_initialized = True
    
    def initialize(self):
        """Nothing to initialize for this simple linker."""
        self.is_initialized = True
    
    def get_single_prediction(self, organization):
        """Match organizations using a simple dictionary lookup."""
        # Create a cache key
        affiliation_string = self._create_cache_key(organization)
        
        # Check cache
        predicted_id, predicted_name, predicted_score = self._get_from_cache(affiliation_string)
        if predicted_id is not None:
            return predicted_id, predicted_name, predicted_score
        
        # Extract organization name
        org_name = organization.get('main', '')
        
        # Simple exact matching
        if org_name in self.organization_dict:
            entry = self.organization_dict[org_name]
            predicted_id = entry.get('id')
            predicted_name = entry.get('name', org_name)
            predicted_score = 1.0
        else:
            # Fuzzy matching
            best_match = None
            best_score = 0
            
            for name, entry in self.organization_dict.items():
                # Calculate similarity score (simple example)
                score = self._calculate_similarity(org_name, name)
                
                if score > best_score:
                    best_score = score
                    best_match = entry
            
            if best_score >= self.threshold_score and best_match:
                predicted_id = best_match.get('id')
                predicted_name = best_match.get('name')
                predicted_score = best_score
            else:
                predicted_id = None
                predicted_name = None
                predicted_score = 0
        
        # Update cache and return
        self._update_cache(affiliation_string, predicted_id, predicted_name, predicted_score)
        return predicted_id, predicted_name, predicted_score
    
    def _calculate_similarity(self, text1, text2):
        """Calculate simple similarity between two strings."""
        # This is a very basic example - replace with a better similarity metric
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
```

## Custom Data Sources

### Creating a Custom Data Source Handler

To create a custom data source, implement the `DataSourceHandler` interface:

```python
from affilgood.entity_linking import DataSourceHandler, DataSourceRegistry

@DataSourceRegistry.register
class CustomDataSourceHandler(DataSourceHandler):
    @property
    def source_id(self):
        """Return the unique identifier for this data source."""
        return "custom_source"
    
    def initialize(self, config):
        """Initialize with configuration."""
        self.config = config or {}
        # Your initialization code here
        # ...
    
    def load_data(self, config):
        """Load data from the source with the given configuration."""
        # Your data loading code here
        # ...
        return data
    
    def get_data_for_indexing(self, config, indices_type='whoosh', **kwargs):
        """Prepare and return data for index creation."""
        # Your index preparation code here
        # ...
        return data, index_id
    
    def map_organization(self, org):
        """Map organization fields to standard fields."""
        # Your field mapping code here
        # ...
        return mapped_org
    
    def format_id_url(self, org_id):
        """Format organization ID as URL."""
        # Your URL formatting code here
        # ...
        return url
```

### Using a Custom Data Source

```python
from affilgood.entity_linking import DataSourceRegistry
from affilgood.entity_linking.dense_linker import DenseLinker

# Register your custom data source (if not using the decorator)
DataSourceRegistry.register(CustomDataSourceHandler)

# Get your registered handler
custom_handler = DataSourceRegistry.get_handler("custom_source")

# Initialize with configuration
custom_handler.initialize({
    'custom_param': 'value'
})

# Use with a linker
linker = DenseLinker(data_source="custom_source")

# Or with AffilGood
from affilgood import AffilGood
affil_good = AffilGood(entity_linkers=linker)
```

## Custom Rerankers

### Creating a Custom Reranker

To create a custom reranker, extend the `BaseReranker` class:

```python
from affilgood.entity_linking.base_reranker import BaseReranker

class CustomReranker(BaseReranker):
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
    
    def rerank(self, affiliation, candidates):
        """
        Rerank candidates based on custom logic.
        
        Args:
            affiliation (str): Original affiliation string
            candidates (list): List of candidate organizations
            
        Returns:
            str: ROR ID of the best match, or None if no good match
        """
        # Your reranking logic here
        # ...
        
        return best_match
```

### Using a Custom Reranker

```python
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Initialize your custom reranker
custom_reranker = CustomReranker(custom_param="value")

# Use with WhooshLinker
whoosh_linker = WhooshLinker(rerank=False)  # Disable built-in reranking
whoosh_linker.reranker = custom_reranker    # Set custom reranker

# Or with AffilGood
from affilgood import AffilGood
affil_good = AffilGood(entity_linkers=whoosh_linker)
```

### Example: Weighted Field Reranker

```python
from affilgood.entity_linking.base_reranker import BaseReranker
import re

class WeightedFieldReranker(BaseReranker):
    def __init__(self, name_weight=3.0, location_weight=1.5, parent_weight=1.0):
        self.name_weight = name_weight
        self.location_weight = location_weight
        self.parent_weight = parent_weight
    
    def rerank(self, affiliation, candidates):
        """Rerank candidates by weighted field matching."""
        if not candidates:
            return None
        
        # Parse affiliation string
        affiliation_parts = self._parse_affiliation(affiliation)
        
        # Score each candidate
        scored_candidates = []
        
        for candidate in candidates:
            # Extract candidate parts using regex
            match = re.match(r"(.*?)(?:, (.*?))?(?:, (.*?))? \((https://ror\.org/[0-9a-z]+)\)", candidate)
            
            if match:
                name = match.group(1).strip() if match.group(1) else ""
                location = match.group(2).strip() if match.group(2) else ""
                parent = match.group(3).strip() if match.group(3) else ""
                ror_id = match.group(4).replace("https://ror.org/", "")
                
                # Calculate component scores
                name_score = self._calculate_similarity(affiliation_parts.get('name', ''), name)
                location_score = self._calculate_similarity(affiliation_parts.get('location', ''), location)
                parent_score = self._calculate_similarity(affiliation_parts.get('parent', ''), parent)
                
                # Calculate weighted score
                weighted_score = (
                    name_score * self.name_weight +
                    location_score * self.location_weight +
                    parent_score * self.parent_weight
                ) / (self.name_weight + self.location_weight + self.parent_weight)
                
                scored_candidates.append({
                    'ror_id': ror_id,
                    'score': weighted_score
                })
        
        # Sort by score and return top candidate
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            return scored_candidates[0]['ror_id']
        
        return None
    
    def _parse_affiliation(self, affiliation):
        """Parse affiliation string into components."""
        # This is a simple example - replace with better parsing logic
        parts = affiliation.split(',')
        
        result = {}
        
        if len(parts) >= 1:
            result['name'] = parts[0].strip()
        
        if len(parts) >= 2:
            result['location'] = parts[1].strip()
        
        if len(parts) >= 3:
            result['parent'] = parts[2].strip()
        
        return result
    
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two strings."""
        if not text1 or not text2:
            return 0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = words1.intersection(words2)
        
        # Jaccard similarity
        return len(intersection) / len(words1.union(words2))
```

## Custom Language Processing

### Creating a Custom Language Detector

```python
def custom_language_detector(text):
    """
    Detect language using custom logic.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Two-letter language code
    """
    # Your language detection logic here
    # ...
    
    return language_code
```

### Creating a Custom Translator

```python
class CustomTranslator:
    def __init__(self, custom_param=None):
        self.custom_param = custom_param
    
    def translate(self, text):
        """
        Translate text to English.
        
        Args:
            text (str): Text to translate
            
        Returns:
            str: Translated text
        """
        # Your translation logic here
        # ...
        
        return translated_text
    
    def translate_batch(self, texts):
        """
        Translate multiple texts.
        
        Args:
            texts (list of str): Texts to translate
            
        Returns:
            list of str: Translated texts
        """
        return [self.translate(text) for text in texts]
```

### Using Custom Language Processing

```python
# Use custom language detector
language = custom_language_detector("Universidad de Barcelona")

# Use custom translator
translator = CustomTranslator()
translated = translator.translate("Universidad de Barcelona")

# Integrate with AffilGood
from affilgood import AffilGood

def preprocess_affiliations(affiliations):
    """Preprocess affiliations with custom language handling."""
    processed = []
    
    for affiliation in affiliations:
        # Detect language
        language = custom_language_detector(affiliation)
        
        # Translate if needed
        if language != 'en':
            affiliation = translator.translate(affiliation)
        
        processed.append(affiliation)
    
    return processed

# Process with AffilGood
affil_good = AffilGood()
preprocessed = preprocess_affiliations(affiliations)
results = affil_good.process(preprocessed)
```

## Integration with External Systems

### Integration with Database Systems

```python
import sqlite3
from affilgood import AffilGood

# Initialize AffilGood
affil_good = AffilGood()

# Connect to database
conn = sqlite3.connect('affiliations.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS affiliations (
    id INTEGER PRIMARY KEY,
    raw_text TEXT,
    organization TEXT,
    ror_id TEXT
)
''')

# Process affiliations from database
cursor.execute('SELECT id, raw_text FROM affiliations WHERE ror_id IS NULL')
rows = cursor.fetchall()

for row_id, affiliation in rows:
    # Process with AffilGood
    results = affil_good.process([affiliation])
    
    if results and results[0]['ror']:
        # Extract first ROR ID
        ror_id = results[0]['ror'][0].split('{https://ror.org/')[1].split('}')[0]
        
        # Extract organization name
        orgs = []
        for ner in results[0]['ner']:
            if 'ORG' in ner:
                orgs.extend(ner['ORG'])
        organization = orgs[0] if orgs else ''
        
        # Update database
        cursor.execute(
            'UPDATE affiliations SET organization = ?, ror_id = ? WHERE id = ?',
            (organization, ror_id, row_id)
        )

# Commit changes
conn.commit()
conn.close()
```

### Integration with Web APIs

```python
from flask import Flask, request, jsonify
from affilgood import AffilGood

app = Flask(__name__)
affil_good = AffilGood()

@app.route('/process', methods=['POST'])
def process_affiliations():
    data = request.json
    
    if not data or 'affiliations' not in data:
        return jsonify({'error': 'No affiliations provided'}), 400
    
    affiliations = data['affiliations']
    
    try:
        results = affil_good.process(affiliations)
        
        # Format results for API response
        formatted_results = []
        
        for result in results:
            orgs = []
            for ner in result['ner']:
                if 'ORG' in ner:
                    orgs.extend(ner['ORG'])
            
            ror_ids = []
            if result['ror']:
                for ror in result['ror']:
                    if '{https://ror.org/' in ror:
                        parts = ror.split('{https://ror.org/')
                        ror_id = parts[1].split('}')[0]
                        score = parts[1].split('}:')[1] if '}:' in parts[1] else None
                        ror_ids.append({
                            'id': ror_id,
                            'score': float(score) if score else None
                        })
            
            formatted_results.append({
                'raw_text': result['raw_text'],
                'organizations': orgs,
                'ror_ids': ror_ids
            })
        
        return jsonify(formatted_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Best Practices

### Component Customization

1. **Start Small**: Customize one component at a time and test thoroughly.
2. **Maintain Interfaces**: Ensure your custom components follow the expected interfaces.
3. **Handle Edge Cases**: Consider empty inputs, multilingual text, and other edge cases.
4. **Use Caching**: Implement caching for performance-critical components.
5. **Test Integration**: Verify that your custom components work correctly with the rest of the pipeline.

### Performance Optimization

1. **Batch Processing**: Process data in batches for better efficiency.
2. **Lightweight Alternatives**: For simple cases, consider lighter alternatives to model-based components.
3. **Caching**: Cache results to avoid redundant processing.
4. **Parallel Processing**: Use parallel processing for large datasets.
5. **Selective Component Usage**: Only use the components you need for your specific use case.

### Error Handling

1. **Graceful Fallbacks**: Provide fallback mechanisms when a component fails.
2. **Input Validation**: Validate inputs before processing.
3. **Meaningful Error Messages**: Return informative error messages for debugging.
4. **Exception Handling**: Catch and handle exceptions appropriately.
5. **Logging**: Log errors and warnings for troubleshooting.
