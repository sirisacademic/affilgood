# Language Processing in AffilGood

AffilGood includes language processing capabilities to handle multilingual affiliation data. However, **translation is disabled by default** because the multilingual models used for retrieval and reranking already support most languages effectively, and translation often does not improve results while adding processing overhead.

## Table of Contents

1. [Overview](#overview)
2. [Language Detection](#language-detection)
3. [Translation Capabilities](#translation-capabilities)
4. [Multilingual Support](#multilingual-support)
5. [Configuration Options](#configuration-options)
6. [Use Cases](#use-cases)

## Overview

AffilGood handles multilingual affiliations through:

1. **Multilingual Models**: The default NER and retrieval models support multiple languages natively
2. **Language Detection**: Identifying the language of affiliation text (available but not used by default)
3. **Optional Translation**: Converting non-English text to English when explicitly enabled
4. **Language-Specific Processing**: Automatic handling of different scripts and character sets

**Important**: Translation is **disabled by default** because:
- The multilingual models (affilgood-NER-multilingual, affilgood-span-multilingual) handle most languages effectively
- Translation adds processing time and complexity
- For most supported languages, translation does not improve matching accuracy
- The dense retrieval models work well with multilingual text

## Language Detection

AffilGood includes language detection capabilities in the `preprocessing` module, though these are not used in the main pipeline by default.

### Heuristic Detection

The `get_language_heuristics` function uses character patterns and language-specific markers:

```python
from affilgood.preprocessing.heuristic_detector import get_language_heuristics

# Detect language
language = get_language_heuristics("Universidad de Barcelona, Barcelona, España")
print(language)  # 'es' (Spanish)
```

This approach works well for:
- Languages with distinctive character sets (Japanese, Korean, Russian)
- Languages with characteristic diacritical marks (Spanish ñ, German ß)
- Academic text with institution-specific patterns

### Combined Detection

For more robust detection, you can combine heuristic and model-based approaches:

```python
from affilgood.preprocessing.language_detector import get_language_combined_heur_langdetect

# Combined approach using heuristics + langdetect
language = get_language_combined_heur_langdetect("Universidad de Barcelona, Barcelona, España")
print(language)  # 'es' (Spanish)
```

## Translation Capabilities

Translation is available through the `LLMTranslator` class but is **not enabled by default** in the main AffilGood pipeline.

### LLM-Based Translation

The `LLMTranslator` supports two modes of operation:

#### Option 1: Local Model (Default)

Uses a local transformer model for translation:

```python
from affilgood.preprocessing.llm_translator import LLMTranslator

# Initialize with local model (default)
translator = LLMTranslator(
    skip_english=True,         # Skip translation for English text
    model_name="google/gemma-2-27b-it",  # Local model to use
    use_external_api=False,    # Use local model (default)
    verbose=False,
    use_cache=True             # Cache translations
)

# Translate affiliations using local model
translated = translator.translate("Universidad de Barcelona, Barcelona, España")
print(translated)  # "University of Barcelona, Barcelona, Spain"
```

**Pros**: Privacy, no API costs, offline capability  
**Cons**: Requires significant GPU memory (4-10GB), slower inference

#### Option 2: External API

Uses an external API service for translation:

```python
from affilgood.preprocessing.llm_translator import LLMTranslator

# Initialize with external API
translator = LLMTranslator(
    skip_english=True,
    model_name="google/gemma-2-27b-it",    # Model available via API
    use_external_api=True,                 # Enable API mode
    api_url="https://api.together.xyz/v1/chat/completions",  # API endpoint
    api_key="your-api-key-here",           # API authentication
    verbose=False,
    use_cache=True                         # Cache API responses
)

# Translate using external API
translated = translator.translate("Universidad de Barcelona, Barcelona, España")
```

**Pros**: Faster inference, no local GPU requirements, potentially better models  
**Cons**: Requires internet connection, API costs, data privacy considerations

#### Batch Translation

Both modes support batch processing:

```python
# Translate multiple affiliations (works with both local and API modes)
affiliations = [
    "Università degli Studi di Milano, Milano, Italia",
    "Universität Heidelberg, Heidelberg, Deutschland"
]
translated_batch = translator.translate_batch(affiliations, batch_size=8)
```

### When to Use Translation

Consider enabling translation for:
- **Legacy systems** that only work with English text
- **Specific domains** where translation has been validated to improve results
- **Custom workflows** where English normalization is required

**Note**: For most use cases, the multilingual models provide better results without translation.

## Multilingual Support

AffilGood's primary multilingual support comes through native multilingual models rather than translation.

### Multilingual Models

AffilGood uses multilingual models by default:

- `SIRIS-Lab/affilgood-span-multilingual`: Span identification for multiple languages
- `SIRIS-Lab/affilgood-NER-multilingual`: Named entity recognition for multiple languages
- Dense retrieval models that work with multilingual text

```python
from affilgood import AffilGood

# Default configuration uses multilingual models
affil_good = AffilGood()

# Process multilingual affiliations directly
affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Università degli Studi di Milano, Milano, Italia", 
    "Universität Heidelberg, Heidelberg, Deutschland",
    "東京大学, 東京, 日本"
]

# No translation needed - models handle multiple languages
results = affil_good.process(affiliations)
```

### Language-Specific Processing

The entity linking components automatically handle:
- Different character encodings and scripts
- Language-specific name variations
- Cross-lingual semantic similarity

## Configuration Options

### Translation Configuration (Optional)

Translation is not configured by default. Here are the configuration options for both modes:

#### Local Model Configuration

```python
from affilgood.preprocessing.llm_translator import LLMTranslator

# Configure for local model usage
translator = LLMTranslator(
    skip_english=True,                    # Skip English text
    model_name="google/gemma-2-27b-it",   # Local transformer model
    use_external_api=False,               # Use local model (default)
    verbose=False,
    use_cache=True,                       # Cache results locally
    cache_expire_after=604800             # Cache for 7 days
)
```

#### External API Configuration

```python
from affilgood.preprocessing.llm_translator import LLMTranslator

# Configure for external API usage
translator = LLMTranslator(
    skip_english=True,                    # Skip English text
    model_name="google/gemma-2-27b-it",   # Model available via API
    use_external_api=True,                # Enable API mode
    api_url="https://api.together.xyz/v1/chat/completions",  # API endpoint
    api_key="your-api-key-here",          # Your API key
    verbose=False,
    use_cache=True,                       # Cache API responses
    cache_expire_after=604800             # Cache for 7 days
)
```

#### Performance Comparison

| Mode | Memory Usage | Speed | Cost | Privacy | Offline |
|------|-------------|-------|------|---------|---------|
| Local Model | 4-10GB GPU | Slower | Hardware only | High | Yes |
| External API | Minimal | Faster | Pay per use | Lower | No |

#### Getting Translation Statistics

Both modes provide usage statistics:

```python
# Get translation statistics
stats = translator.get_stats()
print(f"Translations performed: {stats['translations_performed']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Average translation time: {stats.get('avg_translation_time', 0):.2f}s")

# For API mode, also shows cache hit rate
if translator.use_external_api:
    print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
```

### Language Detection Configuration

Language detection is available but not used in the main pipeline:

```python
from affilgood.preprocessing.language_detector import load_model

# Configure language detection model (optional)
model, tokenizer = load_model(model_type='langdetect')  # or 'e5', 'fasttext'
```

## Use Cases

### Preprocessing with Translation (Advanced)

If you specifically need translation, you can preprocess affiliations:

```python
from affilgood import AffilGood
from affilgood.preprocessing.llm_translator import LLMTranslator

# Only use translation if specifically needed
translator = LLMTranslator(skip_english=True)

# Translate affiliations before processing
affiliations = [
    "Università degli Studi di Milano, Milano, Italia",
    "Universität Heidelberg, Heidelberg, Deutschland"
]
translated = translator.translate_batch(affiliations)

# Process translated affiliations
affil_good = AffilGood()
results = affil_good.process(translated)
```

### Direct Multilingual Processing (Recommended)

The recommended approach is to process multilingual text directly:

```python
from affilgood import AffilGood

# Direct processing - no translation needed
affil_good = AffilGood()

# Process multilingual affiliations directly
affiliations = [
    "Universidad de Barcelona, Barcelona, España",
    "Université Paris-Saclay, Paris, France", 
    "Universität Wien, Wien, Österreich"
]

# Multilingual models handle different languages automatically
results = affil_good.process(affiliations)

for result in results:
    print(f"Original: {result['raw_text']}")
    print(f"ROR: {result['ror']}")
    print("---")
```

### Language-Specific Analysis (Optional)

For analysis purposes, you can detect languages:

```python
from affilgood.preprocessing.language_detector import get_language_heuristics

def analyze_affiliation_languages(affiliations):
    """Analyze the languages present in affiliations."""
    language_counts = {}
    
    for affiliation in affiliations:
        lang = get_language_heuristics(affiliation)
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    return language_counts

# Example usage
affiliations = [
    "Stanford University, CA, USA",
    "Universidad de Barcelona, España", 
    "Université de Paris, France"
]

lang_stats = analyze_affiliation_languages(affiliations)
print(f"Language distribution: {lang_stats}")
```

## Summary

- **Default behavior**: AffilGood processes multilingual text directly using multilingual models
- **Translation**: Available but disabled by default; only enable if specifically needed
- **Language detection**: Available for analysis but not used in the main pipeline
- **Recommendation**: Use the default multilingual processing for best results and performance