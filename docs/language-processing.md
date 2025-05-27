# Language Processing in AffilGood

AffilGood includes language processing capabilities to handle multilingual affiliation data. This document covers language detection, translation, and other language-related features in the AffilGood pipeline.

## Table of Contents

1. [Overview](#overview)
2. [Language Detection](#language-detection)
3. [Translation Capabilities](#translation-capabilities)
4. [Multilingual Support](#multilingual-support)
5. [Configuration Options](#configuration-options)
6. [Performance Considerations](#performance-considerations)
7. [Use Cases](#use-cases)

## Overview

Scientific affiliations appear in many languages, which can challenge entity linking systems. AffilGood addresses this with:

1. **Language Detection**: Identifying the language of affiliation text
2. **Translation**: Converting non-English text to English when beneficial
3. **Multilingual Models**: Using models trained on multilingual data
4. **Language-Specific Handling**: Special processing for certain languages

These capabilities work together to improve the accuracy of the entity linking process for affiliations in any language.

## Language Detection

AffilGood uses a combination of approaches for language detection:

### Heuristic Detection

The `get_language_heur` function in `language_detector.py` uses character patterns and language-specific markers to identify languages:

```python
from affilgood.entity_linking.language_detector import get_language_heur

# Detect language
language = get_language_heur("Universidad de Barcelona, Barcelona, España")
print(language)  # 'es' (Spanish)
```

This approach is particularly effective for:

- Languages with distinctive character sets (e.g., Japanese, Korean, Russian)
- Languages with characteristic diacritical marks (e.g., Spanish ñ, German ß)
- Academic text with institution-specific patterns

### Model-Based Detection

For more complex cases, AffilGood can use model-based language detection:

```python
from affilgood.entity_linking.language_detector import get_language_with_model

# Load a specific model for language detection
from affilgood.entity_linking.language_detector import load_model
model, tokenizer = load_model(model_type='e5')

# Detect language using model
language = get_language_with_model("Universidad de Barcelona, Barcelona, España", model_type='e5')
print(language)  # 'es' (Spanish)
```

AffilGood supports several model types for language detection:

- `e5`: Transformer-based multilingual model
- `fasttext`: FastText language identification model
- `langdetect`: Lightweight language detection library

### Combined Approach

For best results, AffilGood combines heuristic and model-based detection:

```python
from affilgood.entity_linking.language_detector import get_language_combined_heur_e5

# Combined approach
language = get_language_combined_heur_e5("Universidad de Barcelona, Barcelona, España")
print(language)  # 'es' (Spanish)
```

## Translation Capabilities

AffilGood can translate non-English text to English to improve entity matching:

### LLM-Based Translation

The `LLMTranslator` class uses large language models for high-quality translation:

```python
from affilgood.entity_linking.llm_translator import LLMTranslator

# Initialize translator
translator = LLMTranslator(
    skip_english=True,         # Skip translation for English text
    model_name=None,           # Uses default model if None
    use_external_api=False,    # Whether to use external API
    verbose=False,             # Detailed logging
    use_cache=True             # Cache translations
)

# Translate a single affiliation
translated = translator.translate("Universidad de Barcelona, Barcelona, España")
print(translated)  # "University of Barcelona, Barcelona, Spain"

# Translate multiple affiliations
affiliations = [
    "Università degli Studi di Milano, Milano, Italia",
    "Universität Heidelberg, Heidelberg, Deutschland"
]
translated_batch = translator.translate_batch(affiliations)
```

The translator is designed specifically for affiliation strings, with special handling for:

- Institution names (preserving standard English forms)
- Location names (using standard English spellings)
- Academic terms (consistent translation)

### Translation Process

The translation process follows these steps:

1. Detect the language of the input text
2. Skip translation if the text is already in English (optional)
3. Format a prompt for the translation model
4. Generate the translation
5. Clean and post-process the translation
6. Cache the result for future use

### Supported Languages

The translator supports a wide range of languages, including:

- Western European languages (Spanish, French, German, Italian, etc.)
- Eastern European languages (Polish, Czech, Hungarian, etc.)
- East Asian languages (Chinese, Japanese, Korean)
- Arabic, Hebrew, and other scripts

## Multilingual Support

AffilGood's multilingual support extends beyond translation:

### Multilingual Models

AffilGood uses models trained on multilingual data:

- `SIRIS-Lab/affilgood-span-multilingual`: Span identification for multiple languages
- `SIRIS-Lab/affilgood-NER-multilingual`: Named entity recognition for multiple languages
- `SIRIS-Lab/affilgood-affilXLM`: XLM-RoBERTa adapted for multilingual affiliation data

### Language-Specific Processing

For certain languages, AffilGood applies special processing:

#### East Asian Languages

Special handling for Japanese, Chinese, and Korean text:

```python
from affilgood.entity_linking.language_detector import score_japanese_specific, score_chinese_specific, score_korean_specific

# Specialized detection for East Asian languages
jp_score = score_japanese_specific(text)
cn_score = score_chinese_specific(text)
kr_score = score_korean_specific(text)
```

#### Language Adaptation in Entity Linking

The entity linking module adapts to language-specific patterns:

- Handling of diacritical marks and special characters
- Language-aware string normalization
- Script-specific tokenization approaches

## Configuration Options

### Language Detection Configuration

```python
from affilgood.entity_linking.language_detector import load_model

# Configure language detection model
model, tokenizer = load_model(
    model_type='e5'  # 'e5', 'fasttext', 'langdetect', or None
)
```

### Translation Configuration

```python
from affilgood.entity_linking.llm_translator import LLMTranslator

translator = LLMTranslator(
    skip_english=True,         # Skip translation for English text
    model_name="google/gemma-3-27b-it",  # Model to use
    use_external_api=False,    # Whether to use external API
    api_url=None,              # URL for external API (if used)
    api_key=None,              # API key for external API (if used)
    verbose=False,             # Detailed logging
    use_cache=True,            # Use caching for translations
    cache_expire_after=604800  # Cache expiration in seconds (7 days)
)
```

### LLM Translator Caching

The LLM translator includes caching to improve performance:

```python
# Get statistics about the cache
stats = translator.get_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
print(f"Average translation time: {stats['avg_translation_time']:.2f}s")
```

## Performance Considerations

### Memory Usage

- Heuristic language detection: Minimal memory usage
- FastText model: ~100-200MB
- E5 model: ~500MB-1GB
- LLM translator: ~4-10GB depending on model

### Processing Speed

- Heuristic detection: Very fast (1-5ms per text)
- Model-based detection: Moderate (10-50ms per text)
- LLM translation: Slower (100-500ms per text)

### Batch Processing

For efficient processing of many affiliations:

```python
# Batch translation is much more efficient
batch_results = translator.process_batch(affiliations, batch_size=8)
```

### Caching

Enable caching to improve performance for repeated translations:

```python
translator = LLMTranslator(use_cache=True)

# Clear cache if needed
translator.clear_cache(expired_only=True)
```

## Use Cases

### Preprocessing for Entity Linking

Translate affiliations before entity linking to improve matching:

```python
from affilgood import AffilGood
from affilgood.entity_linking.llm_translator import LLMTranslator

# Initialize translator
translator = LLMTranslator(skip_english=True)

# Translate affiliations
affiliations = [
    "Università degli Studi di Milano, Milano, Italia",
    "Universität Heidelberg, Heidelberg, Deutschland"
]
translated = translator.translate_batch(affiliations)

# Process translated affiliations
affil_good = AffilGood()
results = affil_good.process(translated)
```

### Language-Aware Pipeline

Create a pipeline that adapts to the detected language:

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
results = process_multilingual_affiliation("Université Paris-Saclay, Paris, France")
```

### Custom Language Handling

Implement custom language handling for specific languages:

```python
from affilgood.entity_linking.language_detector import get_language_heur

def preprocess_with_language(affiliation):
    # Detect language
    language = get_language_heur(affiliation)
    
    # Language-specific preprocessing
    if language == 'ja':
        # Japanese-specific handling
        # ...
    elif language == 'zh':
        # Chinese-specific handling
        # ...
    elif language == 'ar':
        # Arabic-specific handling
        # ...
    
    return affiliation

# Example usage
preprocessed = preprocess_with_language("東京大学, 東京, 日本")
```