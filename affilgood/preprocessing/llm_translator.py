import re
import warnings
import os
import sys
import io
import logging
import time
import requests
import json
from typing import Dict, Any, List, Optional, Union
from transformers import pipeline, logging as transformers_logging
from datasets import Dataset
from .language_detector import get_language_heur

# Import requests_cache for HTTP caching
import requests_cache

#DEFAULT_MODEL = "TheBloke/neural-chat-7B-v3-2-GPTQ"
DEFAULT_MODEL = "google/gemma-3-27b-it"
DEFAULT_EXTERNAL_MODEL = "google/gemma-2-27b-it"

DISABLE_HF_OUTPUT = False
HF_TOKEN = ""
MODEL_REQUIRES_AUTHENTICATION = False

MAX_NEW_TOKENS = 500  # Adjust based on expected output length
DEFAULT_BATCH_SIZE = 8  # Default batch size for GPU processing

# External API configuration
USE_EXTERNAL_API = False  # Set to True to use external API instead of local model
EXTERNAL_API_URL = "https://api.together.xyz/v1/chat/completions"
EXTERNAL_API_KEY = "..." # Replace with valid API key.
EXTERNAL_API_MAX_RETRIES = 3
EXTERNAL_API_RETRY_DELAY = 2  # seconds

# Cache configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REQUESTS_CACHE_PATH = os.path.join(CURRENT_DIR, 'translation_http_cache')
CACHE_EXPIRATION = 604800  # 7 days in seconds

def get_model_specific_config(model_name: str) -> dict:
    """
    Get model-specific configuration parameters.
    
    Args:
        model_name: Name/identifier of the model
        
    Returns:
        Dictionary with model-specific parameters
    """
    # Base configuration that works for most models
    base_config = {
        'temperature': 0.1,         # Lower temperature for more deterministic output
        'max_tokens': 250,          # Affiliations are short
        'top_p': 0.9,               # Higher top_p for more focused sampling
        'top_k': 40,                # Slightly narrower token selection
        'repetition_penalty': 1.03  # Slight penalty to avoid repetition
    }
    
    # Model-specific adjustments
    if 'llama' in model_name.lower():
        return {
            **base_config,
            'stop': ["<|eot_id|>", "<|eom_id|>"]
        }
    elif 'gemma' in model_name.lower():
        return {
            **base_config,
            'stop': ["<eos>", "<end_of_turn>"]
        }
    elif 'mistral' in model_name.lower() or 'mixtral' in model_name.lower():
        return {
            **base_config,
            'stop': ["</s>"]
        }
    elif 'claude' in model_name.lower():
        return {
            **base_config,
            'stop': ["Human:", "H:"]
        }
    
    # Default configuration
    return base_config

TRANSLATION_PROMPT_LOCAL = """
You are a specialized academic translator focusing on institutional affiliations. 

TRANSLATION PROCESS:
1. FIRST, identify any institution names, cities, or locations in the original text
2. Research the standard English spelling of these proper nouns
3. THEN translate the whole text to English while preserving these identified entities

TRANSLATION RULES:
- Keep all university names, research centers, and geographical locations in their standard English form
- Translate academic titles and departments accurately
- Maintain the original text structure

Text to translate:
"""

TRANSLATION_PROMPT_EXTERNAL = """
Translate the following text to English with EXACT PRECISION. 

CRITICAL RULES:
1. Translate institution names LITERALLY
2. DO NOT substitute any institution with a more famous one
3. Preserve ALL place names and institution names exactly as they appear
4. DO NOT add or remove any information
5. Provide ONLY the direct translation with no explanations

Text to translate:
"""

class LLMTranslator:
    """Translates affiliation strings from any language to English using an LLM."""
            
    def __init__(self, skip_english=True, model_name=None, use_external_api=USE_EXTERNAL_API, 
                 api_url=EXTERNAL_API_URL, api_key=EXTERNAL_API_KEY, verbose=False, 
                 use_cache=True, cache_expire_after=CACHE_EXPIRATION, cache_path=None):
        """
        Initialize the LLM translator.
        
        Args:
            skip_english: Whether to skip translation for English text
            model_name: Name of the Hugging Face model to use
            use_external_api: Whether to use an external API instead of local model
            api_url: URL for the external API (if use_external_api is True)
            api_key: API key for the external API (if use_external_api is True)
            verbose: Whether to show detailed loading information
            use_cache: Whether to use HTTP request caching
            cache_expire_after: Cache expiration time in seconds
            cache_path: Path to the cache directory
        """
        self.verbose = verbose
        self.skip_english = skip_english
        
        self.use_external_api = use_external_api
        
        # Select appropriate model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = DEFAULT_EXTERNAL_MODEL if use_external_api else DEFAULT_MODEL
        
        self.prompt = TRANSLATION_PROMPT_EXTERNAL if use_external_api else TRANSLATION_PROMPT_LOCAL
        
        # External API settings
        self.api_url = api_url
        self.api_key = api_key
        
        # Set up request caching if enabled
        self.use_cache = use_cache
        self.cached_session = None
        
        if self.use_cache and self.use_external_api:
            self._setup_requests_cache(cache_path, cache_expire_after)
        
        # Track stats
        self.stats = {
            "processed": 0,
            "translations_performed": 0,
            "cache_hits": 0,
            "total_processing_time": 0,
            "total_translation_time": 0,
        }
        
        # Initialize pipeline (only for local model)
        self.pipeline = None
        if not self.use_external_api:
            self._load_model()
            
    def _setup_requests_cache(self, cache_path=None, expire_after=CACHE_EXPIRATION):
        """Set up the requests cache for HTTP requests."""
        try:
            cache_path = cache_path if cache_path else REQUESTS_CACHE_PATH
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Initialize the cached session
            self.cached_session = requests_cache.CachedSession(
                cache_name=cache_path,
                backend='sqlite',
                expire_after=expire_after
            )
            
            if self.verbose:
                print(f"HTTP caching initialized at {cache_path}")
                
        except Exception as e:
            self.cached_session = None
            print(f"Warning: Failed to initialize requests_cache: {e}")
            print("Continuing without HTTP caching")
            
    def _load_model(self):
        """Load the LLM model with appropriate logging controls."""
        if self.verbose:
            print(f"Loading local LLM translation model: {self.model_name}")
        
        if MODEL_REQUIRES_AUTHENTICATION:
            try:
                from huggingface_hub import login
                login(HF_TOKEN)
            except Exception as e:
                print(str(e))
        
        if DISABLE_HF_OUTPUT:
            # Store original logging levels
            original_tf_verbosity = transformers_logging.get_verbosity()
            original_logging_level = logging.getLogger().level
            try:
                # Disable all transformers logging
                transformers_logging.set_verbosity_error()
                # Suppress other logging
                logging.getLogger().setLevel(logging.ERROR)
                # Disable warnings
                warnings.filterwarnings("ignore")
                # Redirect stdout/stderr during model loading
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    self.pipeline = pipeline('text-generation', model=self.model_name, device_map="auto")
                finally:
                    # Restore stdout/stderr
                    sys.stdout, sys.stderr = old_stdout, old_stderr
            finally:
                # Restore original logging levels
                transformers_logging.set_verbosity(original_tf_verbosity)
                logging.getLogger().setLevel(original_logging_level)
        else:
            self.pipeline = pipeline('text-generation', model=self.model_name, device_map="auto")

        if self.verbose:
            print(f"LLM translation model loaded successfully")

    def _call_external_api(self, prompt):
        """
        Call an external API for text generation with caching.
        
        Args:
            prompt: The formatted prompt to send to the API
            
        Returns:
            Generated text response from the API
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Get model-specific configuration
        config = get_model_specific_config(self.model_name)
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a specialized academic translator focusing on institutional affiliations."},
                {"role": "user", "content": prompt}
            ],
            **config  # Apply all model-specific configs
        }
        
        # Generate a cache key based on the prompt and model
        cache_key = f"{self.model_name}:{prompt}"
        
        # Implement retry logic for API calls
        for attempt in range(EXTERNAL_API_MAX_RETRIES):
            try:
                if self.verbose:
                    print(f"Calling external API (attempt {attempt+1}/{EXTERNAL_API_MAX_RETRIES})...")
                
                # Use cached session if available, otherwise use regular requests
                if self.cached_session and self.use_cache:
                    response = self.cached_session.post(
                        self.api_url, 
                        headers=headers, 
                        json=data, 
                        timeout=30
                    )
                    
                    # Update stats if this was a cache hit
                    if hasattr(response, 'from_cache') and response.from_cache:
                        self.stats["cache_hits"] += 1
                        if self.verbose:
                            print("Retrieved translation from cache")
                else:
                    response = requests.post(
                        self.api_url, 
                        headers=headers, 
                        json=data, 
                        timeout=30
                    )
                
                response.raise_for_status()  # Raise exception for HTTP errors
                
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    if self.verbose:
                        print(f"Unexpected API response format: {result}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"API call failed: {str(e)}")
                
                # Wait before retrying, unless it's the last attempt
                if attempt < EXTERNAL_API_MAX_RETRIES - 1:
                    retry_delay = EXTERNAL_API_RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    time.sleep(retry_delay)
        
        # If all attempts failed
        if self.verbose:
            print("All API call attempts failed")
        return None

    def translate(self, text: str) -> str:
        """
        Translate the given text to English.
        
        Args:
            text: The text to translate
            
        Returns:
            Translated text
        """
        text = text.strip()        
        
        if not text or not isinstance(text, str):
            return text

        start_time = time.time()
        
        # Detect language
        lang = get_language_heur(text)
        
        # Skip English text if configured to do so
        if self.skip_english and lang == "en":
            return text

        # Format prompt
        prompt = self._format_prompt(text)
        
        # Run translation
        if self.use_external_api:
            response = self._call_external_api(prompt)
            response = response.strip('"').strip() if response else text
            translated_text = response if response else text
        else:
            # Use local model
            pad_token_id = self.pipeline.tokenizer.eos_token_id
            
            # Add safeguards against repeated prompt patterns
            try:
                outputs = self.pipeline(
                    prompt, 
                    max_new_tokens=MAX_NEW_TOKENS, 
                    temperature=0.1, 
                    do_sample=True, 
                    pad_token_id=pad_token_id,
                    num_return_sequences=1  # Get only one output sequence
                )
                
                # Extract translated text
                response = outputs[0]['generated_text'].replace(prompt, '').strip()
                translated_text = self._clean_response(response)
                
                print(translated_text)
                
                # Verify the output is reasonable
                if not translated_text or len(translated_text) < 3:
                    # Fall back to just returning the input
                    if self.verbose:
                        print(f"Warning: Translation produced empty or very short result. Using original text.")
                    translated_text = text
                    
            except Exception as e:
                if self.verbose:
                    print(f"Translation error: {str(e)}")
                translated_text = text  # Fall back to original text
        
        # Update stats
        self.stats["processed"] += 1
        self.stats["translations_performed"] += 1
        processing_time = time.time() - start_time
        self.stats["total_processing_time"] += processing_time
        self.stats["total_translation_time"] += processing_time
        
        if self.verbose and translated_text != text:
            print(f"Original: {text}")
            print(f"Translated: {translated_text}")
            
        return translated_text

    def translate_batch(self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[str]:
        """
        Translate a batch of texts efficiently.
        
        Args:
            texts: List of texts to translate
            batch_size: Number of texts to process in parallel (depends on GPU memory)
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
            
        # Filter out empty or None texts
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        if not valid_texts:
            return [""] * len(texts)
            
        start_time = time.time()
        
        # Check language for all texts first
        if self.skip_english:
            languages = [get_language_heur(text) for text in valid_texts]
        else:
            languages = ["non-en"] * len(valid_texts)  # Placeholder if not skipping
        
        # Process texts differently based on API or local model
        all_results = []
        
        if self.use_external_api:
            # For external API: Process each text individually
            for i, (text, lang) in enumerate(zip(valid_texts, languages)):
                # Skip English texts if configured to do so
                if self.skip_english and lang == "en":
                    all_results.append(text)
                else:
                    translated_text = self.translate(text)
                    all_results.append(translated_text)
        else:
            # For local model: Batch processing
            # First separate English texts (to skip) from non-English ones
            texts_to_translate = []
            english_indices = []
            non_english_indices = []
            
            for i, (text, lang) in enumerate(zip(valid_texts, languages)):
                if self.skip_english and lang == "en":
                    english_indices.append(i)
                else:
                    texts_to_translate.append(text)
                    non_english_indices.append(i)
            
            # Initialize results list with placeholders
            batch_results = [""] * len(valid_texts)
            
            # Fill in English texts (skipped)
            for i in english_indices:
                batch_results[i] = valid_texts[i]
            
            # Process non-English texts in batches
            if texts_to_translate:
                # Create prompts for all texts
                prompts = [self._format_prompt(text) for text in texts_to_translate]
                pad_token_id = self.pipeline.tokenizer.eos_token_id
                
                # Process in smaller batches based on batch_size
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    try:
                        # Process the batch in a single pipeline call
                        outputs = self.pipeline(
                            batch_prompts,
                            max_new_tokens=MAX_NEW_TOKENS,
                            temperature=0.1,
                            do_sample=True,
                            pad_token_id=pad_token_id,
                            batch_size=batch_size
                        )
                        
                        # Extract and clean each response
                        for j, output in enumerate(outputs):
                            if i + j < len(texts_to_translate):
                                response = output['generated_text'].replace(batch_prompts[j], '').strip()
                                translated_text = self._clean_response(response)
                                
                                # Verify output is reasonable
                                if not translated_text or len(translated_text) < 3:
                                    translated_text = texts_to_translate[i + j]
                                
                                # Update results list
                                orig_idx = non_english_indices[i + j]
                                batch_results[orig_idx] = translated_text
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"Batch translation error: {str(e)}")
                        # Fall back to individual processing for this batch
                        for j in range(batch_size):
                            if i + j < len(texts_to_translate):
                                try:
                                    text_idx = i + j
                                    orig_idx = non_english_indices[text_idx]
                                    translated = self.translate(texts_to_translate[text_idx])
                                    batch_results[orig_idx] = translated
                                except:
                                    # If individual translation fails, use original text
                                    batch_results[non_english_indices[i + j]] = texts_to_translate[i + j]
            
            all_results = batch_results
        
        # Update stats
        processing_time = time.time() - start_time
        self.stats["processed"] += len(valid_texts)
        self.stats["translations_performed"] += sum(1 for lang in languages if lang != "en" or not self.skip_english)
        self.stats["total_processing_time"] += processing_time
        self.stats["total_translation_time"] += processing_time
        
        # Map results back to original text list (including empty ones)
        final_results = []
        valid_idx = 0
        for text in texts:
            if text and isinstance(text, str) and text.strip():
                final_results.append(all_results[valid_idx])
                valid_idx += 1
            else:
                final_results.append(text)
        
        return final_results

    def _format_prompt(self, text: str) -> str:
        """Format the prompt for LLM translation."""
        return f'{self.prompt}\n"{text}"\n'

    def _clean_response(self, response: str) -> str:
        """Clean up the LLM response to extract only the translation."""
        # Handle repeated user/assistant patterns that might appear in the output
        if "<|user|>" in response or "<|assistant|>" in response:
            # Extract only the first meaningful response before any repeated patterns
            parts = response.split("<|user|>")
            response = parts[0].strip()
        
        # Remove any explanations or additional text that might follow the translation
        if "Input:" in response:
            response = response.split("Input:")[0]
        
        # Remove any markdown formatting, etc.
        response = response.replace("*", "").replace("#", "").replace("`", "")
        
        # Remove any prefix like "Output:" or "Translation:"
        prefixes = ["Output:", "Translation:", "Translated text:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                
        return response.strip('"').strip()
        
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process a single affiliation string.
        
        Args:
            text: The affiliation string to process
            
        Returns:
            Dict containing:
                - original_text: The original input text
                - is_english: Best guess if original is English (always False when translating)
                - detected_language: Set to "und" as LLM doesn't explicitly identify language
                - processed_text: The translated text
                - translation_performed: Always True for non-empty inputs
        """
        start_time = time.time()
        
        # Simple binary classification: English or non-English
        heur_lang = get_language_heur(text)
        
        result = {
            "original_text": text,
            "is_english": heur_lang == "en",
            "detected_language": heur_lang,
            "processed_text": text,
            "translation_performed": False
        }
        
        # Skip empty text
        if not text or not isinstance(text, str) or text.strip() == "":
            return result
            
        # Skip English
        if self.skip_english and heur_lang == "en":
            return result
            
        # Perform translation
        translated_text = self.translate(text)
        
        # Check if translation changed the text
        if translated_text != text:
            result["processed_text"] = translated_text
            result["translation_performed"] = True
        else:
            # If text didn't change, it was likely English
            result["is_english"] = True
            result["translation_performed"] = True
        
        return result
    
    def process_batch(self, data: Union[Dataset, List[str]], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Dict[str, Any]]:
        """
        Process a batch of affiliation strings, either from a Hugging Face Dataset or a list of strings.
        
        Args:
            data: A Hugging Face Dataset or a list of affiliation strings to process.
            batch_size: The number of texts to process in a single batch.
            
        Returns:
            List of processing results, one for each input text
        """
        # If a Hugging Face Dataset is passed, extract the text column
        if isinstance(data, Dataset):
            texts = data['raw_affiliation_string']  # Adjust the column name as per your dataset
        elif isinstance(data, list):
            texts = data
        else:
            raise TypeError("Input data must be either a Hugging Face Dataset or a list of strings.")
        
        # Process the batch of texts
        start_time = time.time()
        
        # First detect languages for all texts
        languages = [get_language_heur(text) if text and isinstance(text, str) and text.strip() else "und" 
                    for text in texts]
        
        # Filter texts that need translation (non-empty, non-English if skip_english is True)
        texts_to_translate = []
        indices_to_translate = []
        
        for i, (text, lang) in enumerate(zip(texts, languages)):
            if text and isinstance(text, str) and text.strip():
                if not (self.skip_english and lang == "en"):
                    texts_to_translate.append(text)
                    indices_to_translate.append(i)
        
        if self.verbose:
            print(f"Batch contains {len(texts)} texts, {len(texts_to_translate)} need translation")
        
        # Process only texts that need translation
        translated_texts = {}
        if texts_to_translate:
            batch_translations = self.translate_batch(texts_to_translate, batch_size)
            # Map translations back to original indices
            for idx, trans_idx in enumerate(indices_to_translate):
                translated_texts[trans_idx] = batch_translations[idx]
        
        # Create result dictionaries for all texts
        results = []
        for i, (text, lang) in enumerate(zip(texts, languages)):
            # Initialize with defaults
            result = {
                "original_text": text,
                "is_english": lang == "en",
                "detected_language": lang,
                "processed_text": text,
                "translation_performed": False
            }
            
            # Update with translation if performed
            if i in translated_texts:
                translated = translated_texts[i]
                if translated != text:
                    result["processed_text"] = translated
                    result["translation_performed"] = True
            
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics with derived metrics."""
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats["processed"] > 0:
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["processed"]
            stats["cache_hit_rate"] = (stats["cache_hits"] / stats["processed"]) * 100 if stats["processed"] > 0 else 0
        
        if stats["translations_performed"] > 0:
            stats["avg_translation_time"] = stats["total_translation_time"] / stats["translations_performed"]
        
        return stats
