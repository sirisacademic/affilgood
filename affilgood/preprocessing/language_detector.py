#!/usr/bin/env python3

import pycountry
import unicodedata
import re
from unidecode import unidecode
from collections import Counter

# Import the heuristic detection module
from .heuristic_detector import (
    get_language_heuristics,
    score_japanese_specific
)

# Options: 'fasttext', 'e5', 'langdetect', None
LANG_DETECTION_MODEL = None

MODEL = None
TOKENIZER = None

MODEL_PROBS = None

def load_model(model_type='e5'):
    """
    Load the specified language detection model.
    """
    global MODEL, MODEL_PROBS, TOKENIZER, LANG_DETECTION_MODEL
    
    # Update the global variable
    LANG_DETECTION_MODEL = model_type
    
    # Clear previous models
    MODEL = None
    TOKENIZER = None
    
    try:
        if model_type == 'fasttext':
            import fasttext
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
            MODEL = fasttext.load_model(model_path)
            print(f"FastText model loaded from {model_path}")
        elif model_type == 'e5':
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            TOKENIZER = AutoTokenizer.from_pretrained('Mike0307/multilingual-e5-language-detection')
            MODEL = AutoModelForSequenceClassification.from_pretrained('Mike0307/multilingual-e5-language-detection', num_labels=45)
            print("E5 language detection model loaded successfully")
        elif model_type == 'langdetect':
            from langdetect import detect, detect_langs, DetectorFactory
            DetectorFactory.seed =  2 # Set a fixed seed for deterministic results
            MODEL = detect
            MODEL_PROBS = detect_langs
            print("Package langdetect imported successfully")
        else:
            print(f"No model selected (model_type={model_type}). Using heuristic detection only.")
    except Exception as e:
        print(f"Error loading {model_type}: {e}")
    
    # Log the status of the model loading
    if MODEL is None:
        print("No model was loaded successfully.")
    else:
        print(f"Model {model_type} loaded successfully.")
    
    return MODEL, TOKENIZER

def get_language_with_model(text, model_type=None, return_probs=False, default_lang='und'):
    """
    Get language prediction using the specified model.
    """
    global LANG_DETECTION_MODEL, MODEL, TOKENIZER
               
    # Ensure the correct model is loaded
    if model_type and model_type != LANG_DETECTION_MODEL:
        try:
            print(f"==> Loading model: {model_type}")
            load_model(model_type)
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            return default_lang
    
    if MODEL is None:
        print(f"Warning: No model available for {LANG_DETECTION_MODEL}")
        return default_lang
    
    # Use the appropriate detection function
    if LANG_DETECTION_MODEL == 'e5':
        return get_language_e5(text, return_probs=return_probs, default_lang=default_lang)
    elif LANG_DETECTION_MODEL == 'fasttext':
        return get_language_fasttext(text, return_probs=return_probs, default_lang=default_lang)
    elif LANG_DETECTION_MODEL == 'langdetect':
        return get_language_langdetect(text, return_probs=return_probs, default_lang=default_lang)
    else:
        return default_lang

def get_language_combined_heur_langdetect(text, return_probs=False, default_lang='en'):
    """
    Advanced combination of heuristic detection and langdetect probabilities.
    """
 
    # Get heuristic language prediction
    heuristic_lang = get_language_heuristics(text, default_lang=default_lang)

    # Get language probabilities from langdetect
    lang_probs = get_language_with_model(text, model_type='langdetect', return_probs=True)
    
    # Get top langdetect prediction
    langdetect_top = max(lang_probs.items(), key=lambda x: x[1])[0] if lang_probs else default_lang

    if langdetect_top == 'ca':
        return 'ca'

    if langdetect_top == 'nl' and heuristic_lang != 'en':
        return 'nl'
        
    if heuristic_lang in ['en', 'es']:
        return heuristic_lang

    if langdetect_top not in ['ko', 'ja', 'zh']:
        return langdetect_top

    # Special handling for East Asian languages
    if langdetect_top in ['ja', 'zh', 'ko']:
        if langdetect_top == 'ja' or heuristic_lang == 'ja':
            return 'ja'
        if langdetect_top == 'zh':
            return 'zh'
        # Check second best langdetect option
        second_best = sorted(list(lang_probs.items()), key=lambda x: x[1], reverse=True)
        if len(second_best) > 1 and second_best[1][0] == 'zh':
            return 'zh'  # Chinese was second choice
        # Default option is 'ja'
        return 'ja'
        
    elif langdetect_top == heuristic_lang:
        return langdetect_top
        
    else:
        return default_lang

def get_language_combined_heur_e5(text, model_type='e5', default_lang='und'):
    """
    Two-step language detection:
    1. Try enhanced heuristic detection first
    2. If it fails or returns 'und', use model-based detection
    
    Args:
        text (str): The text to analyze
        model_type (str): Type of model to use for fallback ('fasttext', 'e5')
        
    Returns:
        str: Two-letter language code or 'und' if undetermined
    """
    if not isinstance(text, str) or text.strip() == '':
        return 'und'
    
    # First try the enhanced heuristic detection with 'und' as default
    lang_heur = get_language_heuristics(text, default=default_lang)
    
    # Use model-based detection only when heuristic is uncertain
    if lang_heur == 'und':
        try:
            # Use language model detection as fallback
            lang_code = get_language_with_model(text, model_type)
            return lang_code
        except Exception as e:
            print(f"Model detection failed: {e}")
            return lang_heur
    else:
        return lang_heur

def predict_e5(text, device=None):
    """
    Get language prediction probabilities using E5 model.
    
    Args:
        text (str): The text to analyze
        device: PyTorch device to use
        
    Returns:
        torch.Tensor: Probabilities for each language
    """
    
    global LANG_DETECTION_MODEL, MODEL, TOKENIZER
    
    # Safety check
    if LANG_DETECTION_MODEL != 'e5':
        raise ValueError(f"predict_e5 called with incorrect model type: {LANG_DETECTION_MODEL}")
    
    if not hasattr(MODEL, 'to'):
        raise ValueError("E5 model not properly loaded (no 'to' method)")
    
    try:
        import torch
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        MODEL.to(device)
        MODEL.eval()
        
        tokenized = TOKENIZER(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = MODEL(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        return probabilities
    except Exception as e:
        print(f"Error in predict_e5: {e}")
        raise

def get_topk_e5(probabilities, languages, k=3):
    """
    Get top-k language predictions from E5 model.
    
    Args:
        probabilities (torch.Tensor): Probabilities from predict_e5
        languages (list): List of language codes
        k (int): Number of top predictions to return
        
    Returns:
        tuple: Lists of top probabilities and language codes
    """
    import torch
    
    topk_prob, topk_indices = torch.topk(probabilities, k)
    topk_prob = topk_prob.cpu().numpy()[0].tolist()
    topk_indices = topk_indices.cpu().numpy()[0].tolist()
    topk_labels = [languages[index] for index in topk_indices]
    
    return topk_prob, topk_labels

def get_language_e5(text, return_probs=False, default_lang='und'):
    """
    Get language prediction using E5 model.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: Detected language code
    """
    languages = [
        "ar", "eu", "br", "ca", "zh", "zh", "zh", "cv", "cs", "dv", 
        "nl", "en", "eo", "et", "fr", "fy", "ka", "de", "el", "cnh", 
        "id", "ia", "it", "ja", "kab", "rw", "ky", "lv", "mt", "mn", 
        "fa", "pl", "pt", "ro", "rm", "ru", "sah", "sl", "es", "sv", 
        "ta", "tt", "tr", "uk", "cy"
    ]
    
    if MODEL is None:
        print(f"Warning: No model available for {LANG_DETECTION_MODEL}")
        return 'und'
    
    probabilities = predict_e5(text)
    topk_prob, topk_labels = get_topk_e5(probabilities, languages, k=1)
    lang_code = topk_labels[0] if topk_labels else default_lang
    
    return lang_code

def get_language_fasttext(text, default_lang='und'):
    """
    Get language prediction using FastText model.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: Detected language code
    """
    
    if MODEL is None:
        print(f"Warning: No model available for {LANG_DETECTION_MODEL}")
        return default_lang
    
    # Get three-letter language code by means of FastText
    predictions = MODEL.predict(text)
    
    # Extract the language label
    label = predictions[0][0]
    
    # Remove the '__label__' prefix to get the language code
    parts = label.replace("__label__", "").split("_")
    language_code_3chars = parts[0]
    
    return lang_code_3_to_2(language_code_3chars)

def lang_code_3_to_2(code_3):
    """
    Convert ISO 639-3 three-letter language code to ISO 639-1 two-letter code.
    
    Args:
        code_3 (str): Three-letter language code
        
    Returns:
        str: Two-letter language code or 'und' if not found
    """
    try:
        language = pycountry.languages.get(alpha_3=code_3)
        return language.alpha_2 if hasattr(language, 'alpha_2') else 'und'
    except (AttributeError, LookupError):
        return 'und'

def get_language_heur(text, default_lang='en'):
    """
    Detects the language of text based on Unicode character ranges,
    language-specific characters, and institution patterns.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: Language code or 'und' if undetermined
    """
    # Use the enhanced language detection function from the imported module
    return get_language_heuristics(text, default_lang)
    
def get_language_langdetect(text, return_probs=False, default_lang='und'):

    if MODEL is None:
        print(f"Warning: No model available for {LANG_DETECTION_MODEL}")
        return default_lang

    try:
        if return_probs:
            # Returns a list of Language objects with lang and prob attributes
            lang_probabilities = MODEL_PROBS(text)
            # Convert to a dictionary for easier handling
            result = {lang.lang.split('-')[0]: lang.prob for lang in lang_probabilities}
            return result
        else:
            return MODEL(text).split('-')[0]
    except Exception as e:
        print(f"Error detecting language with langdetect: {e}")
        if return_probs:
            return {}
        else:
            return default_lang

