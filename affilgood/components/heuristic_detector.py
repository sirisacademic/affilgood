#!/usr/bin/env python3

import unicodedata
import re
from collections import Counter

# Incompatible characters: strong negative signal for a language
incompatible_chars = {
    'es': set('àÀèÈùÙìÌâÂêÊîÎôÔûÛëËïÏœŒæÆ'),
    'fr': set('ñÑßẞøØğĞıİčČśŚžŽłŁãÃõÕ'),
    'de': set('ñÑéèêëáàâãäåçîïìíóòôõúùûœŒæÆ'),
    'pt': set('ñÑßẞéèêëîïìíóòôùúûüœŒ'),
    'it': set('ñÑßẞäÄëËïÏöÖüÜæÆøØœŒãÃõÕ'),
    'ca': set('ñÑêÊôÔâÂïÏîÎûÛãÃõÕœŒæÆ'),
    'nl': set('ñÑßẞøØæÆàÀèÈìÌòÒùÙ'),
    'ro': set('ñÑßẞæÆøØñÑìÌùÙ'),
    'sv': set('ñÑßẞæÆœŒ'),
    'da': set('ñÑßẞœŒ'),
    'fi': set('ñÑßẞæÆøØœŒ'),
    'is': set('ñÑßẞæÆœŒ'),
    'pl': set('ñÑßẞæÆøØœŒâÂêÊîÎôÔûÛ'),
    'cs': set('ñÑßẞæÆøØœŒ'),
    'hu': set('ñÑßẞæÆøØœŒ'),
    'tr': set('ñÑßẞæÆøØœŒ'),
    'no': set('ñÑßẞœŒ'),
    'id': set('äÄëËïÏöÖüÜæÆøØñÑßẞœŒ'),  # Indonesian
}

# Language-specific characters: strong positive signal for a language
language_specific_chars = {
    'es': set('áéíñóúüÁÉÍÑÓÚÜ¿¡'),
    'fr': set('àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ'),
    'de': set('äöüßÄÖÜ'),
    'pt': set('áàâãçéêíóôõúÁÀÂÃÇÉÊÍÓÔÕÚ'),
    'it': set('àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'),
    'ca': set('àçèéíïòóúüÀÇÈÉÍÏÒÓÚÜ·'),
    'nl': set('áéëíóúüïèêÁÉËÍÓÚÜÏÈÊ'),
    'ro': set('ăâîșțĂÂÎȘȚ'),
    'sv': set('åäöÅÄÖ'),
    'da': set('åæøÅÆØ'),
    'fi': set('äöÄÖ'),
    'is': set('áðéíóúýþæöÁÐÉÍÓÚÝÞÆÖ'),
    'pl': set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'),
    'cs': set('čďěňřšťůžČĎĚŇŘŠŤŮŽ'),
    'hu': set('őűŐŰ'),
    'tr': set('çğıİöşüÇĞÖŞÜ'),
    'uk': set('їієґЇІЄҐ'),
    'bg': set('ъьЪЬ'),
    'fa': set('پچژگ'),
    'no': set('åæøÅÆØ'),
    'id': set(''),  # Indonesian generally uses ASCII but with specific patterns
}

# Common bigrams (letter pairs) for various languages
language_bigrams = {
    'en': ['th', 'he', 'in', 'er', 'an', 'ed', 'on', 're', 'at', 'es'],
    'de': ['ch', 'ei', 'ie', 'sc', 'en', 'er', 'in', 'nd', 'te', 'st'],
    'fr': ['ai', 'es', 'le', 'ou', 'qu', 'en', 'on', 'nt', 're', 'de'],
    'es': ['de', 'en', 'qu', 'er', 'es', 'ar', 'la', 'os', 'el', 'ue'],
    'it': ['ch', 'di', 'la', 'co', 'to', 'ri', 'ti', 'er', 'in', 'no'],
    'pt': ['de', 'ar', 'os', 'qu', 'er', 'es', 'do', 'da', 'ão', 'en'],
    'nl': ['en', 'de', 'er', 'ee', 'ij', 'aa', 'an', 'ge', 'ie', 'te'],
    'sv': ['en', 'er', 'et', 'ar', 'de', 'an', 'tt', 'om', 'på', 'fö'],
    'no': ['en', 'er', 'et', 'de', 'om', 'ar', 'st', 'og', 'på', 'il'],
    'da': ['en', 'er', 'et', 'de', 'ar', 'st', 'fo', 'og', 'at', 'af'],
    'fi': ['in', 'en', 'si', 'is', 'an', 'ss', 'aa', 'll', 'ui', 'tä'],
    'pl': ['ie', 'ni', 'cz', 'rz', 'pr', 'zy', 'po', 'na', 'sz', 'ow'],
    'cs': ['vá', 'st', 'ní', 'ro', 'po', 'je', 'né', 'př', 'ho', 'sk'],
    'hu': ['sz', 'el', 'gy', 'en', 'eg', 'na', 'es', 'et', 'ek', 'le'],
    'tr': ['ar', 'in', 'er', 'en', 'an', 'le', 'bi', 'ir', 'ün', 'ka'],
    'ro': ['ul', 'in', 'er', 'ar', 'nt', 're', 'at', 'de', 'la', 'și'],
    'id': ['ng', 'an', 'en', 'me', 'er', 'ka', 'di', 'in', 'be', 'se'],  # Indonesian
}

# Language-specific trigrams
language_trigrams = {
    'en': ['the', 'and', 'ing', 'ion', 'ent', 'ati', 'for', 'her', 'ter', 'hat'],
    'es': ['ión', 'ent', 'que', 'ade', 'aci', 'est', 'con', 'ien', 'tra', 'por'],
    'id': ['ang', 'men', 'eng', 'kan', 'ber', 'ara', 'nga', 'yan', 'ter', 'ata'],
    'ja': ['のア', 'のサ', 'した', 'ます', 'てい', 'たち', 'です', 'する', 'いる', 'れる'],
    'nl': ['een', 'van', 'sch', 'ing', 'ver', 'oor', 'aan', 'den', 'nde', 'eer'],
    'fr': ['ent', 'ion', 'que', 'les', 'ati', 'our', 'ait', 'ans', 'ant', 'lle'],
    'de': ['ein', 'sch', 'die', 'und', 'der', 'che', 'ung', 'eit', 'ich', 'gen'],
    'it': ['ent', 'one', 'che', 'del', 'ato', 'con', 'are', 'ell', 'lla', 'ion'],
    'pt': ['ent', 'çao', 'ção', 'nto', 'ade', 'com', 'ara', 'est', 'que', 'ito'],
}

# Common function words (articles, prepositions, conjunctions) for various languages
common_function_words = {
    'en': {
        'the': 10, 'of': 9, 'and': 9, 'to': 8, 'in': 8, 'for': 7, 'at': 7, 'with': 6, 
        'by': 6, 'from': 5, 'as': 5, 'on': 5, 'this': 4, 'that': 4
    },
    'es': {
        'el': 10, 'la': 10, 'los': 9, 'las': 9, 'de': 9, 'en': 8, 'y': 8, 'a': 8, 
        'que': 7, 'por': 7, 'con': 7, 'para': 6, 'un': 6, 'una': 6, 'se': 5, 'del': 5
    },
    'id': {
        'yang': 10, 'dan': 9, 'di': 9, 'ke': 8, 'pada': 8, 'untuk': 7, 'dengan': 7,
        'dari': 7, 'ini': 6, 'itu': 6, 'oleh': 5, 'atau': 5, 'tidak': 4, 'dalam': 4
    },
    'ja': {
        'の': 10, 'に': 9, 'は': 9, 'を': 8, 'が': 8, 'と': 7, 'で': 7, 'から': 6,
        'より': 6, 'まで': 5, 'など': 5, 'による': 4, 'において': 4
    },
    'nl': {
        'de': 10, 'het': 10, 'een': 9, 'en': 9, 'van': 9, 'in': 8, 'op': 8, 'voor': 7,
        'met': 7, 'door': 6, 'aan': 6, 'bij': 5, 'als': 5, 'uit': 4, 'over': 4
    },
    'fr': {
        'le': 10, 'la': 10, 'les': 9, 'de': 9, 'à': 8, 'des': 8, 'et': 8, 'en': 7,
        'un': 7, 'une': 6, 'du': 6, 'par': 5, 'pour': 5, 'avec': 4, 'sur': 4
    },
    'de': {
        'der': 10, 'die': 10, 'das': 9, 'und': 9, 'in': 8, 'von': 8, 'mit': 7,
        'für': 7, 'auf': 6, 'zu': 6, 'aus': 5, 'bei': 5, 'nach': 4, 'über': 4
    }
}

# Language patterns for more accurate detection
language_patterns = {
    'de': lambda t: ('ß' in t or 'ch' in t.lower()) and any(c in language_specific_chars['de'] for c in t),
    'fr': lambda t: any(c in 'çÇ' for c in t) or ('eau' in t.lower() or 'aux' in t.lower()),
    'es': lambda t: any(c in 'ñÑ¿¡' for c in t) or ('ll' in t.lower() or 'rr' in t.lower()),
    'it': lambda t: ('cch' in t.lower() or 'zz' in t.lower()) and any(c in language_specific_chars['it'] for c in t),
    'pt': lambda t: any(c in 'ãõÃÕ' for c in t) or ('ção' in t.lower()),
    'sv': lambda t: any(c in 'åÅ' for c in t),
    'no': lambda t: ('og' in t.lower() or 'på' in t.lower()) and any(c in language_specific_chars['no'] for c in t),
    'da': lambda t: ('og' in t.lower() or 'af' in t.lower()) and any(c in language_specific_chars['da'] for c in t),
    'fi': lambda t: ('aa' in t.lower() or 'ii' in t.lower()) and any(c in language_specific_chars['fi'] for c in t),
    'nl': lambda t: ('ij' in t.lower() or 'sch' in t.lower()) and any(c in language_specific_chars['nl'] for c in t),
    'ro': lambda t: ('ul' in t.lower() or 'ș' in t or 'ț' in t),
    'is': lambda t: any(c in 'þÞ' for c in t),
    'cs': lambda t: any(char in "řěďťňšžč" for char in t),
    'hu': lambda t: any(char in "őű" for char in t),
    'tr': lambda t: any(char in "ıİğ" for char in t),
    'pl': lambda t: any(char in "łńśźż" for char in t),
    # Added pattern for Indonesian
    'id': lambda t: any(word in t.lower() for word in ['yang', 'dan', 'untuk', 'dengan', 'dari', 'universitas', 'indonesia']),
    # Added pattern for Japanese
    'ja': lambda t: any(0x3040 <= ord(c) <= 0x30FF for c in t) or any(0x4E00 <= ord(c) <= 0x9FFF for c in t)
}

# Unicode script ranges for non-Latin scripts
script_ranges = {
    'ru': [(0x0400, 0x04FF)],  # Russian - Basic Cyrillic
    'uk': [(0x0400, 0x04FF)],  # Ukrainian
    'bg': [(0x0400, 0x04FF)],  # Bulgarian
    'el': [(0x0370, 0x03FF)],  # Greek
    'pl': [(0x0100, 0x024F)],  # Polish
    'cs': [(0x0100, 0x024F)],  # Czech
    'hu': [(0x0100, 0x024F)],  # Hungarian
    'tr': [(0x0100, 0x024F)],  # Turkish
    'ar': [(0x0600, 0x06FF), (0x0750, 0x077F)],  # Arabic
    'he': [(0x0590, 0x05FF)],  # Hebrew
    'fa': [(0x0600, 0x06FF), (0xFB50, 0xFDFF), (0x0750, 0x077F)],  # Persian (Farsi)
    'zh': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],  # Chinese
    'ja': [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF00, 0xFFEF), (0x4E00, 0x9FFF)],  # Japanese
    'ko': [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],  # Korean
    'hi': [(0x0900, 0x097F)],  # Hindi (Devanagari)
    'bn': [(0x0980, 0x09FF)],  # Bengali
    'ta': [(0x0B80, 0x0BFF)],  # Tamil
    'te': [(0x0C00, 0x0C7F)],  # Telugu
    'kn': [(0x0C80, 0x0CFF)],  # Kannada
    'ml': [(0x0D00, 0x0D7F)],  # Malayalam
    'th': [(0x0E00, 0x0E7F)],  # Thai
    'km': [(0x1780, 0x17FF)],  # Khmer
    'my': [(0x1000, 0x109F)],  # Burmese (Myanmar)
    'lo': [(0x0E80, 0x0EFF)],  # Lao
    'am': [(0x1200, 0x137F)],  # Amharic (Ethiopic)
    'ka': [(0x10A0, 0x10FF)],  # Georgian
    'hy': [(0x0530, 0x058F)],  # Armenian
}

# Unified academic and affiliation keywords (merging academic_common_words and affiliation_hints_weighted)
academic_keywords = {
    'fr': {
        'université': 10, 'école': 8, 'institut': 9, 'laboratoire': 8, 'recherche': 7, 
        'france': 6, 'collège': 5, 'école normale': 5, 'centre de recherche': 7, 
        'sciences': 4, 'arts': 3, 'médecine': 2, 'faculté': 6, 'département': 5,
        'sorbonne': 8, 'enseignement': 4, 'étude': 3, 'académie': 5
    },
    'es': {
        'universidad': 10, 'instituto': 9, 'investigaciones': 8, 'autónoma': 7, 
        'politécnica': 6, 'departamento': 5, 'españa': 5, 'facultad': 4, 'escuela': 3, 
        'centro de investigación': 6, 'escuela técnica': 4, 'ciencias': 3, 'tecnología': 6, 
        'ingeniería': 5, 'estudios': 5, 'colegio': 4, 'academia': 4
    },
    'id': {
        'universitas': 10, 'indonesia': 9, 'institut': 8, 'penelitian': 7, 'fakultas': 6, 
        'pusat': 5, 'teknologi': 5, 'departemen': 4, 'ilmu': 4, 'jakarta': 3, 'bandung': 3,
        'laboratorium': 9, 'teknik': 5, 'studi': 5, 'sekolah': 5, 'perguruan': 4, 'akademi': 4
    },
    'ja': {
        '大学': 10, '研究所': 9, '東京': 9, '日本': 9, '学部': 7, '学院': 6, 'センター': 6, 
        '研究センター': 7, '実験室': 9, '学科': 8, '科学': 6, '技術': 6, '工学': 5, 
        '研究科': 5, '学校': 5, 'アカデミー': 4
    },
    'nl': {
        'universiteit': 10, 'nederland': 9, 'faculteit': 7, 'instituut': 8, 'centrum': 6, 
        'onderzoek': 5, 'technische': 7, 'school': 5, 'academie': 6, 'laboratorium': 9, 
        'afdeling': 8, 'wetenschap': 6, 'technologie': 6, 'techniek': 5, 'studie': 5,
        'hogeschool': 4, 'amsterdam': 8, 'leiden': 8, 'utrecht': 8, 'nijmegen': 8
    },
    'en': {
        'university': 10, 'college': 9, 'institute': 8, 'center': 7, 'research': 6, 
        'school': 5, 'uk': 6, 'usa': 6, 'canada': 6, 'department': 5, 'faculty': 6, 
        'academy': 5, 'institute of technology': 7, 'institute of science': 6,
        'laboratory': 9, 'science': 6, 'technology': 6, 'engineering': 5, 'studies': 5
    },
    'de': {
        'universität': 10, 'hochschule': 9, 'fraunhofer': 8, 'technische': 8, 
        'deutschland': 7, 'akademie': 7, 'institut': 6, 'forschungszentrum': 6, 
        'schule': 5, 'wissenschaften': 6, 'medizinfakultät': 4, 'berlin': 8, 'münchen': 8
    },
    'it': {
        'università': 10, 'dipartimento': 9, 'italia': 8, 'scuola': 7, 'istituto': 8, 
        'centro di ricerca': 6, 'facoltà': 5, 'politecnico': 7, 'accademia': 6, 
        'scienze': 4, 'arte': 3, 'roma': 8, 'milano': 8, 'torino': 8
    },
    'pt': {
        'universidade': 10, 'brasil': 9, 'portugal': 9, 'instituto': 8, 'faculdade': 7, 
        'departamento': 6, 'escola': 5, 'centro de pesquisa': 6, 'tecnologia': 5, 'ciências': 3,
        'são paulo': 8, 'rio de janeiro': 8, 'lisboa': 8
    }
}

def score_language_chars(text):
    """
    Score text based on language-specific characters with penalty handling.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    lang_chars_count = {}
    for lang, char_set in language_specific_chars.items():
        count = sum(1 for char in text if char in char_set)
        if lang in incompatible_chars:
            penalty = sum(1 for char in text if char in incompatible_chars[lang])
            count = max(0, count - penalty * 2)  # Penalty factor can be adjusted
        if count > 0:
            lang_chars_count[lang] = count
    return lang_chars_count


def score_language_bigrams(text):
    """
    Score text based on language-specific bigrams.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    text_lower = text.lower()
    lang_scores = {lang: 0 for lang in language_bigrams}
    
    # Count bigram occurrences for each language
    for lang, bigrams in language_bigrams.items():
        for bigram in bigrams:
            lang_scores[lang] += text_lower.count(bigram)
    
    # Normalize by text length to avoid bias from longer texts
    text_len = max(1, len(text))
    for lang in lang_scores:
        lang_scores[lang] = (lang_scores[lang] / text_len) * 100
        
    return lang_scores


def extract_ngrams(text, n=3):
    """
    Extract n-grams from text.
    
    Args:
        text (str): The text to analyze
        n (int): Size of n-grams (2 for bigrams, 3 for trigrams, etc.)
        
    Returns:
        list: List of n-grams
    """
    # Normalize and clean the text
    text = unicodedata.normalize("NFC", text.lower())
    words = re.findall(r'\w+', text)
    
    # Extract character n-grams from each word
    char_ngrams = []
    for word in words:
        if len(word) >= n:
            for i in range(len(word) - n + 1):
                char_ngrams.append(word[i:i+n])
    
    return char_ngrams


def score_language_trigrams(text):
    """
    Score text based on language-specific trigrams.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    text_trigrams = extract_ngrams(text, 3)
    lang_scores = {lang: 0 for lang in language_trigrams}
    
    if not text_trigrams:
        return {}
    
    # Count trigram occurrences for each language
    for lang, trigrams in language_trigrams.items():
        for trigram in trigrams:
            lang_scores[lang] += text_trigrams.count(trigram)
    
    # Normalize by the number of trigrams to avoid bias from longer texts
    text_len = max(1, len(text_trigrams))
    for lang in lang_scores:
        lang_scores[lang] = (lang_scores[lang] / text_len) * 100
    
    return {k: v for k, v in lang_scores.items() if v > 0}


def score_script_ranges(text):
    """
    Count characters in each Unicode script range for each language.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    lang_counts = {}
    for lang, ranges in script_ranges.items():
        count = 0
        for start, end in ranges:
            count += sum(1 for char in text if start <= ord(char) <= end)
        if count > 0:
            lang_counts[lang] = count
    return lang_counts


def score_common_words(text, word_dict):
    """
    Calculate weighted scores for common word matches.
    
    Args:
        text (str): The text to analyze
        word_dict (dict): Dictionary of languages with their common words and weights
        
    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    # Convert text to lowercase and extract words
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    language_scores = {lang: 0 for lang in word_dict}
    word_counts = Counter(words)
    
    for lang, keywords in word_dict.items():
        for word, weight in keywords.items():
            count = word_counts.get(word, 0)
            if count > 0:
                language_scores[lang] += weight * count
    
    # Normalize by text length to prevent bias towards longer texts
    text_len = max(1, len(words))
    for lang in language_scores:
        language_scores[lang] = (language_scores[lang] / text_len) * 100
                
    return {k: v for k, v in language_scores.items() if v > 0}


def score_academic_keywords(text):
    """
    Calculate weighted scores for academic keyword matches.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: Dictionary with language codes as keys and scores as values
    """
    text_lower = text.lower()
    language_scores = {lang: 0 for lang in academic_keywords}
    
    for lang, keywords in academic_keywords.items():
        for kw, weight in keywords.items():
            if re.search(rf'\b{re.escape(kw)}\b', text_lower):
                language_scores[lang] += weight
                
    return language_scores


def is_latin_script(text):
    """
    Check if text contains only Latin characters (including accented).
    
    Args:
        text (str): The text to analyze
        
    Returns:
        bool: True if text is Latin script, False otherwise
    """
    # Check if all characters are within Latin ranges or not alphabetic
    return all(
        (not c.isalpha()) or  # Skip non-alphabetic characters
        (ord(c) < 0x0530) or  # Basic Latin, Latin-1 Supplement, Latin Extended A/B
        (0x1E00 <= ord(c) <= 0x1EFF)  # Latin Extended Additional
        for c in text
    )


def register_language(lang_code, 
                      specific_chars=None, 
                      incompatible=None,
                      bigrams=None, 
                      trigrams=None,
                      function_words=None,
                      academic_kws=None,
                      script_ranges_list=None,
                      pattern_function=None):
    """
    Register a new language in the detection system.
    
    Args:
        lang_code (str): ISO 639-1 two-letter language code
        specific_chars (set): Set of language-specific characters
        incompatible (set): Set of incompatible characters
        bigrams (list): List of common bigrams
        trigrams (list): List of common trigrams
        function_words (dict): Dictionary mapping common words to weights
        academic_kws (dict): Dictionary mapping academic words to weights
        script_ranges_list (list): List of Unicode code point ranges (tuples)
        pattern_function (callable): Function that takes text and returns bool if pattern matches
    """
    global language_specific_chars, incompatible_chars, language_bigrams
    global language_trigrams, common_function_words, academic_keywords
    global script_ranges, language_patterns
    
    # Register language-specific characters
    if specific_chars:
        language_specific_chars[lang_code] = specific_chars
    
    # Register incompatible characters
    if incompatible:
        incompatible_chars[lang_code] = incompatible
    
    # Register bigrams
    if bigrams:
        language_bigrams[lang_code] = bigrams
    
    # Register trigrams
    if trigrams:
        language_trigrams[lang_code] = trigrams
    
    # Register function words
    if function_words:
        common_function_words[lang_code] = function_words
    
    # Register academic words
    if academic_kws:
        academic_keywords[lang_code] = academic_kws
    
    # Register script ranges
    if script_ranges_list:
        script_ranges[lang_code] = script_ranges_list
    
    # Register pattern function
    if pattern_function:
        language_patterns[lang_code] = pattern_function
    
    print(f"Language '{lang_code}' registered successfully")


def get_language_heuristics(text, default_lang='en'):
    """
    Enhanced language detection using multiple features
    """
    if not isinstance(text, str) or text.strip() == '':
        return default_lang
    
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    
    # Handle empty text or text with only ASCII digits/punctuation
    if not text or all(ord(c) < 128 and (c.isdigit() or not c.isalnum()) for c in text):
        return default_lang
    
    # Evaluate East Asian languages with improved differentiation
    jp_score = score_japanese_specific(text)
    kr_score = score_korean_specific(text)
    cn_score = score_chinese_specific(text)
    
    # Make decisions based on scores
    if jp_score > 0:
        return 'ja'  # If any clear Japanese indicators, prioritize Japanese
    
    if kr_score > 0:
        return 'ko'  # If any clear Korean indicators, return Korean
    
    if cn_score > 0:
        return 'zh'  # If Chinese indicators and no Japanese or Korean, return Chinese
    
    # Check for script ranges (as before)
    script_scores = score_script_ranges(text)
    
    # Handle Cyrillic and other scripts similar to your existing code
    if any(lang in script_scores for lang in ['ru', 'uk', 'bg']):
        if any(char in language_specific_chars.get('uk', set()) for char in text):
            return 'uk'
        elif any(char in language_specific_chars.get('bg', set()) for char in text):
            return 'bg'
        elif 'ru' in script_scores:
            return 'ru'
    
    # Persian vs Arabic
    if 'fa' in script_scores and any(char in language_specific_chars.get('fa', set()) for char in text):
        return 'fa'
    elif 'ar' in script_scores:
        return 'ar'
    
    # If we have a match in script ranges, return the one with highest score
    if script_scores:
        return max(script_scores, key=script_scores.get)
    
    # For Latin scripts, combine multiple features
    if is_latin_script(text):
        # Initialize combined scores
        combined_scores = {}
        
        # 1. Language-specific characters
        char_scores = score_language_chars(text)
        for lang, score in char_scores.items():
            combined_scores[lang] = combined_scores.get(lang, 0) + score * 2  # Higher weight
            
        # 2. Bigrams
        bigram_scores = score_language_bigrams(text)
        for lang, score in bigram_scores.items():
            combined_scores[lang] = combined_scores.get(lang, 0) + score
            
        # 3. Trigrams
        trigram_scores = score_language_trigrams(text)
        for lang, score in trigram_scores.items():
            combined_scores[lang] = combined_scores.get(lang, 0) + score * 1.5  # Higher weight
            
        # 4. Common function words
        function_word_scores = score_common_words(text, common_function_words)
        for lang, score in function_word_scores.items():
            combined_scores[lang] = combined_scores.get(lang, 0) + score * 10  # Higher weight
            
        # 5. Academic keywords (especially important for affiliation strings)
        academic_word_scores = score_academic_keywords(text)
        for lang, score in academic_word_scores.items():
            combined_scores[lang] = combined_scores.get(lang, 0) + score * 2.5
            
        # Apply language patterns for better precision on specific cases
        for lang, pattern_fn in language_patterns.items():
            if pattern_fn(text):
                combined_scores[lang] = combined_scores.get(lang, 0) + 50  # Strong boost
        
        # First check for highly distinctive characters which are strong indicators
        if any(c in 'ãõÃÕ' for c in text):  # Unique to Portuguese
            combined_scores['pt'] = combined_scores.get('pt', 0) + 75
        if any(c in 'ñÑ' for c in text):  # Distinctive for Spanish
            combined_scores['es'] = combined_scores.get('es', 0) + 75
        if any(c in 'åÅ' for c in text):  # Nordic languages
            if any(c in 'æøÆØ' for c in text):
                if 'på' in text.lower():
                    combined_scores['no'] = combined_scores.get('no', 0) + 75
                else:
                    combined_scores['da'] = combined_scores.get('da', 0) + 75
            else:
                combined_scores['sv'] = combined_scores.get('sv', 0) + 75
        if any(c in 'őűŐŰ' for c in text):  # Hungarian
            combined_scores['hu'] = combined_scores.get('hu', 0) + 75
        if any(c in 'řěďťňŘĚĎŤŇ' for c in text):  # Czech
            combined_scores['cs'] = combined_scores.get('cs', 0) + 75
        if any(c in 'łńśźżŁŃŚŹŻ' for c in text):  # Polish
            combined_scores['pl'] = combined_scores.get('pl', 0) + 75
        if any(c in 'ıİğĞ' for c in text):  # Turkish
            combined_scores['tr'] = combined_scores.get('tr', 0) + 75
        if any(c in 'șțȘȚ' for c in text):  # Romanian
            combined_scores['ro'] = combined_scores.get('ro', 0) + 75
        if any(c in 'þÞðÐ' for c in text):  # Icelandic
            combined_scores['is'] = combined_scores.get('is', 0) + 75
        if 'ß' in text:  # German
            combined_scores['de'] = combined_scores.get('de', 0) + 75
            
        # Special case for Indonesian
        if any(word in text.lower() for word in ['indonesia', 'universitas', 'depok', 'jakarta']):
            combined_scores['id'] = combined_scores.get('id', 0) + 75
            
        # Special case for Dutch
        if 'ij' in text.lower() and any(word in text.lower() for word in ['universiteit', 'nederland', 'amsterdam']):
            combined_scores['nl'] = combined_scores.get('nl', 0) + 75
        
        # Select language with highest combined score
        if combined_scores:
            return max(combined_scores, key=combined_scores.get)
        
        # Default to English if there are no accented characters
        text_lower = text.lower()
        if not any(char in 'áéíóúàèìòùâêîôûäëïöü' for char in text_lower):
            return default_lang
    
    # Default to the specified default value if nothing else matches
    return default_lang


def score_japanese_specific(text):
    """
    Special detection for Japanese based on specific character ranges.
    """
    # Count characters in Japanese-specific ranges
    hiragana_count = sum(1 for c in text if 0x3040 <= ord(c) <= 0x309F)
    katakana_count = sum(1 for c in text if 0x30A0 <= ord(c) <= 0x30FF)
    kanji_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    
    # Japanese-specific punctuation
    jp_punct_count = sum(1 for c in text if c in '、。「」『』・')
    
    # Only count as Japanese if there are kana characters
    # This is the key change - require hiragana or katakana to identify as Japanese
    if hiragana_count == 0 and katakana_count == 0:
        return 0  # Not Japanese if no kana
    
    # Apply weighted scoring (can be adjusted)
    score = hiragana_count * 2 + katakana_count * 1.5 + kanji_count * 0.5 + jp_punct_count * 0.5
    
    return score
    
def score_chinese_specific(text):
    """
    Special detection for Chinese based on specific character ranges.
    """
    # Count Han characters (without kana presence)
    han_count = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    
    # Chinese-specific punctuation
    cn_punct_count = sum(1 for c in text if c in '，。：""''；？！（）')
    
    # Check for absence of Japanese-specific characters
    hiragana_count = sum(1 for c in text if 0x3040 <= ord(c) <= 0x309F)
    katakana_count = sum(1 for c in text if 0x30A0 <= ord(c) <= 0x30FF)
    
    # If there's any kana, it's likely not pure Chinese
    if hiragana_count > 0 or katakana_count > 0:
        return 0
    
    # Apply weighted scoring
    score = han_count * 1 + cn_punct_count * 0.5
    
    return score

def score_korean_specific(text):
    """
    Special detection for Korean based on Hangul presence.
    """
    # Count Hangul characters
    hangul_count = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)
    
    # Korean-specific punctuation and other Korean-specific ranges
    kr_punct_count = sum(1 for c in text if c in '…·')
    
    # Count Hangul Jamo (Korean alphabet components)
    jamo_count = sum(1 for c in text if 0x1100 <= ord(c) <= 0x11FF)
    
    # If there's no Hangul, it's not Korean
    if hangul_count == 0 and jamo_count == 0:
        return 0
    
    # Apply weighted scoring
    score = hangul_count * 2 + jamo_count * 1.5 + kr_punct_count * 0.5
    
    return score
    
