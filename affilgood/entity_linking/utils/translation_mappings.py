import os
from unidecode import unidecode

# Group translations by category for controlled multi-pass replacement
INSTITUTION_TYPE_TRANSLATIONS = {
    'Universität': 'University',
    'Hochschule': 'University',
    'Fachhochschule': 'University of Applied Sciences',
    'Technische Universität': 'Technical University',
    'Technische Hochschule': 'Technical University',
    'Medizinische Universität': 'Medical University',
    'Institut': 'Institute',
    'Fakultät': 'Faculty',
    'Forschung': 'Research',
    'École': 'School',
    'Faculté': 'Faculty',
    'Recherche': 'Research',
    'Universidad': 'University',
    'Universidade': 'University',
    'Università': 'University',
    'Escuela': 'School',
    'Escola': 'School',
    'Scuola': 'School',
    'Instituto': 'Institute',
    'Istituto': 'Institute',
    'Facultad': 'Faculty',
    'Faculdade': 'Faculty',
    'Facoltà': 'Faculty',
    'Investigación': 'Research',
    'Investigação': 'Research',
    'Ricerca': 'Research',
    'Ciencias': 'Sciences',
    'Ciências': 'Sciences',
    'Scienze': 'Sciences',
    'Universiteit': 'University',
    'Hogeschool': 'University of Applied Sciences',
    'Faculteit': 'Faculty',
    'Onderzoek': 'Research',
    'Wetenschappen': 'Sciences',
    'Universitet': 'University',
    'Högskola': 'University',
    'Høgskole': 'University',
}

LOCATION_TRANSLATIONS = {
    'Kärnten': 'Carinthia',
    'Wien': 'Vienna',
    'Steiermark': 'Styria',
    'Tirol': 'Tyrol',
    'Bayern': 'Bavaria',
    'Köln': 'Cologne',
    'München': 'Munich',
    'Zürich': 'Zurich',
    'Genève': 'Geneva',
    'Genf': 'Geneva',
    'Milano': 'Milan',
    'Roma': 'Rome',
    'Napoli': 'Naples',
    'Moskva': 'Moscow',
    'København': 'Copenhagen',
    'Warszawa': 'Warsaw',
    'Praha': 'Prague',
    'Athína': 'Athens',
    'Lisboa': 'Lisbon',
}

def translate_institution_name(name):
    """
    Translate an institution name by applying multiple term replacements.
    First replaces institution type terms, then location/region terms in a separate pass.
    
    Args:
        name (str): The original institution name
        
    Returns:
        list: Potential translations of the name
    """
    if not name:
        return []
    
    translations = []
    original_name = name
    
    # First pass: Apply single-term replacements (like before)
    for original, translated in INSTITUTION_TYPE_TRANSLATIONS.items():
        if original in name:
            translations.append(name.replace(original, translated))
    
    # Second pass: Apply location translations to the original name
    for loc_original, loc_translated in LOCATION_TRANSLATIONS.items():
        if loc_original in name:
            translations.append(name.replace(loc_original, loc_translated))
    
    # Third pass: Apply location translations to the institution type translations
    # to get combined translations (e.g., "University of Applied Sciences Carinthia")
    institution_translations = translations.copy()  # Copy to avoid modifying while iterating
    for trans in institution_translations:
        for loc_original, loc_translated in LOCATION_TRANSLATIONS.items():
            if loc_original in trans:
                translations.append(trans.replace(loc_original, loc_translated))
    
    # Remove duplicates and the original name
    translations = list(set(translations))
    if original_name in translations:
        translations.remove(original_name)
    
    return translations
