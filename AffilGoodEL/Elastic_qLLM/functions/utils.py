import os 
import pandas as pd
from unidecode import unidecode

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# File with alternative names for countries and language codes.
COUNTRIES_FILE = f'{DIR_PATH}/countries_languages.tsv'
COUNTRY_ENG_NAME_COL = 'country_exonym'
COUNTRY_ALT_NAMES_COL = 'country_alternative'
COUNTRY_LANG_CODES_COL = 'lang_codes'
COUNTRY_COL_SEPARATOR = '|'

# Load countries.
countries = pd.read_csv(COUNTRIES_FILE, sep='\t')
countries.fillna('', inplace=True)
countries.drop_duplicates(subset='country_code', inplace=True)
countries.set_index('country_code', inplace=True)
countries_dict = countries.to_dict('index')

############################################
### Functions

def get_variants_text(text):
#--------------------------
  variants = list(set([text, unidecode(text)])) if text else []
  return variants

def get_variants_list(list_texts):
#--------------------------------
  variants = [text for text in list_texts if text]
  variants.extend([unidecode(text) for text in list_texts if text])
  return list(set(variants))
  
def get_variants_country(country_code):
#-------------------------------------
  country_names = []
  country_code = country_code.upper()
  if country_code in countries_dict:
    country = countries_dict[country_code]
    country_names.append(country[COUNTRY_ENG_NAME_COL])
    country_names.extend([c.strip() for c in country[COUNTRY_ALT_NAMES_COL].split(COUNTRY_COL_SEPARATOR)])
  return get_variants_list(country_names)

def get_languages_country(country_code):
#-------------------------------------
  country_languages = []
  country_code = country_code.upper()
  if country_code in countries_dict:
    country = countries_dict[country_code]
    country_languages.extend([l.strip().lower() for l in country[COUNTRY_LANG_CODES_COL].split(COUNTRY_COL_SEPARATOR)])
  return get_variants_list(country_languages)

# TODO: Load from file, separate by language.
def get_legal_entities():
#-----------------------
  list_legal_types = ['AB', 'AG', 'BV', 'CORP', 'CV', 'EIRELI', 'GMBH', 'GmbH', '&', 'CO' 'KG' 'HB', 'KB', 'KG', 'LDA', 'LLC', 'LLP', 'LTD', 'LIMITED', 'NV', 'OHG', 'PLC', 'PVT', 'SA', 'SARL', 'SCS', 'SL', 'SNC', 'SPA', 'SRL', 'VOF']
  list_legal_types.extend([s.title() for s in list_legal_types])
  list_legal_types.extend([s.lower() for s in list_legal_types])
  list_legal_types.extend(['.'.join([c for c in s]) for s in list_legal_types])
  list_legal_types.extend([f'{s}.' for s in list_legal_types])
  return list_legal_types

# Some stopwords common in organization names in multiple languages.
# TODO: Load from file.
def get_stopwords(language='u'):
#------------------------------
  stopwords = {}
  stopwords['en'] = ['at', 'for', 'with', 'into', 'from', 'in', 'near', 'to', 'the', 'a', 'an', 'of']
  stopwords['es'] = ['a', 'de', 'en', 'por', 'con', 'sin', 'desde', 'hasta', 'sobre', 'bajo', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas']
  stopwords['de'] = ['von', 'zu', 'in', 'auf', 'unter', 'über', 'nach', 'mit', 'für', 'der', 'die', 'das', 'des', 'dem', 'den', 'ein', 'eine', 'eines', 'einer']
  stopwords['it'] = ['di', 'a', 'da', 'in', 'su', 'con', 'per', 'il', 'lo', 'la', 'l\'', 'i', 'gli', 'le', 'l\'', 'un', 'uno', 'una']
  stopwords['fr'] = ['de', 'à', 'dans', 'pour', 'sur', 'sous', 'avec', 'le', 'la', 'les', 'l\'', 'du', 'de la', 'des', 'un', 'une', 'des']
  stopwords['pt'] = ['de', 'em', 'a', 'para', 'com', 'por', 'sobre', 'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'unas']
  stopwords['sv'] = ['på', 'i', 'av', 'till', 'från', 'under', 'över', 'en', 'ett', 'den', 'det', 'de']
  stopwords['no'] = ['på', 'i', 'av', 'til', 'fra', 'under', 'over', 'en', 'et', 'den', 'det', 'de']
  stopwords['fi'] = ['kanssa', 'ilman', 'päällä', 'alaisella', 'alla', 'ympärillä']
  stopwords['nl'] = ['van', 'in', 'op', 'voor', 'naar', 'over', 'onder', 'de', 'het', 'een']
  stopwords['ru'] = ['в', 'на', 'за', 'под', 'о', 'у', 'с']
  stopwords['u'] = []
  for lang in stopwords:
    stopwords['u'].extend(stopwords[lang])
  return set(stopwords[language])






