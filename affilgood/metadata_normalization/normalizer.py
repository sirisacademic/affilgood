# normalizer.py
import csv
from langdetect import detect
import translators as ts
import country_converter as coco
import logging
import pandas as pd
import os.path
import re

coco_logger = coco.logging.getLogger()
coco_logger.setLevel(logging.CRITICAL)

def trans_country(x):
    """
    Translates country names.
    """
    if type(x) == str and len(x) > 2:
        try:
            or_lang = detect(x.title())
            if or_lang != "en":
                return ts.translate_text(x, from_language=or_lang, to_language="en", translator='argos')
            else:
                return x
        except Exception as e:
            # print("An error occurred: ", e)
            return x
    return x
def clean_string(search_str):
    search_str = search_str.title()
    search_str = ', '.join([re.sub(r'[\!\(\)\-\[\]\{\}\;\:\'\"\\\,\<\>\.\/\?\@\#\$\%\^\&\*\_\~\·]+', '', x).strip() for x in search_str.split(',') if not x.isspace() and x != ''])
    return search_str

def create_search_string(ag_city, ag_region, ror_city, country_norm):
    """
    For a given affiliation, creates its corresponding OSM search string.
    It uses affilgood or ror city and (normalised) country.
    If the country is not available, it uses the affilgood region.
    If country and region are not available, it uses just the city.
    If city is not available, it returns an empty string.
    """
    country = country_norm
    if type(ror_city) == str and ror_city != "":
        city = ror_city
        region = None
    else:
        city = ag_city
        region = ag_region

    if type(city) == str and city != '':
        if "|" in city:
            city = ", ".join(city.split("|"))
        if type(country) == str and country != '':
            if "|" in country:
                country = ", ".join(country.split("|"))
            search_str = ", ".join([city, country])
        elif type(region) == str and region != '':
            if "|" in region:
                region = ", ".join(region.split("|"))
            search_str = ", ".join([city, region])
        else:
            search_str = city
        search_str = clean_string(search_str)
        return search_str
    else:
        return ""

class Normalizer:
    def __init__(self, normalization_rules, cache_fname = 'cache.csv'):
        # Load normalization rules directly as a parameter
        self.rules = normalization_rules

        COUNTRY_CODES = []
        COUNTRY_SHORT_NAMES = []

        with open('country_data.tsv', mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                COUNTRY_CODES.append(row['ISO3'])
                COUNTRY_SHORT_NAMES.append(row['name_short'])

    def normalize_metadata(self, entities):
        # Apply normalization rules to linked entities
        normalized_data = [self.apply_rules(entity) for entity in linked_entities]
        return normalized_data
    
    def convert(self, country):
        # Clean and normalize the country name
        country = clean_string(country)

        # Check if the result is already cached
        if country in convert_cache:
            return convert_cache[country]

        # Perform the conversion and cache the result
        res = coco.convert(names=trans_country(country), to='name_short')
        if res == "not found" or not isinstance(res, str):
            convert_cache[country] = country  # Cache original country name if not found
        else:
            convert_cache[country] = res  # Cache the converted result

        return convert_cache[country]
    
    def normalize_country(item):
        """
        Normalizes a single item's country using a simplified approach without pandas.
        """
        country = item.get("country", "").title()
        # Example normalization; implement your specific normalization logic here
        normalized_country = country if country in COUNTRY_SHORT_NAMES else convert_country(country)
        return normalized_country

    def apply_rules(self, entity):
        # Normalize an individual entity based on rules
        return entity  # Placeholder