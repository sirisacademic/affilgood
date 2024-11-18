# normalizer.py
import csv
import os
import re
from langdetect import detect
import translators as ts
import country_converter as coco
import logging
import time
from geopy.geocoders import Nominatim
from tqdm.notebook import tqdm

SUPPORTED_LANGUAGES = [
    'af', 'am', 'ar', 'as', 'az', 'ba', 'bg', 'bho', 'bn', 'bo', 'brx', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'doi', 
    'dsb', 'dv', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fj', 'fo', 'fr', 'fr-CA', 'ga', 'gl', 'gom', 
    'gu', 'ha', 'he', 'hi', 'hne', 'hr', 'hsb', 'ht', 'hu', 'hy', 'id', 'ig', 'ikt', 'is', 'it', 'iu', 'iu-Latn', 
    'ja', 'ka', 'kk', 'km', 'kmr', 'kn', 'ko', 'ks', 'ku', 'ky', 'ln', 'lo', 'lt', 'lug', 'lv', 'lzh', 'mai', 
    'mg', 'mi', 'mk', 'ml', 'mn-Cyrl', 'mn-Mong', 'mr', 'ms', 'mt', 'mww', 'my', 'nb', 'ne', 'nl', 'nso', 'nya', 
    'or', 'otq', 'pa', 'pl', 'prs', 'ps', 'pt', 'pt-PT', 'ro', 'ru', 'run', 'rw', 'sd', 'si', 'sk', 'sl', 'sm', 
    'sn', 'so', 'sq', 'sr-Cyrl', 'sr-Latn', 'st', 'sv', 'sw', 'ta', 'te', 'th', 'ti', 'tk', 'tlh-Latn', 'tn', 
    'to', 'tt', 'ty', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yo', 'yua', 'yue', 'zh-Hans', 'zh-Hant', 'zu'
]

# Configure logging
coco_logger = coco.logging.getLogger()
coco_logger.setLevel(logging.CRITICAL)

class GeoNormalizer:
    def __init__(self, cache_fname='affilgood/metadata_normalization/cache.csv'):
        self.country_to_iso_mapping, self.country_codes = self.load_country_mappings()
        self.convert_cache = {}
        self.cache_fname = 'affilgood/metadata_normalization/cache.csv'
        self.cache = self.load_cache(cache_fname)
        self.app = Nominatim(user_agent="siris_app")
        coco_logger = coco.logging.getLogger()
        coco_logger.setLevel(logging.CRITICAL)

    def load_country_mappings(self):
        """
        Load mappings from a CSV file into a dictionary.
        """
        mappings = {}
        try:
            with open('affilgood/metadata_normalization/openrefine-countries-normalized.csv', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    iso_alpha_3 = row["ISO 3166-1 alpha-3 code"]
                    iso_alpha_2 = row.get("ISO 3166-1 alpha-2 code", "")
                    country_original = row["country_original"]
                    country_official = row["COUNTRY"]
                    mappings.update({
                        iso_alpha_3: country_official,
                        iso_alpha_2: country_official,
                        country_original: country_official,
                        country_official: country_official,
                    })

            with open("affilgood/metadata_normalization/country_data.tsv", 'r') as file:
                reader = csv.DictReader(file, delimiter='\t')
                country_codes = {row["name_short"]:row["ISO3"] for row in reader}

        except FileNotFoundError:
            logging.error("Country normalization data file not found.")
            raise
        return mappings, country_codes

    def clean_string(self, search_str):
        """
        Cleans and formats a string by removing unwanted characters and title-casing it.
        """
        search_str = search_str.title()
        search_str = ', '.join([re.sub(r'[\!\(\)\-\[\]\{\}\;\:\'\"\\\,\<\>\.\/\?\@\#\$\%\^\&\*\_\~\·]+', '', x).strip() for x in search_str.split(',') if not x.isspace() and x != ''])
        return search_str

    def trans_country(self, x):
        """
        Translates country names.
        """
        if type(x) == str and len(x) > 2:

            or_lang = detect(x.title())
            if or_lang != "EN":
                if or_lang in SUPPORTED_LANGUAGES:
                    return ts.translate_text(x, from_language=or_lang, to_language="en") # , translator='argos'
                else:
                    return ts.translate_text(x, or_lang= "en", target_lang="EN") # , translator='argos'
            else:
                return x

    def convert(self,country):
        """
        Converts a country name to a standard short name using caching and translation.
        """
        country = self.clean_string(country)
        if country in self.convert_cache:
            return self.convert_cache[country]

        translated_country = self.trans_country(country)
        converted_country = coco.convert(names=translated_country, to='name_short')
        result = country if converted_country == "not found" or not isinstance(converted_country, str) else converted_country

        self.convert_cache[country] = result
        return result

    def normalize_country(self, countries):
        """
        Normalizes a list of country names, returning a list of tuples with standardized country names and ISO codes.
        
        Parameters:
        - countries (list of str): List of country names to be normalized.

        Returns:
        - list of tuple: List of tuples where each tuple contains the normalized country name and ISO code.
        """
        # Return None if the input is None
        if countries is None:
            return None

        normalized_results = []
        
        for country in countries:
            if country in self.country_to_iso_mapping:
                normalized_country = self.country_to_iso_mapping.get(country.title(), '')
            else:
                # Process the entry if not in the mapping
                normalized_country = self.convert(country)
                
            # Look up the ISO code for the normalized country
            country_norm = normalized_country
            country_code = self.country_codes.get(normalized_country, '')

            # Add the result as a tuple
            normalized_results.append((country_norm, country_code))
        
        return normalized_results
    
    def create_search_string(self,entities):
        """
        Creates an OSM search string for an affiliation based on city, region, and country information.
        Extracts values from a dictionary with keys like 'CITY', 'REGION', and 'COUNTRY'.
        
        Parameters:
        - entities (dict): Dictionary with keys such as 'CITY', 'REGION', and 'COUNTRY'.
        
        Returns:
        - str: A formatted search string based on available location data.
        """
        # Extract values from the dictionary, using the first element if available
        city = entities.get("CITY", [None])[0]
        country = entities.get("COUNTRY", [None])[0]
        region = entities.get("REGION", [None])[0]

        # If there's no city information, return an empty string
        if not city:
            return None

        # Join parts based on their availability
        search_parts = [city]
        if region:
            search_parts.append(region)
        if country:
            search_parts.append(country)
    
        return self.clean_string(", ".join(search_parts)).replace(', ,',',')
    
    def load_cache(self,fname):
        if os.path.isfile(fname):
            with open(fname, mode='r') as infile:
                reader = csv.reader(infile)
                mydict = {rows[0]:rows[0:10] for rows in reader}
            return mydict
        else:
            mydict = {}

            with open(self.cache_fname, mode='w', newline='') as file:
                pass  # No content is written
            return mydict
        
    def fetch_data(self, search_string, attempt, app):
        """
        It makes a single call to OSM API.
        If call returns error status code, it tries again (up to MAX_NUM_ATTEMPTS times).
        Each time it waits an additional second.
        If all attempts fail, returns an exception and whole pipeline halts.
        """
        MAX_NUM_ATTEMPTS = 5
        
        try:
            res = app.geocode(search_string, addressdetails=True, language ="en", featuretype='city')
            if res is None:
                res = app.geocode(search_string, addressdetails=True, language ="en")
            return res
        except Exception:
            print("Retrying geocoding. Attempt number {}.".format(attempt))
            if attempt <= MAX_NUM_ATTEMPTS:
                time.sleep(2**attempt)
                self.fetch_data(search_string, attempt + 1, app)
            else:
                print("TIME OUT")
                raise Exception(res)
            
    def get_geocoding_ents(self, search_string, app):
        """
        Makes single call to OSM and returns fields of interest.
        """
        if search_string is not None and search_string != "":
            data = self.fetch_data(search_string, 1, app)
            if data is None:
                return (search_string, None, None, None, None, None, None, None, None, None)
            # this is for when adresstype="place" and "place is not an address field"
            if data.raw is not None and data.raw['addresstype'] in data.raw["address"]:
                address = data.raw["address"]
                city  = next((data.raw['address'].get(key) for key in ["city", "village", "municipality","town"] if data.raw['address'].get(key) is not None),None)
                state_district = address.get('state_district', None)
                county = address.get('county', None)
                province = address.get('province', None)
                state = address.get('state', None)
                region = address.get('region', None)
                country = address.get('country', None)
                latlng = (data.raw["lat"], data.raw["lon"])
                return search_string, city, state_district, county, province, state, region, country, latlng, data.raw["osm_id"]
            else:
                return (search_string, None, None, None, None, None, None, None, None, None)
        else:
            return (search_string, None, None, None, None, None, None, None, None, None)

        
    def get_cached_and_new_searches(self, search_queries):

        # read cacha here again
        self.cache = self.load_cache(self.cache_fname)

        cached_results_list = [self.cache[k] for k in search_queries if k in self.cache]
        # cached_results = pd.DataFrame(cached_results_list, columns=["search_string",
        #                                                         "osm_city",
        #                                                         "osm_state_district",
        #                                                         "osm_county",
        #                                                         "osm_province",
        #                                                         "osm_state",
        #                                                         "osm_region",
        #                                                         "osm_country",
        #                                                         "osm_lat_long",
        #                                                         "osm_id"])
        new_searches = list({k for k in search_queries if k is not None and k not in self.cache})

        new_results = []
        for query in tqdm(new_searches, desc='Processing new queries of OSM'):
            try:
                time.sleep(1)
                res = self.get_geocoding_ents(query, self.app)
                new_results.append(res)
            except:
                pass
        results = cached_results_list+new_results

        # add to cache
        self.cache.update({rows[0]: rows[:10] for rows in new_results if rows[0] not in self.cache})

        with open(self.cache_fname, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(new_results)

        return results
    
    def transform_osm_results_to_metadata(self, result):
        # Corresponding keys
        keys = ["CITY", "STATE_DISTRICT", "COUNTY", "PROVINCE", "STATE", "REGION", "COUNTRY", "COORDS", "OSM_ID"]

        # Create dictionary with non-None values only
        result = {key: value for key, value in zip(keys, result) if value not in (None, '')}

        return result
            
    def normalize(self, text_list):
        """
        Process a list of text data for span identification.

        Parameters:
        - text_list (list of dict): List of dictionaries, each containing "raw_text" and "span_entities".

        Returns:
        - List of dicts: Each dict contains the original text, span entities, and ner entities grouped by entity group for each span entity.
        """
        # Flatten all span_entities into a single list for batch processing
        ner_to_process = []
        ner_index_map = []  # Track the position in text_list and span_entities index

        for idx, item in enumerate(text_list):
            ner_entities = item.get("ner", [])
            for ner_idx, ner in enumerate(ner_entities):
                # Clean and optionally apply title case to the span text
                ner_to_process.append(ner)
                ner_index_map.append((idx, ner_idx))  # Record which item and span this belongs to

        # Run the osm identification
        outputs = [self.normalize_country(entities.get('COUNTRY',None)) for entities in ner_to_process]

        # Replace COUNTRY extracted in the NER dict...

        # create search strings

        # Initialize the results structure for each item in text_list
        results = [{"raw_text": item["raw_text"], "span_entities": item["span_entities"], "ner": item["ner"],"ner_raw": item["ner_raw"], "osm":[]} for item in text_list]

        for idx, ner in enumerate(ner_to_process):
            # Map each output back to the corresponding text_list item and span_entities index
            entities = outputs[idx]

            # Append ner entities for the current span to the correct entry in results
            item_idx, ner_idx = ner_index_map[idx]
            # Ensure that each item in "ner" corresponds to each span in "span_entities"
            if len(results[item_idx]["osm"]) <= ner_idx:
                results[item_idx]["osm"].append({})
            results[item_idx]["osm"][ner_idx] = entities

        # create search strings
        search_queries = [self.create_search_string(entity) for entity in ner_to_process]

        osm_results = self.get_cached_and_new_searches(search_queries)

        osm_dict = {item[0]: list(item[1:]) for item in osm_results}
        osm_entity_results = [self.transform_osm_results_to_metadata(osm_dict[query]) if query is not None else None for query in search_queries]

        for idx, ner in enumerate(ner_to_process):
            # Map each output back to the corresponding text_list item and span_entities index
            entities = osm_entity_results[idx]

            # Append ner entities for the current span to the correct entry in results
            item_idx, ner_idx = ner_index_map[idx]
            # Ensure that each item in "ner" corresponds to each span in "span_entities"
            if len(results[item_idx]["osm"]) <= ner_idx:
                results[item_idx]["osm"].append({})
            results[item_idx]["osm"][ner_idx] = entities

        return results