# normalizer.py
import os
import csv
import re
from langdetect import detect
import translators as ts
import country_converter as coco
import logging
import time
from geopy.geocoders import Nominatim
from geopy.adapters import AioHTTPAdapter
import requests
import requests_cache

# Define the path to the cache file relative to this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTRY_CACHE_FILE_PATH = os.path.join(CURRENT_DIR, 'country_cache.csv')

# Set up requests cache for HTTP requests
OSM_CACHE_PATH = os.path.join(CURRENT_DIR, 'osm_http_cache')
OSM_CACHE_EXPIRATION = 604800  # Cache for 7 days

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

# Configure logging - suppress country_converter warnings
logging.getLogger('country_converter').setLevel(logging.CRITICAL)

class GeoNormalizer:
    def __init__(self, use_country_cache=True, use_osm_cache=True, country_cache_fname=None, osm_cache_path=None, osm_cache_expiration=None):
        """
        Initialize the GeoNormalizer with configurable caching mechanisms.
        
        Args:
            use_country_cache: Whether to use local file-based caching for country normalization (default: True)
            use_osm_cache: Whether to use HTTP request caching for OSM geocoding (default: True)
            country_cache_fname: Custom path for country cache file (default: COUNTRY_CACHE_FILE_PATH)
            osm_cache_path: Custom path for OSM HTTP cache (default: OSM_CACHE_PATH)
            osm_cache_expiration: Custom expiration time for OSM cache in seconds (default: 604800 - 7 days)
        """
        self.country_to_iso_mapping, self.country_codes = self.load_country_mappings()
        self.convert_cache = {}
        self.trans_cache = {}
        
        # Country cache configuration
        self.use_country_cache = use_country_cache
        self.country_cache_fname = country_cache_fname if country_cache_fname else COUNTRY_CACHE_FILE_PATH
        
        if self.use_country_cache:
            # Create directory for country cache if it doesn't exist
            os.makedirs(os.path.dirname(self.country_cache_fname), exist_ok=True)
            self.country_cache = self.load_country_cache(self.country_cache_fname)
        else:
            self.country_cache = {}
            if logging.getLogger().level <= logging.INFO:
                print(f"Country cache is disabled")
        
        # OSM HTTP cache configuration
        self.use_osm_cache = use_osm_cache
        self.osm_cache_path = osm_cache_path if osm_cache_path else OSM_CACHE_PATH
        self.osm_cache_expiration = osm_cache_expiration if osm_cache_expiration else OSM_CACHE_EXPIRATION
        
        # Initialize Nominatim with requests_cache session if OSM caching is enabled
        self.app = Nominatim(user_agent="siris_app")
        
        if self.use_osm_cache:
            try:
                # Create a cached session
                cached_session = requests_cache.CachedSession(
                    cache_name=self.osm_cache_path,
                    backend='sqlite',
                    expire_after=self.osm_cache_expiration
                )
                self.app.session = cached_session
                if logging.getLogger().level <= logging.INFO:
                    print(f"OSM HTTP cache initialized at {self.osm_cache_path}")
            except Exception as e:
                print(f"Warning: Failed to initialize requests_cache for OSM: {e}")
                print("Continuing without OSM HTTP caching")
                self.app.session = None
        else:
            self.app.session = None
            if logging.getLogger().level <= logging.INFO:
                print(f"OSM HTTP cache is disabled")
   
    def load_country_mappings(self):
        """
        Load mappings and country codes from available CSV/TSV files.
        """
        mappings = {}
        country_codes = {}
        
        # Path to the directory containing the data files
        data_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            # Path to openrefine-countries-normalized.csv
            openrefine_path = os.path.join(data_dir, 'openrefine-countries-normalized.csv')
            
            # Load mappings from `openrefine-countries-normalized.csv` if it exists.
            if os.path.exists(openrefine_path):
                with open(openrefine_path, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        iso_alpha_3 = row["ISO 3166-1 alpha-3 code"]
                        iso_alpha_2 = row.get("ISO 3166-1 alpha-2 code", "")
                        country_original = row["country_original"]
                        country_official = row["COUNTRY"]
                        # Populate mappings from this file.
                        mappings.update({
                            iso_alpha_3: country_official,
                            iso_alpha_2: country_official,
                            country_original: country_official,
                            country_official: country_official,
                        })
            else:
                logging.warning(f"File not found: {openrefine_path}. Using country_data.tsv for mappings.")
                
            # Path to country_data.tsv
            country_data_path = os.path.join(data_dir, 'country_data.tsv')
            
            # Load `country_data.tsv` to ensure mappings and country codes.
            if os.path.exists(country_data_path):
                with open(country_data_path, 'r') as file:
                    reader = csv.DictReader(file, delimiter='\t')
                    for row in reader:
                        name_short = row["name_short"]
                        iso_alpha_3 = row["ISO3"]
                        iso_alpha_2 = row.get("ISO2", "")
                        name_official = row["name_official"]
                        # Populate mappings only if the key is not already present
                        mappings.setdefault(name_short, name_official)
                        mappings.setdefault(name_official, name_official)
                        mappings.setdefault(iso_alpha_3, name_official)
                        mappings.setdefault(iso_alpha_2, name_official)
                        # Populate country_codes.
                        if name_short not in country_codes:
                            country_codes[name_short] = iso_alpha_3
            else:
                raise FileNotFoundError(f"File not found: {country_data_path}. Cannot load country mappings or country codes.")
        except FileNotFoundError as e:
            logging.error(f"Country normalization data file not found: {e}")
            raise
            
        return mappings, country_codes

    def clean_string(self, search_str):
        """
        Cleans and formats a string by removing unwanted characters and title-casing it.
        """
        search_str = search_str.title()
        search_str = ', '.join([re.sub(r'[\!\(\)\-\[\]\{\}\;\:\'\"\\\,\<\>\.\/\?\@\#\$\%\^\&\*\_\~\·]+', '', x).strip() for x in search_str.split(',') if not x.isspace() and x != ''])
        return search_str

    def trans_country(self, country_name):
        """
        Translates country names from their original language to English.
        Uses a cache to avoid repeated translations of the same text.
        
        Parameters:
        - country_name (str): Country name to translate
        
        Returns:
        - str: Translated country name or original string if translation not needed/possible
        """
        # Return early if input is not a suitable string
        if not isinstance(country_name, str) or len(country_name) <= 2:
            return country_name
        
        # Check if we already have this translation in cache
        if country_name in self.trans_cache:
            return self.trans_cache[country_name]
        
        try:
            # Detect language
            original_language = detect(country_name.title())
            
            # Only translate if not already in English
            if original_language.lower() != "en":
                try:
                    if original_language in SUPPORTED_LANGUAGES:
                        # Use detected language for translation
                        translated_text = ts.translate_text(country_name, from_language=original_language, to_language="en")
                    else:
                        # If language not supported, try auto-detection
                        translated_text = ts.translate_text(country_name, to_language="en")
                    
                    # Store in cache and return
                    self.trans_cache[country_name] = translated_text
                    return translated_text
                except Exception as e:
                    # Log translation error and return original
                    # print(f"Translation error for '{country_name}': {str(e)}")
                    self.trans_cache[country_name] = country_name  # Cache the original to avoid retrying failed translations
                    return country_name
            
            # English text doesn't need translation
            self.trans_cache[country_name] = country_name
            return country_name
        except Exception as e:
            # Log language detection error and return original
            print(f"Language detection error for '{country_name}': {str(e)}")
            self.trans_cache[country_name] = country_name  # Cache the original to avoid retrying failed detections
            return country_name

    def convert(self,country):
        """
        Converts a country name to a standard short name using caching and translation.
        """
        country = self.clean_string(country)
        if country in self.convert_cache:
            return self.convert_cache[country]

        if country:
            translated_country = self.trans_country(country)
            converted_country = coco.convert(names=translated_country, to='name_short')
            result = country if converted_country == "not found" or not isinstance(converted_country, str) else converted_country
        else:
            result = ""
            
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
        if countries is None or len(countries) == 0:
            return []

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
    
    def create_search_string(self, entities):
        """
        Creates an OSM search string for an affiliation, tagging feature types.

        Parameters:
        - entities (dict): Dictionary with keys such as 'CITY', 'REGION', and 'COUNTRY' - or 'ORG'.

        Returns:
        - str: A formatted search string prefixed with feature type (e.g., "country:Austria").
        """
        # Extract values from the dictionary, using the first element if available
        org = entities.get("ORG", [None])[0] if entities.get("ORG") else ""
        city = entities.get("CITY", [None])[0] if entities.get("CITY") else ""
        country = entities.get("COUNTRY", [None])[0] if entities.get("COUNTRY") else ""
        region = entities.get("REGION", [None])[0] if entities.get("REGION") else ""

        # Default feature type.
        feature_type = "settlement"

        # If we have a city, use it as the primary search component
        if city:
            feature_type = "city"
            search_parts = [city]
            if region:
                search_parts.append(region)
            if country:
                search_parts.append(country)
        # If no city but we have region, use region as primary search component
        elif region:
            feature_type = "state"
            search_parts = [region]
            if country:
                search_parts.append(country)
        # If only country, decide if we want to use it (could be too broad)
        elif country:
            feature_type = "country"
            search_parts = [country]
        else:
            # No location information available
            return None

        return f"{feature_type}:" + self.clean_string(", ".join(search_parts)).replace(', ,',',')
        
    def load_country_cache(self, fname):
        """
        Load country normalization cache from file.
        
        Args:
            fname: Path to the cache file
            
        Returns:
            Dictionary with cached country normalization data
        """
        if not self.use_country_cache:
            return {}
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        
        if os.path.isfile(fname):
            try:
                with open(fname, mode='r') as infile:
                    reader = csv.reader(infile)
                    mydict = {rows[0]:rows[0:10] for rows in reader}
                
                if logging.getLogger().level <= logging.INFO:
                    print(f"Loaded {len(mydict)} entries from country cache: {fname}")
                    
                return mydict
            except Exception as e:
                print(f"Error loading country cache: {e}")
                # Create an empty file if there was an error
                with open(fname, mode='w', newline='') as file:
                    pass
                return {}
        else:
            # Create an empty file if it doesn't exist
            with open(fname, mode='w', newline='') as file:
                pass
            return {}
     
    def fetch_data(self, search_string, attempt, app):
        """
        It makes a single call to OSM API.
        If call returns error status code, it tries again (up to MAX_NUM_ATTEMPTS times).
        Each time it waits an additional second.
        If all attempts fail, returns an exception and whole pipeline halts.
        """
        MAX_NUM_ATTEMPTS = 5
        
        try:
            featuretype, _, location_query = search_string.partition(":")
            res = app.geocode(location_query, addressdetails=True, language ="en", featuretype=featuretype)
            if res is None:
                res = app.geocode(location_query, addressdetails=True, language ="en")
            return res
        except Exception:
            print("Retrying geocoding. Attempt number {}.".format(attempt))
            if attempt <= MAX_NUM_ATTEMPTS:
                time.sleep(2**attempt)
                self.fetch_data(location_query, attempt + 1, app)
            else:
                print("TIME OUT")
                raise Exception("Geocoding timeout")

    def get_geocoding_ents(self, search_string, app):
        """
        Makes single call to OSM and returns fields of interest.
        Uses caching if enabled.
        """
        if search_string is not None and search_string != "":
            # Session is already configured based on use_osm_cache in __init__
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
        """
        Get results from cache and perform new searches for uncached queries.
        
        Args:
            search_queries: List of search queries
            
        Returns:
            List of results for each query
        """
        # Read country cache again if country caching is enabled
        if self.use_country_cache:
            self.country_cache = self.load_country_cache(self.country_cache_fname)

        # Get results from country cache for cached queries
        cached_results_list = [
            self.country_cache[k] for k in search_queries 
            if k is not None and k in self.country_cache and self.use_country_cache
        ]
        
        # Identify queries that need new searches
        new_searches = list({
            k for k in search_queries 
            if k is not None and (not self.use_country_cache or k not in self.country_cache)
        })

        # Process new searches
        new_results = []
        for query in new_searches:
            try:
                time.sleep(1)
                res = self.get_geocoding_ents(query, self.app)
                new_results.append(res)
            except Exception as e:
                # Add error handling for failed geocoding
                print(f"Failed to geocode '{query}': {str(e)}")
                # Add a placeholder result for the failed query
                new_results.append((query, None, None, None, None, None, None, None, None, None))
                
        # Combine cached and new results
        results = cached_results_list + new_results

        # Add new results to country cache if country caching is enabled
        if self.use_country_cache:
            self.country_cache.update({rows[0]: rows[:10] for rows in new_results if rows[0] not in self.country_cache})
            
            # Write new results to country cache file
            try:
                with open(self.country_cache_fname, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(new_results)
            except Exception as e:
                print(f"Error updating country cache file: {e}")

        return results

    
    def transform_osm_results_to_metadata(self, result):
        # Corresponding keys
        keys = ["CITY", "STATE_DISTRICT", "COUNTY", "PROVINCE", "STATE", "REGION", "COUNTRY", "COORDS", "OSM_ID"]

        # Create dictionary with non-None values only
        result = {key: value for key, value in zip(keys, result) if value not in (None, '')}

        return result
            
    def normalize(self, entity_list):
        """
        Process a list of text data for span identification.

        Parameters:
        - entity_list (list of dict): List of dictionaries, each containing "raw_text" and "span_entities".

        Returns:
        - List of dicts: Each dict contains the original text, span entities, and ner entities grouped by entity group for each span entity.
        """
        # Flatten all span_entities into a single list for batch processing
        ner_to_process = []
        ner_index_map = []  # Track the position in entity_list and span_entities index

        for idx, item in enumerate(entity_list):
            ner_entities = item.get("ner", [])
            for ner_idx, ner in enumerate(ner_entities):
                # Clean and optionally apply title case to the span text
                ner_to_process.append(ner)
                ner_index_map.append((idx, ner_idx))  # Record which item and span this belongs to

        # Run the osm identification
        outputs = [self.normalize_country(entities.get('COUNTRY',None)) for entities in ner_to_process]

        # Initialize the results structure for each item in entity_list
        results = [{"raw_text": item["raw_text"], "span_entities": item["span_entities"], "ner": item["ner"],"ner_raw": item["ner_raw"], "osm":[]} for item in entity_list]

        for idx, ner in enumerate(ner_to_process):
            # Map each output back to the corresponding entity_list item and span_entities index
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

        osm_dict = {item[0]: list(item[1:]) for item in osm_results if item[0] is not None}
        osm_entity_results = []

        for query in search_queries:
            if query is None:
                osm_entity_results.append(None)
            else:
                try:
                    if query in osm_dict:
                        result = osm_dict[query]
                        osm_entity_results.append(self.transform_osm_results_to_metadata(result))
                    else:
                        # Handle missing queries
                        print(f"Warning: No OSM result found for query '{query}'")
                        osm_entity_results.append(None)
                except Exception as e:
                    print(f"Error processing OSM result for query '{query}': {str(e)}")
                    osm_entity_results.append(None)

        for idx, ner in enumerate(ner_to_process):
            # Map each output back to the corresponding entity_list item and span_entities index
            entities = osm_entity_results[idx]

            # Append ner entities for the current span to the correct entry in results
            item_idx, ner_idx = ner_index_map[idx]
            # Ensure that each item in "ner" corresponds to each span in "span_entities"
            if len(results[item_idx]["osm"]) <= ner_idx:
                results[item_idx]["osm"].append({})
            results[item_idx]["osm"][ner_idx] = entities

        return results
    
    def set_country_cache_enabled(self, enabled=True):
        """
        Enable or disable country caching.
        
        Args:
            enabled: True to enable caching, False to disable
            
        Returns:
            dict: Status of the operation
        """
        old_status = self.use_country_cache
        self.use_country_cache = enabled
        
        if enabled and not old_status:
            # Load cache if enabling
            self.country_cache = self.load_country_cache(self.country_cache_fname)
            return {
                "status": "Country cache enabled",
                "cache_file": self.country_cache_fname,
                "entries": len(self.country_cache)
            }
        elif not enabled and old_status:
            # Clear cache if disabling
            cache_size = len(self.country_cache)
            self.country_cache = {}
            return {
                "status": "Country cache disabled",
                "cache_entries_cleared": cache_size
            }
        
        return {
            "status": f"Country cache already {'enabled' if enabled else 'disabled'}"
        }
    
    def set_osm_cache_enabled(self, enabled=True):
        """
        Enable or disable OSM HTTP request caching.
        
        Args:
            enabled: True to enable caching, False to disable
            
        Returns:
            dict: Status of the operation
        """
        old_status = self.use_osm_cache
        self.use_osm_cache = enabled
        
        if enabled and not old_status:
            # Initialize cached session
            try:
                cached_session = requests_cache.CachedSession(
                    cache_name=self.osm_cache_path,
                    backend='sqlite',
                    expire_after=self.osm_cache_expiration
                )
                self.app.session = cached_session
                return {
                    "status": "OSM cache enabled",
                    "cache_path": self.osm_cache_path
                }
            except Exception as e:
                self.use_osm_cache = False
                return {
                    "status": f"Failed to enable OSM cache: {str(e)}"
                }
        elif not enabled and old_status:
            # Remove cached session
            self.app.session = None
            return {
                "status": "OSM cache disabled"
            }
        
        return {
            "status": f"OSM cache already {'enabled' if enabled else 'disabled'}"
        }
    
    def get_cache_stats(self):
        """
        Get statistics about both caching mechanisms.
        
        Returns:
            dict: Statistics about both caches
        """
        stats = {
            "country_cache": {
                "enabled": self.use_country_cache,
                "file_path": self.country_cache_fname if self.use_country_cache else None,
                "entries": len(self.country_cache) if self.use_country_cache else 0
            },
            "osm_cache": {
                "enabled": self.use_osm_cache,
                "path": self.osm_cache_path if self.use_osm_cache else None,
                "expiration_days": self.osm_cache_expiration / (60 * 60 * 24) if self.use_osm_cache else None
            }
        }
        
        # Add country cache file size if available
        if self.use_country_cache and os.path.exists(self.country_cache_fname):
            stats["country_cache"]["file_size_kb"] = os.path.getsize(self.country_cache_fname) / 1024
        
        # Add OSM cache stats if available
        if self.use_osm_cache:
            cache_db = f"{self.osm_cache_path}.sqlite"
            if os.path.exists(cache_db):
                stats["osm_cache"]["file_size_mb"] = os.path.getsize(cache_db) / (1024 * 1024)
                
                # Try to get count of cached responses if sqlite3 is available
                try:
                    import sqlite3
                    conn = sqlite3.connect(cache_db)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM responses")
                    count = cursor.fetchone()[0]
                    conn.close()
                    stats["osm_cache"]["cached_responses"] = count
                except:
                    pass
        
        return stats
    
    def clear_country_cache(self):
        """
        Clear the country cache.
        
        Returns:
            dict: Status of the operation
        """
        if not self.use_country_cache:
            return {"status": "Country cache is disabled"}
            
        cache_size = len(self.country_cache)
        self.country_cache = {}
        
        # Truncate the cache file
        try:
            with open(self.country_cache_fname, 'w', newline='') as f:
                pass
        except Exception as e:
            return {
                "status": f"Error clearing country cache file: {str(e)}",
                "memory_cache_cleared": True,
                "entries_cleared": cache_size
            }
            
        return {
            "status": "Country cache cleared",
            "entries_cleared": cache_size
        }
    
    def clear_osm_cache(self):
        """
        Clear the OSM HTTP request cache.
        
        Returns:
            dict: Status of the operation
        """
        if not self.use_osm_cache:
            return {"status": "OSM cache is disabled"}
            
        # Try to clear the cache through the requests_cache API
        try:
            if hasattr(self.app.session, 'cache'):
                self.app.session.cache.clear()
                return {"status": "OSM cache cleared"}
            else:
                # Recreate the session as a way to clear the cache
                cached_session = requests_cache.CachedSession(
                    cache_name=self.osm_cache_path,
                    backend='sqlite',
                    expire_after=self.osm_cache_expiration
                )
                self.app.session = cached_session
                return {"status": "OSM cache recreated"}
        except Exception as e:
            return {"status": f"Error clearing OSM cache: {str(e)}"}
