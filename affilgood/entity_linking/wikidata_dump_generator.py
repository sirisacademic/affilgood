from typing import List, Dict, Optional, Union
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
import pycountry
import requests_cache
import json
import logging
import time
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_linking.constants import WIKIDATA_ORG_TYPES_SHORT, WIKIDATA_ORG_TYPES_EXTENDED, COUNTRY_LANGS_FILE

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
EMAIL_WIKIDATA_ENDPOINT = "sirislab@sirisacademic.com"
TIMEOUT_COUNT_REQUEST = 60
TIMEOUT_BATCH_REQUEST = 120
DELAY_BASE_RETRY = 10
MAX_NUM_RETRIES = 10
MAX_NUM_BATCHES = 25
BATCH_SIZE = 50

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class WikiDataCache:
    """Cache for WikiData SPARQL query results using Parquet format."""
    
    def __init__(self, cache_dir=None, cache_expiration_days=30):
        """
        Initialize the WikiData cache.
        
        Args:
            cache_dir: Directory to store cache files. If None, uses a default
                       directory in the user's home folder.
            cache_expiration_days: Number of days until cache entries expire.
        """
        try:
            if cache_dir is None:
                cache_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'wikidata_cache'
                )
            
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_expiration = timedelta(days=cache_expiration_days)
            
            # Create subdirectories for different query types
            self.count_cache_dir = self.cache_dir / "count_queries"
            self.batch_cache_dir = self.cache_dir / "batch_queries"
            self.dataset_cache_dir = self.cache_dir / "complete_datasets"
            
            # Create directories with fallback to string paths
            for cache_subdir in [self.count_cache_dir, self.batch_cache_dir, self.dataset_cache_dir]:
                try:
                    cache_subdir.mkdir(exist_ok=True)
                except Exception:
                    # Fallback to string paths if pathlib has issues
                    cache_subdir = os.path.join(cache_dir, cache_subdir.name)
                    os.makedirs(cache_subdir, exist_ok=True)
            
            # Cache statistics
            self.hits = 0
            self.misses = 0
            self.saved_queries = 0
            
        except Exception as e:
            print(f"Error initializing WikiDataCache: {e}")
            # Fallback to memory-only operation
            self.cache_dir = None
            self.count_cache_dir = None
            self.batch_cache_dir = None
            self.dataset_cache_dir = None
            self.cache_expiration = timedelta(days=cache_expiration_days)
            self.hits = 0
            self.misses = 0
            self.saved_queries = 0
    
    def get_dataset_cache_paths(self, countries, org_types):
        """Get cache file paths for a complete dataset (both parquet and json)."""
        if self.dataset_cache_dir is None:
            return None, None
            
        # Create a consistent cache key from countries and org_types
        if isinstance(countries, list):
            countries_str = "_".join(sorted(countries))
        else:
            countries_str = str(countries)
            
        if isinstance(org_types, list):
            org_types_str = "_".join(sorted(org_types))
        else:
            org_types_str = str(org_types)
        
        cache_key = f"dataset_{countries_str}_{org_types_str}"
        
        # Handle both Path and string paths
        if isinstance(self.dataset_cache_dir, Path):
            parquet_path = self.dataset_cache_dir / f"{cache_key}.parquet"
            json_path = self.dataset_cache_dir / f"{cache_key}.json"
        else:
            parquet_path = os.path.join(self.dataset_cache_dir, f"{cache_key}.parquet")
            json_path = os.path.join(self.dataset_cache_dir, f"{cache_key}.json")
        
        return parquet_path, json_path
    
    def get_cached_dataset(self, countries, org_types):
        """
        Get a cached complete dataset if available and not expired.
        
        Args:
            countries: List of countries or single country
            org_types: List of organization types or single type
            
        Returns:
            DataFrame if cached, None otherwise
        """
        try:
            parquet_path, json_path = self.get_dataset_cache_paths(countries, org_types)
            
            if not parquet_path or not json_path or not os.path.exists(parquet_path) or not os.path.exists(json_path):
                self.misses += 1
                return None
                
            try:
                # Load metadata from JSON
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Check if cache is expired
                timestamp_str = metadata.get('timestamp', '2000-01-01')
                if isinstance(timestamp_str, str):
                    cache_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    cache_time = timestamp_str
                    
                if datetime.now() - cache_time > self.cache_expiration:
                    # Cache expired
                    self.misses += 1
                    return None
                    
                # Load dataset from Parquet
                dataset = pd.read_parquet(parquet_path)
                
                # Valid cache
                self.hits += 1
                return dataset
                
            except Exception as e:
                # If any error occurs, treat as cache miss
                print(f"Error reading dataset cache files {parquet_path}, {json_path}: {e}")
                self.misses += 1
                return None
        except Exception as e:
            print(f"Error in get_cached_dataset: {e}")
            self.misses += 1
            return None
    
    def cache_dataset(self, countries, org_types, dataset):
        """
        Cache a complete dataset using Parquet format.
        
        Args:
            countries: List of countries or single country
            org_types: List of organization types or single type
            dataset: DataFrame to cache
        """
        try:
            if self.dataset_cache_dir is None:
                return
                
            parquet_path, json_path = self.get_dataset_cache_paths(countries, org_types)
            if not parquet_path or not json_path:
                return
                
            # Save dataset as Parquet
            dataset.to_parquet(parquet_path, index=False)
            
            # Save metadata as JSON
            metadata = {
                'countries': countries,
                'org_types': org_types,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.saved_queries += 1
            
        except Exception as e:
            print(f"Error caching dataset: {e}")
    
    def get_batch_cache_paths(self, type_qid, country_qid, limit, offset):
        """Get cache file paths for a batch query (both parquet and json)."""
        if self.batch_cache_dir is None:
            return None, None
            
        cache_key = f"batch_{type_qid}_{country_qid}_{limit}_{offset}"
        
        # Handle both Path and string paths
        if isinstance(self.batch_cache_dir, Path):
            parquet_path = self.batch_cache_dir / f"{cache_key}.parquet"
            json_path = self.batch_cache_dir / f"{cache_key}.json"
        else:
            parquet_path = os.path.join(self.batch_cache_dir, f"{cache_key}.parquet")
            json_path = os.path.join(self.batch_cache_dir, f"{cache_key}.json")
        
        return parquet_path, json_path
    
    def get_cached_batch(self, type_qid, country_qid, limit, offset):
        """
        Get a cached batch result if available and not expired.
        
        Args:
            type_qid: Organization type QID
            country_qid: Country QID
            limit: Batch size limit
            offset: Result offset
            
        Returns:
            List of results if cached, None otherwise
        """
        try:
            parquet_path, json_path = self.get_batch_cache_paths(type_qid, country_qid, limit, offset)
            
            if not parquet_path or not json_path or not os.path.exists(parquet_path) or not os.path.exists(json_path):
                self.misses += 1
                return None
                
            try:
                # Load metadata from JSON
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Check if cache is expired
                timestamp_str = metadata.get('timestamp', '2000-01-01')
                if isinstance(timestamp_str, str):
                    cache_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    cache_time = timestamp_str
                    
                if datetime.now() - cache_time > self.cache_expiration:
                    # Cache expired
                    self.misses += 1
                    return None
                    
                # Load results from Parquet
                df = pd.read_parquet(parquet_path)
                results = df.to_dict('records') if not df.empty else []
                
                # Valid cache
                self.hits += 1
                return results
                
            except Exception as e:
                # If any error occurs, treat as cache miss
                print(f"Error reading batch cache files {parquet_path}, {json_path}: {e}")
                self.misses += 1
                return None
        except Exception as e:
            print(f"Error in get_cached_batch: {e}")
            self.misses += 1
            return None
    
    def cache_batch(self, type_qid, country_qid, limit, offset, results):
        """
        Cache a batch query result using Parquet format.
        
        Args:
            type_qid: Organization type QID
            country_qid: Country QID
            limit: Batch size limit
            offset: Result offset
            results: Result data to cache
        """
        try:
            if self.batch_cache_dir is None:
                return
                
            parquet_path, json_path = self.get_batch_cache_paths(type_qid, country_qid, limit, offset)
            if not parquet_path or not json_path:
                return
                
            # Save results as Parquet
            if results:
                df = pd.DataFrame(results)
                df.to_parquet(parquet_path, index=False)
            else:
                # Create empty parquet file
                pd.DataFrame().to_parquet(parquet_path, index=False)
            
            # Save metadata as JSON
            metadata = {
                'type_qid': type_qid,
                'country_qid': country_qid,
                'limit': limit,
                'offset': offset,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.saved_queries += 1
            
        except Exception as e:
            print(f"Error caching batch: {e}")
    
    def clear_expired_datasets(self):
        """Clear expired dataset cache entries."""
        if self.dataset_cache_dir is None:
            return 0
            
        cleared_count = 0
        try:
            cache_dir = self.dataset_cache_dir
            if isinstance(cache_dir, Path):
                json_files = cache_dir.glob("dataset_*.json")
            else:
                json_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) 
                              if f.startswith("dataset_") and f.endswith(".json")]
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                        
                    timestamp_str = metadata.get('timestamp', '2000-01-01')
                    if isinstance(timestamp_str, str):
                        cache_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        cache_time = timestamp_str
                        
                    if datetime.now() - cache_time > self.cache_expiration:
                        # Remove both JSON and Parquet files
                        os.remove(json_file)
                        parquet_file = Path(json_file).with_suffix('.parquet')
                        if os.path.exists(parquet_file):
                            os.remove(parquet_file)
                        cleared_count += 1
                        
                except Exception as e:
                    print(f"Error checking cache file {json_file}: {e}")
                    
        except Exception as e:
            print(f"Error clearing expired datasets: {e}")
            
        return cleared_count
    
    def clear_expired_caches(self):
        """Clear all expired cache entries to free up disk space."""
        if self.cache_dir is None:
            return {'datasets': 0, 'counts': 0, 'batches': 0}
        
        cleared_counts = {
            'datasets': 0,
            'counts': 0, 
            'batches': 0
        }
        
        try:
            # Clear expired dataset caches
            cleared_counts['datasets'] = self.clear_expired_datasets()
            
            # Clear expired count caches (still JSON)
            if self.count_cache_dir is not None:
                cache_dir = self.count_cache_dir
                if isinstance(cache_dir, Path):
                    cache_files = cache_dir.glob("count_*.json")
                else:
                    cache_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) 
                                  if f.startswith("count_") and f.endswith(".json")]
                
                for cache_file in cache_files:
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                            
                        cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                        if datetime.now() - cache_time > self.cache_expiration:
                            os.remove(cache_file)
                            cleared_counts['counts'] += 1
                            
                    except Exception as e:
                        print(f"Error checking count cache file {cache_file}: {e}")
            
            # Clear expired batch caches
            if self.batch_cache_dir is not None:
                cache_dir = self.batch_cache_dir
                if isinstance(cache_dir, Path):
                    json_files = cache_dir.glob("batch_*.json")
                else:
                    json_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) 
                                  if f.startswith("batch_") and f.endswith(".json")]
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            metadata = json.load(f)
                            
                        timestamp_str = metadata.get('timestamp', '2000-01-01')
                        if isinstance(timestamp_str, str):
                            cache_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            cache_time = timestamp_str
                            
                        if datetime.now() - cache_time > self.cache_expiration:
                            # Remove both JSON and Parquet files
                            os.remove(json_file)
                            parquet_file = Path(json_file).with_suffix('.parquet')
                            if os.path.exists(parquet_file):
                                os.remove(parquet_file)
                            cleared_counts['batches'] += 1
                            
                    except Exception as e:
                        print(f"Error checking batch cache file {json_file}: {e}")
                        
        except Exception as e:
            print(f"Error clearing expired caches: {e}")
        
        return cleared_counts
    
    def get_cache_info(self):
        """Get information about cache usage."""
        if self.cache_dir is None:
            return {
                'cache_enabled': False,
                'cache_dir': None,
                'cache_stats': {'hits': self.hits, 'misses': self.misses, 'saved_queries': self.saved_queries}
            }
        
        cache_info = {
            'cache_enabled': True,
            'cache_dir': str(self.cache_dir),
            'cache_stats': {
                'hits': self.hits, 
                'misses': self.misses, 
                'saved_queries': self.saved_queries
            },
            'cache_files': {
                'datasets': 0,
                'counts': 0,
                'batches': 0
            }
        }
        
        try:
            # Count dataset cache files (parquet files)
            if self.dataset_cache_dir is not None:
                cache_dir = self.dataset_cache_dir
                if isinstance(cache_dir, Path):
                    cache_info['cache_files']['datasets'] = len(list(cache_dir.glob("dataset_*.parquet")))
                else:
                    cache_info['cache_files']['datasets'] = len([f for f in os.listdir(cache_dir) 
                                                                if f.startswith("dataset_") and f.endswith(".parquet")])
            
            # Count count cache files (still JSON)
            if self.count_cache_dir is not None:
                cache_dir = self.count_cache_dir
                if isinstance(cache_dir, Path):
                    cache_info['cache_files']['counts'] = len(list(cache_dir.glob("count_*.json")))
                else:
                    cache_info['cache_files']['counts'] = len([f for f in os.listdir(cache_dir) 
                                                              if f.startswith("count_") and f.endswith(".json")])
            
            # Count batch cache files (parquet files)
            if self.batch_cache_dir is not None:
                cache_dir = self.batch_cache_dir
                if isinstance(cache_dir, Path):
                    cache_info['cache_files']['batches'] = len(list(cache_dir.glob("batch_*.parquet")))
                else:
                    cache_info['cache_files']['batches'] = len([f for f in os.listdir(cache_dir) 
                                                               if f.startswith("batch_") and f.endswith(".parquet")])
                    
        except Exception as e:
            print(f"Error getting cache info: {e}")
        
        return cache_info
    
    # Keep these methods unchanged (they still use JSON for count caching)
    def get_count_cache_path(self, type_qid, country_qid):
        """Get cache file path for a count query."""
        if self.count_cache_dir is None:
            return None
            
        cache_key = f"count_{type_qid}_{country_qid}"
        
        # Handle both Path and string paths
        if isinstance(self.count_cache_dir, Path):
            return self.count_cache_dir / f"{cache_key}.json"
        else:
            return os.path.join(self.count_cache_dir, f"{cache_key}.json")
    
    def get_cached_count(self, type_qid, country_qid):
        """Get a cached count result if available and not expired."""
        try:
            cache_path = self.get_count_cache_path(type_qid, country_qid)
            
            if cache_path is None or not os.path.exists(cache_path):
                self.misses += 1
                return None
                
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check if cache is expired
                cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time > self.cache_expiration:
                    # Cache expired
                    self.misses += 1
                    return None
                    
                # Valid cache
                self.hits += 1
                return cached_data.get('count')
                
            except Exception:
                # If any error occurs, treat as cache miss
                self.misses += 1
                return None
        except Exception as e:
            print(f"Error in get_cached_count: {e}")
            self.misses += 1
            return None
    
    def cache_count(self, type_qid, country_qid, count):
        """Cache a count query result."""
        try:
            if self.count_cache_dir is None:
                return
                
            cache_path = self.get_count_cache_path(type_qid, country_qid)
            if cache_path is None:
                return
                
            cache_data = {
                'type_qid': type_qid,
                'country_qid': country_qid,
                'count': count,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            self.saved_queries += 1
            
        except Exception as e:
            print(f"Error caching count: {e}")
               
class WikidataDumpGenerator:

    def __init__(self, verbose=False):
        self.endpoint_url = WIKIDATA_SPARQL_ENDPOINT
        self.headers = {"User-Agent": f"WikidataDumpGeneratorBot/1.0 ({EMAIL_WIKIDATA_ENDPOINT})"}
        
        # Initialize cache for WikiData queries
        self.cache = WikiDataCache()

        self.verbose = verbose
        if self.verbose:
            logger.info("Initializing WikidataDumpGenerator...")

        # Clean up expired cache entries on startup
        try:
            cleared = self.cache.clear_expired_caches()
            if self.verbose and any(cleared.values()):
                logger.info(f"Cleared expired cache entries: {cleared['datasets']} datasets, "
                           f"{cleared['counts']} counts, {cleared['batches']} batches")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error clearing expired caches: {e}")
        
        # Keep requests_cache for backward compatibility, but we'll primarily use our custom cache
        self.session = requests_cache.CachedSession(
            cache_name='sparql_cache',
            backend='sqlite',
            expire_after=86400
        )
        
        # Load language codes for countries
        lang_codes_df = pd.read_csv(COUNTRY_LANGS_FILE, sep='\t').fillna("")
        self.lang_lookup = lang_codes_df.set_index('country_exonym')['lang_codes'].str.split('|').to_dict()
        
        # Load organization types
        with open(WIKIDATA_ORG_TYPES_SHORT, 'r', encoding='utf-8') as f:
            qid_to_type = json.load(f)
            self.organisation_types_short = {v: k for k, v in qid_to_type.items()}  # Name to QID

        with open(WIKIDATA_ORG_TYPES_EXTENDED, 'r', encoding='utf-8') as f:
            qid_to_type_ext = json.load(f)
            self.organisation_types_extended = {v: k for k, v in qid_to_type_ext.items()}  # Name to QID
            
        # Reverse mappings (QID to name)
        self.country_map = None
        self.org_type_map_short = qid_to_type
        self.org_type_map_extended = qid_to_type_ext
        
        if self.verbose:
            logger.info("Initialization complete.")

    def get_qid_from_iso_code(self, iso_code: str) -> str:
        """
        Query WikiData to get the QID for a given ISO country code.
        
        Args:
            iso_code: 2-letter ISO country code (e.g., 'MD', 'MC', 'RS')
            
        Returns:
            WikiData QID or None if not found
        """
        from SPARQLWrapper import SPARQLWrapper, JSON
        
        try:
            # WikiData property P297 is "ISO 3166-1 alpha-2 code"
            query = f"""
            SELECT ?country WHERE {{
                ?country wdt:P297 "{iso_code.upper()}" .
            }}
            """
            
            sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
            sparql._session = self.session  # Use the same session for caching
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            
            results = sparql.query().convert()
            
            if results['results']['bindings']:
                # Extract QID from the full URI
                country_uri = results['results']['bindings'][0]['country']['value']
                qid = country_uri.split('/')[-1]  # Extract Q123 from http://www.wikidata.org/entity/Q123
                
                if self.verbose:
                    logger.info(f"Resolved ISO code {iso_code} to QID {qid}")
                
                return qid
            else:
                if self.verbose:
                    logger.warning(f"No WikiData entity found for ISO code: {iso_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying WikiData for ISO code {iso_code}: {e}")
            return None


    def get_country_qids_from_wikidata(self):
        """
        Query Wikidata for countries and their QIDs.
        
        Returns:
            Dict mapping country names to QIDs
        """
        if self.country_map is not None:
            return self.country_map
            
        sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
        sparql._session = self.session
        
        query = """
        SELECT ?country ?countryLabel WHERE {
        ?country wdt:P31 wd:Q6256.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        countries = {res["countryLabel"]["value"]: res["country"]["value"].split("/")[-1] 
                    for res in results["results"]["bindings"]}
        
        # Cache the result
        self.country_map = countries
        return countries

    def resolve_country_qid(self, country: str) -> str:
        """
        Enhanced resolver: Resolve a country name, WikiData QID, or ISO code to a WikiData QID.
        
        Args:
            country: Country name, WikiData QID (e.g., 'Q217'), or ISO code (e.g., 'MD', 'MC', 'RS')
            
        Returns:
            WikiData QID for the country
        """
        import pycountry
        
        # If it's already a QID (starts with Q), return it
        if isinstance(country, str) and country.startswith('Q') and country[1:].isdigit():
            return country
        
        # If it looks like an ISO code (2-3 letter uppercase), try ISO lookup first
        if isinstance(country, str) and len(country) in [2, 3] and country.isupper():
            qid = self.get_qid_from_iso_code(country)
            if qid:
                return qid
        
        # Try to resolve short codes via pycountry + ISO lookup
        if isinstance(country, str) and len(country) in [2, 3]:
            try:
                if len(country) == 2:
                    country_obj = pycountry.countries.get(alpha_2=country.upper())
                else:
                    country_obj = pycountry.countries.get(alpha_3=country.upper())
                
                if country_obj:
                    qid = self.get_qid_from_iso_code(country_obj.alpha_2)
                    if qid:
                        return qid
            except:
                pass
        
        # Get the map of country names to QIDs (existing functionality)
        country_map = self.get_country_qids_from_wikidata()
        
        # Try direct lookup
        if country in country_map:
            return country_map[country]
            
        # Try case-insensitive lookup
        for name, qid in country_map.items():
            if name.lower() == country.lower():
                return qid
        
        # Try with pycountry to get official name, then try ISO lookup
        try:
            country_obj = pycountry.countries.lookup(country)
            
            # First try ISO code lookup
            qid = self.get_qid_from_iso_code(country_obj.alpha_2)
            if qid:
                return qid
                
            # Try with the official name from pycountry
            country_name = country_obj.name
            if country_name in country_map:
                return country_map[country_name]
                
            # Try with common name variations
            for name, qid in country_map.items():
                if country_name.lower() in name.lower() or name.lower() in country_name.lower():
                    return qid
                    
        except LookupError:
            pass
            
        # Not found
        logger.warning(f"Could not resolve country: {country}")
        return None

    def resolve_org_type_qid(self, org_type: str) -> str:
        """
        Resolve an organization type name or QID to a Wikidata QID.
        
        Args:
            org_type: Organization type name or QID (e.g., 'university' or 'Q3918')
            
        Returns:
            Wikidata QID for the organization type
        """
        # If it's already a QID (starts with Q), return it
        if isinstance(org_type, str) and org_type.startswith('Q'):
            return org_type
            
        # Try in short list
        if org_type in self.organisation_types_short:
            return self.organisation_types_short[org_type]
            
        # Try in extended list
        if org_type in self.organisation_types_extended:
            return self.organisation_types_extended[org_type]
            
        # Try case-insensitive lookup in short list
        for name, qid in self.organisation_types_short.items():
            if name.lower() == org_type.lower():
                return qid
                
        # Try case-insensitive lookup in extended list
        for name, qid in self.organisation_types_extended.items():
            if name.lower() == org_type.lower():
                return qid
                
        # Try reverse lookup in case a QID value was passed as name
        if org_type in self.org_type_map_short:
            return org_type
        if org_type in self.org_type_map_extended:
            return org_type
            
        # Not found
        logger.warning(f"Could not resolve organization type: {org_type}")
        return None

    def fetch_results(self, type_qid, country_qid):
        """
        Query Wikidata for organizations of a specific type in a specific country
        using a batched approach with retries and caching.
        Automatically splits large datasets alphabetically.
        """
        if self.verbose:
            logger.info(f"fetch_results: Fetching data for type {type_qid} in country {country_qid}...")
        
        # Get count first (your existing code)
        total_count = None
        try:
            total_count = self.cache.get_cached_count(type_qid, country_qid)
            
            if total_count is None:
                # Your existing count query code here...
                count_query = f"""
                SELECT (COUNT(*) as ?count) WHERE {{
                    ?id wdt:P31 wd:{type_qid};
                        wdt:P17 wd:{country_qid}.
                }}
                """
                
                sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
                sparql.setTimeout(TIMEOUT_COUNT_REQUEST)
                sparql.setQuery(count_query)
                sparql.setReturnFormat(JSON)
                
                try:
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    
                    def execute_query():
                        try:
                            count_results = sparql.query().convert()
                            result_queue.put(int(count_results['results']['bindings'][0]['count']['value']))
                        except Exception as e:
                            result_queue.put(None)
                    
                    query_thread = threading.Thread(target=execute_query)
                    query_thread.daemon = True
                    query_thread.start()
                    
                    try:
                        total_count = result_queue.get(timeout=15)
                        if total_count is not None:
                            self.cache.cache_count(type_qid, country_qid, total_count)
                            if self.verbose:
                                logger.info(f"Found {total_count} organizations to fetch")
                    except queue.Empty:
                        if self.verbose:
                            logger.warning("Count query timed out. Proceeding with batched retrieval.")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Count query failed: {e}. Proceeding with batched retrieval.")
        except Exception as e:
            if self.verbose:
                logger.warning(f"Count query exception: {e}. Proceeding without count information.")
        
        # NEW: Auto-split logic
        if total_count and total_count > 2000:
            if self.verbose:
                logger.info(f"Large dataset detected ({total_count} items). Using alphabetical splitting...")
            return self._fetch_results_with_alphabetical_split(type_qid, country_qid, total_count)
        else:
            # Use existing logic for smaller datasets
            return self._fetch_results_normal(type_qid, country_qid, total_count)

    def _fetch_results_normal(self, type_qid, country_qid, total_count=None):
        """
        Your existing fetch_results logic for normal-sized datasets.
        """
        all_results = []
        batch_size = BATCH_SIZE
        offset = 0
        max_batches = MAX_NUM_BATCHES
        batch_num = 1
        
        # If small enough, fetch in one go
        if total_count and total_count <= batch_size:
            batch_results = self._fetch_batch_with_retry(type_qid, country_qid, limit=total_count, offset=0)
            return batch_results if batch_results else []
        
        # Your existing batch processing loop
        while batch_num <= max_batches:
            try:
                # Check cache first
                cached_batch = None
                try:
                    cached_batch = self.cache.get_cached_batch(type_qid, country_qid, batch_size, offset)
                except Exception as cache_error:
                    if self.verbose:
                        logger.warning(f"Cache error: {cache_error}. Proceeding without cache.")
                
                if cached_batch is not None:
                    if self.verbose:
                        logger.info(f"Using cached batch at offset {offset}")
                    batch_results = cached_batch
                else:
                    # Fetch from WikiData with timeout protection
                    try:
                        import threading
                        import queue
                        
                        result_queue = queue.Queue()
                        
                        def execute_batch_fetch():
                            try:
                                results = self._fetch_batch_with_retry(type_qid, country_qid, limit=batch_size, offset=offset)
                                result_queue.put(results)
                            except Exception as e:
                                result_queue.put([])
                        
                        query_thread = threading.Thread(target=execute_batch_fetch)
                        query_thread.daemon = True
                        query_thread.start()
                        
                        try:
                            batch_results = result_queue.get(timeout=60)
                        except queue.Empty:
                            if self.verbose:
                                logger.warning(f"Batch query at offset {offset} timed out. Proceeding to next batch.")
                            batch_results = []
                    except Exception as batch_error:
                        if self.verbose:
                            logger.error(f"Error during batch retrieval: {batch_error}")
                        batch_results = []
                    
                    # Cache successful results
                    if batch_results:
                        try:
                            self.cache.cache_batch(type_qid, country_qid, batch_size, offset, batch_results)
                        except Exception as cache_error:
                            if self.verbose:
                                logger.warning(f"Error caching batch: {cache_error}")
                
                # If no results, we're done
                if not batch_results:
                    if self.verbose:
                        logger.info(f"No more results at offset {offset}. Completed retrieval.")
                    break
                    
                all_results.extend(batch_results)
                
                if self.verbose:
                    logger.info(f"Batch {batch_num}: Retrieved {len(batch_results)} organizations (offset {offset})")
                
                offset += len(batch_results)
                batch_num += 1
                
                # Check if we have everything
                if total_count is not None and len(all_results) >= total_count:
                    if self.verbose:
                        logger.info(f"Retrieved all {len(all_results)} organizations. Completed retrieval.")
                    break
                    
                time.sleep(1)  # Brief pause between batches
                
            except Exception as e:
                logger.error(f"Unexpected error in batch processing: {e}")
                offset += batch_size
                batch_num += 1
        
        if self.verbose:
            logger.info(f"Total organizations fetched: {len(all_results)}")
        
        return all_results

    def _fetch_results_with_alphabetical_split(self, type_qid, country_qid, total_count):
        """
        Fetch results for large datasets by splitting alphabetically with retry logic.
        """
        # Define alphabetical ranges that should give roughly equal splits
        alphabet_ranges = [
            ('A', 'C'), ('D', 'F'), ('G', 'I'), ('J', 'L'),
            ('M', 'O'), ('P', 'R'), ('S', 'U'), ('V', 'Z')
        ]
        
        all_results = []
        successful_splits = 0
        failed_splits = []
        
        if self.verbose:
            logger.info(f"Splitting {total_count} organizations into {len(alphabet_ranges)} alphabetical ranges...")
        
        # First pass: try all splits
        for i, (start_letter, end_letter) in enumerate(alphabet_ranges):
            try:
                if self.verbose:
                    logger.info(f"Processing split {i+1}/{len(alphabet_ranges)}: {start_letter}-{end_letter}")
                
                # Fetch this alphabetical range
                split_results = self._fetch_alphabetical_range(
                    type_qid, country_qid, start_letter, end_letter
                )
                
                if split_results:
                    all_results.extend(split_results)
                    successful_splits += 1
                    
                    if self.verbose:
                        logger.info(f"✅ Split {start_letter}-{end_letter}: Retrieved {len(split_results)} organizations")
                else:
                    if self.verbose:
                        logger.warning(f"❌ Split {start_letter}-{end_letter}: Failed or no organizations found")
                    failed_splits.append((start_letter, end_letter))
                
                # Brief pause between splits to be nice to WikiData
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing alphabetical split {start_letter}-{end_letter}: {e}")
                failed_splits.append((start_letter, end_letter))
                continue
        
        # Second pass: retry failed splits with longer delays
        if failed_splits:
            if self.verbose:
                logger.info(f"Retrying {len(failed_splits)} failed splits with longer delays...")
            
            retry_successful = 0
            
            for start_letter, end_letter in failed_splits:
                try:
                    if self.verbose:
                        logger.info(f"Retrying split: {start_letter}-{end_letter}")
                    
                    # Longer delay before retry
                    time.sleep(30)  # 30 seconds before retry
                    
                    # Retry the split
                    split_results = self._fetch_alphabetical_range_with_retry(
                        type_qid, country_qid, start_letter, end_letter
                    )
                    
                    if split_results:
                        all_results.extend(split_results)
                        retry_successful += 1
                        
                        if self.verbose:
                            logger.info(f"✅ Retry {start_letter}-{end_letter}: Retrieved {len(split_results)} organizations")
                    else:
                        if self.verbose:
                            logger.warning(f"❌ Retry {start_letter}-{end_letter}: Still failed")
                    
                except Exception as e:
                    logger.error(f"Error retrying split {start_letter}-{end_letter}: {e}")
                    continue
            
            successful_splits += retry_successful
            
            if self.verbose:
                logger.info(f"Retry phase complete: {retry_successful}/{len(failed_splits)} splits recovered")
        
        if self.verbose:
            logger.info(f"Alphabetical splitting complete:")
            logger.info(f"  Successful splits: {successful_splits}/{len(alphabet_ranges)}")
            logger.info(f"  Failed splits: {len(alphabet_ranges) - successful_splits}")
            logger.info(f"  Total organizations retrieved: {len(all_results)}")
            coverage = (len(all_results) / total_count * 100) if total_count > 0 else 0
            logger.info(f"  Coverage: {coverage:.1f}% of expected {total_count}")
        
        return all_results

    def _fetch_alphabetical_range_with_retry(self, type_qid, country_qid, start_letter, end_letter, max_retries=3):
        """
        Fetch an alphabetical range with enhanced retry logic.
        """
        for retry_attempt in range(max_retries):
            try:
                if retry_attempt > 0:
                    # Exponential backoff for retries
                    delay = 60 * (2 ** (retry_attempt - 1))  # 60s, 120s, 240s
                    if self.verbose:
                        logger.info(f"Retry attempt {retry_attempt}/{max_retries-1} for {start_letter}-{end_letter} after {delay}s delay")
                    time.sleep(delay)
                
                # Try to fetch the range
                results = self._fetch_alphabetical_range(type_qid, country_qid, start_letter, end_letter)
                
                if results:
                    return results
                elif retry_attempt == max_retries - 1:
                    # Last attempt failed
                    if self.verbose:
                        logger.warning(f"All retry attempts failed for {start_letter}-{end_letter}")
                    return []
                    
            except Exception as e:
                if retry_attempt == max_retries - 1:
                    # Last attempt
                    if self.verbose:
                        logger.error(f"Final retry failed for {start_letter}-{end_letter}: {e}")
                    return []
                else:
                    if self.verbose:
                        logger.warning(f"Retry {retry_attempt} failed for {start_letter}-{end_letter}: {e}")
                    continue
        
        return []

    def _fetch_batch_with_alphabet_filter(self, type_qid, country_qid, start_letter, end_letter, limit, offset):
        """
        Fetch a batch with alphabetical filtering applied, with enhanced error handling.
        """
        max_attempts = 3
        base_delay = 10
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    delay = base_delay * (attempt + 1)
                    if self.verbose:
                        logger.info(f"Retry attempt {attempt} for batch {start_letter}-{end_letter} offset {offset} after {delay}s")
                    time.sleep(delay)
                
                # Create SPARQL query with name filter
                sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
                sparql._session = self.session
                
                # Increase timeout for problematic splits
                timeout = TIMEOUT_BATCH_REQUEST + (30 * attempt)  # 60s, 90s, 120s
                sparql.setTimeout(timeout)
                
                # Modified query with alphabetical filter
                query = f"""
                SELECT ?id ?name_en 
                    (GROUP_CONCAT(DISTINCT CONCAT(LANG(?alias), ":", STR(?alias)); separator="|") AS ?aliases)
                    (GROUP_CONCAT(DISTINCT CONCAT(LANG(?name_full), ":", STR(?name_full)); separator="|") AS ?all_names)
                    (GROUP_CONCAT(DISTINCT ?acronym; separator="|") AS ?acronyms)
                    ?city ?region ?country_name ?country_code
                    (GROUP_CONCAT(DISTINCT ?parent_name; separator="|") AS ?relationships)
                    (GROUP_CONCAT(DISTINCT ?website; separator="|") AS ?websites)
                    ?coordinates ?formation_label
                    ?hq_city ?hq_country_name
                    ?ror ?grid ?lei ?crunchbase
                WHERE {{
                    ?id wdt:P31 wd:{type_qid};
                        wdt:P17 wd:{country_qid}.

                    # Get English name for filtering
                    ?id rdfs:label ?name_en. 
                    FILTER(LANG(?name_en) = "en")
                    
                    # Alphabetical filter - names starting with letters in range
                    FILTER(REGEX(STR(?name_en), "^[{start_letter}-{end_letter}]", "i"))

                    # All your existing OPTIONAL clauses...
                    OPTIONAL {{ ?id rdfs:label ?name_full. }}
                    OPTIONAL {{ ?id skos:altLabel ?alias. }}
                    OPTIONAL {{ ?id wdt:P1813 ?acronym. }}
                    OPTIONAL {{ ?id wdt:P6782 ?ror. }}
                    OPTIONAL {{ ?id wdt:P2427 ?grid. }}
                    OPTIONAL {{ ?id wdt:P1278 ?lei. }}
                    OPTIONAL {{ ?id wdt:P2088 ?crunchbase. }}
                    OPTIONAL {{ ?id wdt:P856 ?website. }}
                    OPTIONAL {{ ?id wdt:P625 ?coordinates. }}
                    OPTIONAL {{ 
                        ?id wdt:P740 ?formation_location.
                        ?formation_location rdfs:label ?formation_label.
                        FILTER(LANG(?formation_label) = "en")
                    }}

                    OPTIONAL {{
                        ?id wdt:P131 ?city_item.
                        ?city_item wdt:P31/wdt:P279* wd:Q515.
                        ?city_item rdfs:label ?city.
                        FILTER(LANG(?city) = "en")

                        OPTIONAL {{
                        ?city_item wdt:P131 ?region_item.
                        ?region_item rdfs:label ?region.
                        FILTER(LANG(?region) = "en")
                        }}

                        OPTIONAL {{
                        ?city_item wdt:P17 ?country_item.
                        ?country_item rdfs:label ?country_name.
                        ?country_item wdt:P298 ?country_code.
                        FILTER(LANG(?country_name) = "en")
                        }}
                    }}

                    OPTIONAL {{
                        ?id wdt:P749 ?parent.
                        ?parent rdfs:label ?parent_name.
                        FILTER(LANG(?parent_name) = "en")
                    }}

                    OPTIONAL {{
                        ?id wdt:P159 ?hq_item.
                        ?hq_item rdfs:label ?hq_city.
                        FILTER(LANG(?hq_city) = "en")
                        OPTIONAL {{
                        ?hq_item wdt:P17 ?hq_country.
                        ?hq_country rdfs:label ?hq_country_name.
                        FILTER(LANG(?hq_country_name) = "en")
                        }}
                    }}

                    # Fallback country info
                    wd:{country_qid} rdfs:label ?fallback_country_label.
                    FILTER(LANG(?fallback_country_label) = "en")
                    wd:{country_qid} wdt:P298 ?fallback_country_code.

                    BIND(COALESCE(?country_name, ?fallback_country_label) AS ?country_name)
                    BIND(COALESCE(?country_code, ?fallback_country_code) AS ?country_code)
                }}
                GROUP BY ?id ?name_en ?city ?region ?country_name ?country_code
                        ?coordinates ?formation_label ?hq_city ?hq_country_name
                        ?ror ?grid ?website ?lei ?crunchbase
                ORDER BY ?name_en
                LIMIT {limit}
                OFFSET {offset}
                """
                
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                
                results = sparql.query().convert()
                rows = results['results']['bindings']
                
                # Convert to flat list of dicts (same as your existing code)
                flattened_rows = []
                for row in rows:
                    flat_row = {}
                    for key, value in row.items():
                        flat_row[key] = value.get('value', None)
                    flattened_rows.append(flat_row)
                
                return flattened_rows
                
            except Exception as e:
                error_str = str(e).lower()
                
                if attempt < max_attempts - 1:
                    if "timeout" in error_str or "504" in error_str:
                        if self.verbose:
                            logger.warning(f"Timeout on attempt {attempt+1} for {start_letter}-{end_letter}, retrying...")
                        continue
                    elif "429" in error_str or "too many requests" in error_str:
                        if self.verbose:
                            logger.warning(f"Rate limit on attempt {attempt+1} for {start_letter}-{end_letter}, waiting longer...")
                        time.sleep(60)  # Extra delay for rate limits
                        continue
                    else:
                        if self.verbose:
                            logger.warning(f"Error on attempt {attempt+1} for {start_letter}-{end_letter}: {e}")
                        continue
                else:
                    # Final attempt failed
                    if self.verbose:
                        logger.error(f"All attempts failed for {start_letter}-{end_letter}: {e}")
                    return []
        
        return []

    def _fetch_alphabetical_range(self, type_qid, country_qid, start_letter, end_letter):
        """
        Fetch organizations whose names start with letters in the given range.
        """
        all_results = []
        batch_size = BATCH_SIZE
        offset = 0
        max_batches = MAX_NUM_BATCHES  # Prevent infinite loops
        batch_num = 1
        
        while batch_num <= max_batches:
            try:
                # Check if this specific range+offset is cached
                cache_key_suffix = f"{start_letter}{end_letter}"
                cached_batch = None
                try:
                    # Use a modified cache key that includes the alphabet range
                    cached_batch = self.cache.get_cached_batch(
                        f"{type_qid}_{cache_key_suffix}", country_qid, batch_size, offset
                    )
                except:
                    pass  # Cache miss, continue normally
                
                if cached_batch is not None:
                    if self.verbose:
                        logger.info(f"Using cached batch for {start_letter}-{end_letter} at offset {offset}")
                    batch_results = cached_batch
                else:
                    # Fetch this batch with alphabetical filter
                    batch_results = self._fetch_batch_with_alphabet_filter(
                        type_qid, country_qid, start_letter, end_letter, batch_size, offset
                    )
                    
                    # Cache the result
                    if batch_results:
                        try:
                            self.cache.cache_batch(
                                f"{type_qid}_{cache_key_suffix}", country_qid, batch_size, offset, batch_results
                            )
                        except:
                            pass  # Cache failure is not critical
                
                # If no results, we're done with this range
                if not batch_results:
                    break
                
                all_results.extend(batch_results)
                
                if self.verbose and len(all_results) % 100 == 0:  # Log every 100 orgs
                    logger.info(f"  {start_letter}-{end_letter}: {len(all_results)} orgs so far...")
                
                # Update for next batch
                offset += len(batch_results)
                batch_num += 1
                
                # Brief pause between batches
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in alphabetical range {start_letter}-{end_letter} at offset {offset}: {e}")
                break
        
        return all_results


    def _fetch_batch_with_retry(self, type_qid, country_qid, limit=BATCH_SIZE, offset=0, max_retries=MAX_NUM_RETRIES):
        """
        Fetch a single batch of organizations from Wikidata with retries.
        
        Args:
            type_qid: Wikidata QID for the organization type
            country_qid: Wikidata QID for the country
            limit: Number of results to return
            offset: Offset for pagination
            max_retries: Maximum number of retry attempts
                
        Returns:
            List of organization dictionaries for this batch
        """
        retry_count = 0
        current_limit = limit
        delay_base = DELAY_BASE_RETRY  # Base delay in seconds
        
        while retry_count <= max_retries:
            try:
                # Fetch batch
                batch_results = self._fetch_batch(type_qid, country_qid, limit=current_limit, offset=offset)
                return batch_results
                
            except EndPointInternalError as e:
                if "TimeoutException" in str(e):
                    # For timeouts, reduce batch size and retry
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        # Reduce batch size for retry
                        current_limit = max(10, current_limit // 2)
                        
                        # Calculate exponential backoff delay
                        delay = delay_base * (2 ** (retry_count - 1))
                        
                        if self.verbose:
                            logger.warning(f"Batch timed out at offset {offset}. "
                                          f"Retry {retry_count}/{max_retries} with reduced batch size {current_limit} "
                                          f"after {delay}s delay")
                        
                        # Wait before retry
                        time.sleep(delay)
                    else:
                        # Log final failure
                        if self.verbose:
                            logger.error(f"Batch at offset {offset} failed after {max_retries} retries")
                        return []
                else:
                    # Other endpoint errors - log and retry with same parameters
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        delay = delay_base * (2 ** (retry_count - 1))
                        
                        if self.verbose:
                            logger.warning(f"Endpoint error at offset {offset}. "
                                         f"Retry {retry_count}/{max_retries} after {delay}s delay: {e}")
                        
                        # Wait before retry
                        time.sleep(delay)
                    else:
                        # Log final failure
                        if self.verbose:
                            logger.error(f"Batch at offset {offset} failed after {max_retries} retries: {e}")
                        return []
                        
            except Exception as e:
                # Handle other exceptions
                retry_count += 1
                
                if retry_count <= max_retries:
                    delay = delay_base * (2 ** (retry_count - 1))
                    
                    if self.verbose:
                        logger.warning(f"Error fetching batch at offset {offset}. "
                                     f"Retry {retry_count}/{max_retries} after {delay}s delay: {e}")
                    
                    # Wait before retry
                    time.sleep(delay)
                else:
                    # Log final failure
                    if self.verbose:
                        logger.error(f"Batch at offset {offset} failed after {max_retries} retries: {e}")
                    return []
                    
        # Should never reach here, but just in case
        return []

    def _fetch_batch(self, type_qid, country_qid, limit=BATCH_SIZE, offset=0):
        """
        Fetch a single batch of organizations from Wikidata.
        
        Args:
            type_qid: Wikidata QID for the organization type
            country_qid: Wikidata QID for the country
            limit: Number of results to return
            offset: Offset for pagination
                
        Returns:
            List of organization dictionaries for this batch
        """
        # Set SPARQL endpoint
        sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
        
        # Set cache session
        sparql._session = self.session
        
        # Set timeout
        sparql.setTimeout(TIMEOUT_BATCH_REQUEST)
        
        # Create query with LIMIT and OFFSET
        query = f"""
        SELECT ?id ?name_en 
            (GROUP_CONCAT(DISTINCT CONCAT(LANG(?alias), ":", STR(?alias)); separator="|") AS ?aliases)
            (GROUP_CONCAT(DISTINCT CONCAT(LANG(?name_full), ":", STR(?name_full)); separator="|") AS ?all_names)
            (GROUP_CONCAT(DISTINCT ?acronym; separator="|") AS ?acronyms)
            ?city ?region ?country_name ?country_code
            (GROUP_CONCAT(DISTINCT ?parent_name; separator="|") AS ?relationships)
            (GROUP_CONCAT(DISTINCT ?website; separator="|") AS ?websites)
            ?coordinates ?formation_label
            ?hq_city ?hq_country_name
            ?ror ?grid ?lei ?crunchbase
        WHERE {{
        ?id wdt:P31 wd:{type_qid};  # instance of type
            wdt:P17 wd:{country_qid}.      # country

        OPTIONAL {{ ?id rdfs:label ?name_en. FILTER(LANG(?name_en) = "en") }}
        OPTIONAL {{ ?id rdfs:label ?name_full. }}
        OPTIONAL {{ ?id skos:altLabel ?alias. }}
        OPTIONAL {{ ?id wdt:P1813 ?acronym. }}
        OPTIONAL {{ ?id wdt:P6782 ?ror. }}
        OPTIONAL {{ ?id wdt:P2427 ?grid. }}
        OPTIONAL {{ ?id wdt:P1278 ?lei. }}
        OPTIONAL {{ ?id wdt:P2088 ?crunchbase. }}
        OPTIONAL {{ ?id wdt:P856 ?website. }}
        OPTIONAL {{ ?id wdt:P625 ?coordinates. }}
        OPTIONAL {{ 
            ?id wdt:P740 ?formation_location.
            ?formation_location rdfs:label ?formation_label.
            FILTER(LANG(?formation_label) = "en")
        }}

        OPTIONAL {{
            ?id wdt:P131 ?city_item.
            ?city_item wdt:P31/wdt:P279* wd:Q515.
            ?city_item rdfs:label ?city.
            FILTER(LANG(?city) = "en")

            OPTIONAL {{
            ?city_item wdt:P131 ?region_item.
            ?region_item rdfs:label ?region.
            FILTER(LANG(?region) = "en")
            }}

            OPTIONAL {{
            ?city_item wdt:P17 ?country_item.
            ?country_item rdfs:label ?country_name.
            ?country_item wdt:P298 ?country_code.
            FILTER(LANG(?country_name) = "en")
            }}
        }}

        OPTIONAL {{
            ?id wdt:P749 ?parent.
            ?parent rdfs:label ?parent_name.
            FILTER(LANG(?parent_name) = "en")
        }}

        OPTIONAL {{
            ?id wdt:P159 ?hq_item.
            ?hq_item rdfs:label ?hq_city.
            FILTER(LANG(?hq_city) = "en")
            OPTIONAL {{
            ?hq_item wdt:P17 ?hq_country.
            ?hq_country rdfs:label ?hq_country_name.
            FILTER(LANG(?hq_country_name) = "en")
            }}
        }}

        # Fallback if missing
        wd:{country_qid} rdfs:label ?fallback_country_label.
        FILTER(LANG(?fallback_country_label) = "en")
        wd:{country_qid} wdt:P298 ?fallback_country_code.

        BIND(COALESCE(?country_name, ?fallback_country_label) AS ?country_name)
        BIND(COALESCE(?country_code, ?fallback_country_code) AS ?country_code)

        }}
        GROUP BY ?id ?name_en ?city ?region ?country_name ?country_code
                ?coordinates ?formation_label ?hq_city ?hq_country_name
                ?ror ?grid ?website ?lei ?crunchbase
        LIMIT {limit}
        OFFSET {offset}
        """
        
        #print(query)

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        results = sparql.query().convert()
        
        # Extract bindings
        rows = results['results']['bindings']
        
        # Convert to flat list of dicts
        flattened_rows = []
        for row in rows:
            flat_row = {}
            for key, value in row.items():
                flat_row[key] = value.get('value', None)
            flattened_rows.append(flat_row)
        
        return flattened_rows

        
    def expand_organization_locations(self, row):
        # Convert external_ids to a simple dict if it's not already
        external_ids = {} if pd.isna(row.get('external_ids')) else row.get('external_ids')

        base_data = {
            'id': row['id'],
            'name': row.get('name_en'),
            'aliases': row.get('aliases', ''),
            'all_names': row.get('all_names', ''),
            'acronyms': row.get('acronyms'),
            'relationships': row.get('relationships'),
            'websites': row.get('websites'),
            'external_ids': row.get('external_ids'),
        }

        entries = []

        city = row.get('city')
        region = row.get('region')
        country_name = row.get('country_name')
        country_code = row.get('country_code')
        hq_city = row.get('hq_city')
        hq_country_name = row.get('hq_country_name')
        coordinates = row.get('coordinates')
        formation_label = row.get('formation_label')

        # 1. Primary location
        if pd.notna(city) and pd.notna(country_name):
            entry = base_data.copy()
            entry.update({
                'location_type': 'primary',
                'city': city,
                'region': region,
                'country_name': country_name,
                'country_code': country_code,
            })
            entries.append(entry)

        # 1.b. Primary location with no city
        if pd.isna(city) and pd.notna(country_name) and country_name != hq_country_name:
            entry = base_data.copy()
            entry.update({
                'location_type': 'country_only',
                'city': None,
                'region': None,
                'country_name': country_name,
                'country_code': country_code,
                'coordinates': coordinates,
            })
            entries.append(entry)

        # 2. Headquarters (if different)
        if (
            pd.notna(hq_city) and pd.notna(hq_country_name) and
            (hq_city != city or hq_country_name != country_name)
        ):
            entry = base_data.copy()
            entry.update({
                'location_type': 'headquarters',
                'city': hq_city,
                'region': None,
                'country_name': hq_country_name,
                'country_code': None,
            })
            entries.append(entry)

        # 3. Fallback (Formation Label)
        if not entries and pd.notna(formation_label):
            city_fallback = formation_label if pd.isna(city) and pd.isna(hq_city) else None
            entry = base_data.copy()
            entry.update({
                'location_type': 'formation_label',
                'city': city_fallback,
                'region': None,
                'country_name': country_name,
                'country_code': country_code,
            })
            entries.append(entry)

        # 4. Fallback (Coordinates)
        if not entries and pd.notna(coordinates):
            entry = base_data.copy()
            entry.update({
                'location_type': 'coordinates',
                'city': None,
                'region': None,
                'country_name': country_name,
                'country_code': country_code,
                'coordinates': coordinates,
            })
            entries.append(entry)

        return entries


    # Step 2: Define helper to parse LANG:label format into a dict
    def parse_lang_values(self, value_str):
        if pd.isna(value_str):
            return {}
        parts = value_str.split('|')
        return {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts if ':' in p}

    # Step 3: Main processing
    def process_lang_name_variants(self, row):
        lang_codes = self.lang_lookup.get(row['country_name'], ['en'])
        
        # Parse all_names and aliases into dicts
        names_dict = self.parse_lang_values(row.get('all_names', ''))
        aliases_dict = self.parse_lang_values(row.get('aliases', ''))

        # Keep only official langs + English
        filtered_names = {k: v for k, v in names_dict.items() if k in lang_codes or k == 'en'}
        filtered_aliases = {k: v for k, v in aliases_dict.items() if k in lang_codes or k == 'en'}
        
        # Determine main name: prefer first match in lang_codes, fallback to English, else use current name_en
        name = None
        for lang in lang_codes:
            if lang in filtered_names:
                name = filtered_names[lang]
                break
        if not name:
            name = filtered_names.get('en', row['name'])

        return pd.Series({
            'name': name,
            'all_names': filtered_names,
            'aliases': filtered_aliases
        })
    
    def country_flag_emoji_from_name(self, name: str) -> str:
        """
        Returns the emoji flag for a given country name.
        """
        try:
            country = pycountry.countries.lookup(name)
            alpha_2 = country.alpha_2.upper()
            return ''.join(chr(127397 + ord(char)) for char in alpha_2)
        except LookupError:
            return ''
    
    def get_index(self, countries: Optional[Union[List[str], str]] = None, 
                 org_types: Optional[Union[List[str], str]] = None):
        """
        Get organization data from Wikidata.
        
        Args:
            countries: Country names or QIDs, 'all', or None for all countries
            org_types: Organization type names or QIDs, 'short', 'extended', or None for short list
            
        Returns:
            DataFrame of organizations
        """
        cached_dataset = self.cache.get_cached_dataset(countries, org_types)
        if cached_dataset is not None:
            if self.verbose:
                logger.info(f"Using cached complete dataset with {len(cached_dataset)} organizations")
            return cached_dataset
        
        # Get country QIDs map
        name_to_qid = self.get_country_qids_from_wikidata()
        qid_to_country_name = {qid: name for name, qid in name_to_qid.items()}

        # Process countries parameter
        country_qids = []
        if isinstance(countries, str):
            if countries == "all":
                country_qids = list(qid_to_country_name.keys())
            else:
                resolved_qid = self.resolve_country_qid(countries)
                if resolved_qid:
                    country_qids = [resolved_qid]
                else:
                    logger.warning(f"Could not resolve country: {countries}")
                    country_qids = []
        elif isinstance(countries, list):
            for country in countries:
                resolved_qid = self.resolve_country_qid(country)
                if resolved_qid:
                    country_qids.append(resolved_qid)
        elif countries is None:
            country_qids = list(qid_to_country_name.keys())

        # Process org_types parameter
        org_type_qids = []
        if isinstance(org_types, str):
            if org_types == "short":
                org_type_qids = list(self.organisation_types_short.values())
            elif org_types == "extended":
                org_type_qids = list(self.organisation_types_extended.values())
            else:
                resolved_qid = self.resolve_org_type_qid(org_types)
                if resolved_qid:
                    org_type_qids = [resolved_qid]
                else:
                    logger.warning(f"Could not resolve organization type: {org_types}")
                    org_type_qids = []
        elif isinstance(org_types, list):
            for org_type in org_types:
                resolved_qid = self.resolve_org_type_qid(org_type)
                if resolved_qid:
                    org_type_qids.append(resolved_qid)
        elif org_types is None:
            org_type_qids = list(self.organisation_types_short.values())

        if self.verbose:
            logger.info(f"Processing {len(country_qids)} countries and {len(org_type_qids)} organization types")
            logger.info(f"Organization type QIDs: {org_type_qids}")

        temp_dir = Path("/tmp/org_index")
        temp_dir.mkdir(parents=True, exist_ok=True)

        all_rows = []

        # Create a single master progress bar that shows overall progress
        total_combinations = len(country_qids) * len(org_type_qids)
        with tqdm(total=total_combinations, desc="Fetching WikiData organizations", leave=True) as master_pbar:
            for country_qid in country_qids:
                country_name = qid_to_country_name.get(country_qid, "Unknown")
                flag = self.country_flag_emoji_from_name(country_name)
                
                for org_type_qid in org_type_qids:
                    try:
                        # Get organization type name for logging
                        org_type_name = self.org_type_map_short.get(org_type_qid) or self.org_type_map_extended.get(org_type_qid) or org_type_qid
                        
                        # Create a descriptive progress message
                        progress_msg = f"Processing {flag} {country_name} - {org_type_name}"
                        
                        if self.verbose:
                            logger.info(f"Fetching data for type {org_type_name} ({org_type_qid}) in country {country_name} ({country_qid})...")
                        
                        # Update the master progress bar description
                        master_pbar.set_description(progress_msg)
                        
                        # Fetch results for this country-org type combination
                        res = self.fetch_results(org_type_qid, country_qid)
                        
                        if self.verbose:
                            logger.info(f"Retrieved {len(res)} results for {country_name}/{org_type_name}")
                        
                        # Process results if we have any
                        if res:
                            df = pd.DataFrame(res)
                            
                            if not df.empty:
                                # Simplify the workflow
                                df_clean = df.drop(columns=['crunchbase', 'ror', 'grid', 'lei'], errors='ignore')
                                df_clean['external_ids'] = df.apply(
                                    lambda row: {k: row[k] for k in ['crunchbase', 'ror', 'grid', 'lei'] 
                                                if k in row and pd.notna(row[k])} or None, 
                                    axis=1
                                )
                                
                                for _, row in df_clean.iterrows():
                                    all_rows.extend(self.expand_organization_locations(row))
                                
                                # Save partial results to disk
                                try:
                                    expanded_df = pd.DataFrame(all_rows)
                                    expanded_df.to_parquet(temp_dir / f"{org_type_qid}_{country_qid}.parquet", index=False)
                                except Exception as save_error:
                                    logger.error(f"Error saving partial results: {save_error}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {org_type_qid}, {country_qid}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Update the master progress bar
                    master_pbar.update(1)

        if not all_rows:
            logger.warning("No data retrieved. Returning empty DataFrame.")
            return pd.DataFrame()

        # Final processing
        try:
            expanded_df = pd.DataFrame(all_rows)
            
            # Apply language processing
            expanded_df[['name', 'all_names', 'aliases']] = expanded_df.apply(self.process_lang_name_variants, axis=1)

            # Filter and deduplicate
            expanded_df = expanded_df[expanded_df.name.notnull()] \
                                  .drop_duplicates(['id', 'name', 'city', 'country_name'], keep='first')

            # Filter by countries in language lookup
            valid_countries = expanded_df.country_name.isin(self.lang_lookup.keys())
            if valid_countries.any():
                expanded_df = expanded_df[valid_countries].reset_index(drop=True)
            else:
                # If no countries match, keep all and warn
                logger.warning("No countries match the language lookup. Keeping all countries.")
                expanded_df = expanded_df.reset_index(drop=True)

            self.cache.cache_dataset(countries, org_types, expanded_df)
            if self.verbose:
                logger.info(f"Cached complete dataset with {len(expanded_df)} organizations")

            return expanded_df
        except Exception as final_error:
            logger.error(f"Error in final processing: {final_error}")
            import traceback
            traceback.print_exc()
            # Return an empty DataFrame instead of None to avoid ambiguity
            return pd.DataFrame()
