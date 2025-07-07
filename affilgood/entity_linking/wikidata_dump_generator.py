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
TIMEOUT_COUNT_REQUEST = 30
TIMEOUT_BATCH_REQUEST = 60
DELAY_BASE_RETRY = 5
MAX_NUM_RETRIES = 5

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class WikiDataCache:
    """Cache for WikiData SPARQL query results."""
    
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
            
            # Use try/except for each directory creation
            try:
                self.count_cache_dir.mkdir(exist_ok=True)
            except Exception:
                # Fallback to string paths if pathlib has issues
                self.count_cache_dir = os.path.join(cache_dir, "count_queries")
                os.makedirs(self.count_cache_dir, exist_ok=True)
                
            try:
                self.batch_cache_dir.mkdir(exist_ok=True)
            except Exception:
                # Fallback to string paths if pathlib has issues
                self.batch_cache_dir = os.path.join(cache_dir, "batch_queries")
                os.makedirs(self.batch_cache_dir, exist_ok=True)
            
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
            self.cache_expiration = timedelta(days=cache_expiration_days)
            self.hits = 0
            self.misses = 0
            self.saved_queries = 0
    
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
    
    def get_batch_cache_path(self, type_qid, country_qid, limit, offset):
        """Get cache file path for a batch query."""
        if self.batch_cache_dir is None:
            return None
            
        cache_key = f"batch_{type_qid}_{country_qid}_{limit}_{offset}"
        
        # Handle both Path and string paths
        if isinstance(self.batch_cache_dir, Path):
            return self.batch_cache_dir / f"{cache_key}.pickle"
        else:
            return os.path.join(self.batch_cache_dir, f"{cache_key}.pickle")
    
    def get_cached_count(self, type_qid, country_qid):
        """
        Get a cached count result if available and not expired.
        
        Args:
            type_qid: Organization type QID
            country_qid: Country QID
            
        Returns:
            Count value if cached, None otherwise
        """
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
        """
        Cache a count query result.
        
        Args:
            type_qid: Organization type QID
            country_qid: Country QID
            count: Count value to cache
        """
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
            cache_path = self.get_batch_cache_path(type_qid, country_qid, limit, offset)
            
            if cache_path is None or not os.path.exists(cache_path):
                self.misses += 1
                return None
                
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                # Check if cache is expired
                cache_time = cached_data.get('timestamp')
                if not cache_time or datetime.now() - cache_time > self.cache_expiration:
                    # Cache expired
                    self.misses += 1
                    return None
                    
                # Valid cache
                self.hits += 1
                return cached_data.get('results')
                
            except Exception as e:
                # If any error occurs, treat as cache miss
                print(f"Error reading cache file {cache_path}: {e}")
                self.misses += 1
                return None
        except Exception as e:
            print(f"Error in get_cached_batch: {e}")
            self.misses += 1
            return None
    
    def cache_batch(self, type_qid, country_qid, limit, offset, results):
        """
        Cache a batch query result.
        
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
                
            cache_path = self.get_batch_cache_path(type_qid, country_qid, limit, offset)
            if cache_path is None:
                return
                
            cache_data = {
                'type_qid': type_qid,
                'country_qid': country_qid,
                'limit': limit,
                'offset': offset,
                'results': results,
                'timestamp': datetime.now()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            self.saved_queries += 1
            
        except Exception as e:
            print(f"Error caching batch: {e}")
            
    def clear_expired(self):
        """Clear expired cache entries to free up disk space."""
        if self.count_cache_dir is None or self.batch_cache_dir is None:
            return 0
            
        cleared_count = 0
        now = datetime.now()
        
        try:
            # Clear expired count caches
            try:
                count_files = list(Path(self.count_cache_dir).glob("*.json"))
            except Exception:
                # Fallback to os.listdir if pathlib has issues
                count_files = [os.path.join(self.count_cache_dir, f) 
                              for f in os.listdir(self.count_cache_dir) 
                              if f.endswith('.json')]
                
            for cache_file in count_files:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        
                    cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                    if now - cache_time > self.cache_expiration:
                        os.remove(cache_file)
                        cleared_count += 1
                except Exception:
                    # If the file is corrupted, remove it
                    try:
                        os.remove(cache_file)
                        cleared_count += 1
                    except:
                        pass
        except Exception as e:
            print(f"Error clearing count cache: {e}")
        
        try:
            # Clear expired batch caches
            try:
                batch_files = list(Path(self.batch_cache_dir).glob("*.pickle"))
            except Exception:
                # Fallback to os.listdir if pathlib has issues
                batch_files = [os.path.join(self.batch_cache_dir, f) 
                              for f in os.listdir(self.batch_cache_dir) 
                              if f.endswith('.pickle')]
                
            for cache_file in batch_files:
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        
                    cache_time = cached_data.get('timestamp')
                    if not cache_time or now - cache_time > self.cache_expiration:
                        os.remove(cache_file)
                        cleared_count += 1
                except Exception:
                    # If the file is corrupted, remove it
                    try:
                        os.remove(cache_file)
                        cleared_count += 1
                    except:
                        pass
        except Exception as e:
            print(f"Error clearing batch cache: {e}")
        
        return cleared_count
    
    def get_stats(self):
        """Get cache statistics."""
        if self.count_cache_dir is None or self.batch_cache_dir is None:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "saved_queries": self.saved_queries,
                "status": "Memory-only mode (no disk cache)"
            }
            
        try:
            # Count cache files
            try:
                count_files = len(list(Path(self.count_cache_dir).glob("*.json")))
            except Exception:
                # Fallback to os.listdir if pathlib has issues
                count_files = len([f for f in os.listdir(self.count_cache_dir) if f.endswith('.json')])
                
            try:
                batch_files = len(list(Path(self.batch_cache_dir).glob("*.pickle")))
            except Exception:
                # Fallback to os.listdir if pathlib has issues
                batch_files = len([f for f in os.listdir(self.batch_cache_dir) if f.endswith('.pickle')])
            
            # Estimate cache size
            total_size = 0
            try:
                # Use os.path.getsize which is more reliable
                for dirpath, dirnames, filenames in os.walk(self.cache_dir):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        total_size += os.path.getsize(fp)
            except Exception as e:
                print(f"Error calculating cache size: {e}")
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "saved_queries": self.saved_queries,
                "count_files": count_files,
                "batch_files": batch_files,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024) if total_size > 0 else 0,
                "cache_dir": str(self.cache_dir),
                "expiration_days": self.cache_expiration.days
            }
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {
                "hits": self.hits,
                "misses": self.misses,
                "saved_queries": self.saved_queries,
                "error": str(e)
            }


class WikidataDumpGenerator:

    def __init__(self, verbose=False, cache_dir=None):
        self.endpoint_url = WIKIDATA_SPARQL_ENDPOINT
        self.headers = {"User-Agent": "WikidataDumpGeneratorBot/1.0 (your-email@example.com)"}
        
        # Initialize cache for WikiData queries
        self.cache = WikiDataCache(cache_dir=cache_dir)
        
        # Initialize results cache directory
        self.results_cache_dir = None
        if cache_dir:
            self.results_cache_dir = os.path.join(cache_dir, 'wikidata_results')
            os.makedirs(self.results_cache_dir, exist_ok=True)
        elif hasattr(self.cache, 'cache_dir') and self.cache.cache_dir:
            self.results_cache_dir = os.path.join(str(self.cache.cache_dir), 'wikidata_results')
            os.makedirs(self.results_cache_dir, exist_ok=True)
        
        # Keep requests_cache for backward compatibility, but we'll primarily use our custom cache
        self.session = requests_cache.CachedSession(
            cache_name='sparql_cache',
            backend='sqlite',
            expire_after=86400
        )
        
        self.verbose = verbose
        if self.verbose:
            logger.info("Initializing WikidataDumpGenerator...")
        
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
    
    def _get_results_cache_key(self, countries, org_types):
        """Generate cache key for final results"""
        if isinstance(countries, list):
            countries_str = '_'.join(sorted(countries))
        else:
            countries_str = str(countries)
            
        if isinstance(org_types, list):
            org_types_str = '_'.join(sorted(org_types))
        else:
            org_types_str = str(org_types)
            
        return f"results_{countries_str}_{org_types_str}"

    def _save_results_cache(self, results_df, countries, org_types):
        """Save final results to cache"""
        if not self.results_cache_dir:
            return
            
        try:
            cache_key = self._get_results_cache_key(countries, org_types)
            cache_file = os.path.join(self.results_cache_dir, f"{cache_key}.pickle")
            
            cache_data = {
                'results': results_df,
                'countries': countries,
                'org_types': org_types,
                'timestamp': datetime.now().isoformat(),
                'total_organizations': len(results_df)
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            # Save metadata separately for easy inspection
            metadata_file = os.path.join(self.results_cache_dir, f"{cache_key}_metadata.json")
            metadata = {
                'countries': countries,
                'org_types': org_types,
                'timestamp': cache_data['timestamp'],
                'total_organizations': cache_data['total_organizations']
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            if self.verbose:
                logger.info(f"Saved {len(results_df)} organizations to results cache: {cache_key}")
                
        except Exception as e:
            logger.warning(f"Failed to save results cache: {e}")

    def _load_results_cache(self, countries, org_types):
        """Load final results from cache"""
        if not self.results_cache_dir:
            return None
            
        try:
            cache_key = self._get_results_cache_key(countries, org_types)
            cache_file = os.path.join(self.results_cache_dir, f"{cache_key}.pickle")
            
            if not os.path.exists(cache_file):
                return None
                
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Optionally check if cache is too old (30 days by default)
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(days=30):
                if self.verbose:
                    logger.info(f"Results cache expired for {cache_key}")
                return None
                
            if self.verbose:
                logger.info(f"Loaded {len(cache_data['results'])} organizations from results cache: {cache_key}")
                
            return cache_data['results']
            
        except Exception as e:
            logger.warning(f"Failed to load results cache: {e}")
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
        Resolve a country name or QID to a Wikidata QID.
        
        Args:
            country: Country name or QID
            
        Returns:
            Wikidata QID or None if not found
        """
        if country.startswith('Q') and country[1:].isdigit():
            return country
            
        name_to_qid = self.get_country_qids_from_wikidata()
        
        # Direct match
        if country in name_to_qid:
            return name_to_qid[country]
            
        # Try case-insensitive match
        for name, qid in name_to_qid.items():
            if name.lower() == country.lower():
                return qid
                
        # Try pycountry lookup
        try:
            country_obj = pycountry.countries.lookup(country)
            if country_obj.name in name_to_qid:
                return name_to_qid[country_obj.name]
            if hasattr(country_obj, 'official_name') and country_obj.official_name in name_to_qid:
                return name_to_qid[country_obj.official_name]
        except LookupError:
            pass
            
        return None

    def get_country_flag(self, country_name: str) -> str:
        """
        Get the flag emoji for a country.
        
        Args:
            country_name: Name of the country
            
        Returns:
            Flag emoji or empty string if not found
        """
        try:
            country = pycountry.countries.lookup(country_name)
            alpha_2 = country.alpha_2.upper()
            return ''.join(chr(127397 + ord(char)) for char in alpha_2)
        except LookupError:
            return ''

    def fetch_results(self, type_qid, country_qid):
        """
        Fetch organizations from Wikidata for a specific type and country.
        
        Args:
            type_qid: Wikidata QID for the organization type
            country_qid: Wikidata QID for the country
                
        Returns:
            List of organization dictionaries
        """
        if self.verbose:
            logger.info(f"fetch_results: Fetching data for type {type_qid} in country {country_qid}...")
        
        # Results will be collected here
        all_results = []
        
        # Batch processing parameters
        batch_size = 100  # Smaller batches are less likely to time out
        offset = 0
        max_batches = 20  # Limit total number of batches to prevent excessive queries
        
        # Skip count query if it's taking too long or causing issues
        # Just start with batch retrieval directly
        try:
            # Try to get from cache first with very short timeout
            total_count = self.cache.get_cached_count(type_qid, country_qid)
            
            if total_count is not None:
                if self.verbose:
                    logger.info(f"Using cached count: {total_count} organizations")
            else:
                # Count query with a strict timeout to avoid hanging
                count_query = f"""
                SELECT (COUNT(*) as ?count) WHERE {{
                    ?id wdt:P31 wd:{type_qid};
                        wdt:P17 wd:{country_qid}.
                }}
                """
                
                sparql = SPARQLWrapper(self.endpoint_url)
                sparql._session = self.session
                sparql.setQuery(count_query)
                sparql.setReturnFormat(JSON)
                sparql.setTimeout(TIMEOUT_COUNT_REQUEST)
                
                try:
                    count_results = sparql.query().convert()
                    total_count = int(count_results["results"]["bindings"][0]["count"]["value"])
                    
                    # Cache the count
                    self.cache.cache_count(type_qid, country_qid, total_count)
                    
                    if self.verbose:
                        logger.info(f"Found {total_count} organizations to fetch")
                except Exception as count_error:
                    if self.verbose:
                        logger.warning(f"Count query failed: {count_error}. Proceeding with batch retrieval.")
                    total_count = None
                    
        except Exception as e:
            if self.verbose:
                logger.warning(f"Error getting count: {e}. Proceeding with batch retrieval.")
            total_count = None
        
        # Batch retrieval loop
        batch_num = 1
        while batch_num <= max_batches:
            try:
                # Check cache first
                batch_results = self.cache.get_cached_batch(type_qid, country_qid, batch_size, offset)
                
                if batch_results is not None:
                    if self.verbose:
                        logger.info(f"Using cached batch at offset {offset}")
                else:
                    # Fetch from Wikidata
                    try:
                        batch_results = self._fetch_batch_with_retry(type_qid, country_qid, batch_size, offset)
                        
                        # Cache the batch result if successful
                        if batch_results:
                            try:
                                self.cache.cache_batch(type_qid, country_qid, batch_size, offset, batch_results)
                            except Exception as cache_error:
                                if self.verbose:
                                    logger.warning(f"Error caching batch: {cache_error}")
                                    
                    except Exception as batch_error:
                        if self.verbose:
                            logger.error(f"Error during batch retrieval: {batch_error}")
                        batch_results = []
                
                # If no results returned, we've reached the end
                if not batch_results:
                    if self.verbose:
                        logger.info(f"No more results at offset {offset}. Completed retrieval.")
                    break
                    
                # Add batch results to total results
                all_results.extend(batch_results)
                
                if self.verbose:
                    logger.info(f"Batch {batch_num}: Retrieved {len(batch_results)} organizations (offset {offset})")
                
                # Update offset for next batch
                offset += len(batch_results)  # Use actual count rather than batch_size
                batch_num += 1
                
                # If we know the total and have fetched all items, stop
                if total_count is not None and len(all_results) >= total_count:
                    if self.verbose:
                        logger.info(f"Retrieved all {len(all_results)} organizations. Completed retrieval.")
                    break
                    
                # Optional delay between batches to reduce load on server
                time.sleep(1)
                
            except Exception as e:
                # Unexpected error in batch processing
                logger.error(f"Unexpected error in batch processing: {e}")
                
                # Try to continue with next batch
                offset += batch_size
                batch_num += 1
        
        if self.verbose:
            logger.info(f"Total organizations fetched: {len(all_results)}")
            
            # Show cache statistics
            try:
                cache_stats = self.cache.get_stats()
                logger.info(f"Cache statistics: {cache_stats['hits']} hits, {cache_stats['misses']} misses")
            except Exception as stats_error:
                logger.warning(f"Error getting cache stats: {stats_error}")
            
        return all_results

    def _fetch_batch_with_retry(self, type_qid, country_qid, limit=100, offset=0, max_retries=MAX_NUM_RETRIES):
        """
        Fetch a single batch of organizations from Wikidata with retries.
        
        Args:
            type_qid: Organization type QID
            country_qid: Country QID
            limit: Number of results to fetch
            offset: Starting offset
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of organization dictionaries
        """
        query = f"""
        SELECT ?id ?name_en ?name_full ?alias ?acronym ?ror ?grid ?lei ?crunchbase ?website ?coordinates
               ?formation_label ?city ?region ?country_name ?country_code ?parent_name 
               ?hq_city ?hq_country_name ?fallback_country_label ?fallback_country_code
        WHERE {{
            ?id wdt:P31 wd:{type_qid};
                wdt:P17 wd:{country_qid}.

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
        }}
        LIMIT {limit}
        OFFSET {offset}
        """
        
        for attempt in range(max_retries):
            try:
                sparql = SPARQLWrapper(self.endpoint_url)
                sparql._session = self.session
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                sparql.setTimeout(TIMEOUT_BATCH_REQUEST)
                
                results = sparql.query().convert()
                
                # Process results
                organizations = []
                for result in results["results"]["bindings"]:
                    org_data = {
                        "id": result["id"]["value"].split("/")[-1],
                        "name_en": result.get("name_en", {}).get("value", ""),
                        "name_full": result.get("name_full", {}).get("value", ""),
                        "alias": result.get("alias", {}).get("value", ""),
                        "acronym": result.get("acronym", {}).get("value", ""),
                        "ror": result.get("ror", {}).get("value", ""),
                        "grid": result.get("grid", {}).get("value", ""),
                        "lei": result.get("lei", {}).get("value", ""),
                        "crunchbase": result.get("crunchbase", {}).get("value", ""),
                        "website": result.get("website", {}).get("value", ""),
                        "coordinates": result.get("coordinates", {}).get("value", ""),
                        "formation_location": result.get("formation_label", {}).get("value", ""),
                        "city": result.get("city", {}).get("value", ""),
                        "region": result.get("region", {}).get("value", ""),
                        "country_name": result.get("country_name", {}).get("value", 
                                                  result.get("fallback_country_label", {}).get("value", "")),
                        "country_code": result.get("country_code", {}).get("value", 
                                                  result.get("fallback_country_code", {}).get("value", "")),
                        "parent_name": result.get("parent_name", {}).get("value", ""),
                        "hq_city": result.get("hq_city", {}).get("value", ""),
                        "hq_country_name": result.get("hq_country_name", {}).get("value", "")
                    }
                    organizations.append(org_data)
                
                return organizations
                
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = DELAY_BASE_RETRY * (2 ** attempt)  # Exponential backoff
                    if self.verbose:
                        logger.warning(f"Error fetching batch at offset {offset}. Retry {attempt + 1}/{max_retries} after {delay}s delay: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch batch after {max_retries} attempts: {e}")
                    raise
        
        return []

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
        # Process parameters
        processed_countries = self._process_countries_param(countries)
        processed_org_types = self._process_org_types_param(org_types)
        
        # Try to load from results cache first
        cached_results = self._load_results_cache(processed_countries, processed_org_types)
        if cached_results is not None:
            if self.verbose:
                logger.info("Using cached WikiData results - skipping data collection")
            return cached_results
        
        # If no cache, proceed with normal data collection
        if self.verbose:
            logger.info("No cached results found - collecting from WikiData")
        
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
                # Try to resolve as a single type name
                if org_types in self.organisation_types_extended:
                    org_type_qids = [self.organisation_types_extended[org_types]]
                elif org_types in self.organisation_types_short:
                    org_type_qids = [self.organisation_types_short[org_types]]
                else:
                    logger.warning(f"Unknown organization type: {org_types}")
                    org_type_qids = []
        elif isinstance(org_types, list):
            for org_type in org_types:
                if org_type in self.organisation_types_extended:
                    org_type_qids.append(self.organisation_types_extended[org_type])
                elif org_type in self.organisation_types_short:
                    org_type_qids.append(self.organisation_types_short[org_type])
                else:
                    logger.warning(f"Unknown organization type: {org_type}")
        elif org_types is None:
            org_type_qids = list(self.organisation_types_short.values())

        if self.verbose:
            logger.info(f"Processing {len(country_qids)} countries and {len(org_type_qids)} organization types")
            logger.info(f"Organization type QIDs: {org_type_qids}")

        # Collect all organizations
        all_organizations = []
        
        # Use tqdm for progress tracking
        total_combinations = len(country_qids) * len(org_type_qids)
        with tqdm(total=total_combinations, desc="Fetching WikiData organizations") as pbar:
            for country_qid in country_qids:
                country_name = qid_to_country_name.get(country_qid, country_qid)
                country_flag = self.get_country_flag(country_name)
                
                for org_type_qid in org_type_qids:
                    org_type_name = self.org_type_map_extended.get(org_type_qid, org_type_qid)
                    
                    pbar.set_description(f"Processing {country_flag} {country_name} - {org_type_name}")
                    
                    if self.verbose:
                        logger.info(f"Fetching data for type {org_type_name} ({org_type_qid}) in country {country_name} ({country_qid})...")
                    
                    try:
                        organizations = self.fetch_results(org_type_qid, country_qid)
                        
                        # Add metadata to each organization
                        for org in organizations:
                            org['org_type'] = org_type_name
                            org['org_type_qid'] = org_type_qid
                            org['query_country'] = country_name
                            org['query_country_qid'] = country_qid
                            org['country_flag'] = country_flag
                        
                        all_organizations.extend(organizations)
                        
                        if self.verbose:
                            logger.info(f"Retrieved {len(organizations)} results for {country_name}/{org_type_name}")
                            
                    except Exception as e:
                        logger.error(f"Error fetching data for {country_name}/{org_type_name}: {e}")
                    
                    pbar.update(1)

        # Create DataFrame
        if all_organizations:
            df = pd.DataFrame(all_organizations)
            
            # Clean up the DataFrame
            df = df.drop_duplicates(subset=['id'])
            df = df.reset_index(drop=True)
            
            if self.verbose:
                logger.info(f"Created DataFrame with {len(df)} unique organizations")
        else:
            # Create empty DataFrame with expected columns
            df = pd.DataFrame(columns=[
                'id', 'name_en', 'name_full', 'alias', 'acronym', 'ror', 'grid', 'lei', 
                'crunchbase', 'website', 'coordinates', 'formation_location', 'city', 
                'region', 'country_name', 'country_code', 'parent_name', 'hq_city', 
                'hq_country_name', 'org_type', 'org_type_qid', 'query_country', 
                'query_country_qid', 'country_flag'
            ])
            
            if self.verbose:
                logger.info("No organizations found, created empty DataFrame")
        
        # Save results to cache before returning
        self._save_results_cache(df, processed_countries, processed_org_types)
        
        return df
    
    def _process_countries_param(self, countries):
        """Process and normalize the countries parameter for caching"""
        if isinstance(countries, list):
            return sorted(countries)
        elif isinstance(countries, str):
            return countries
        elif countries is None:
            return 'all'
        else:
            return str(countries)
    
    def _process_org_types_param(self, org_types):
        """Process and normalize the org_types parameter for caching"""
        if isinstance(org_types, list):
            return sorted(org_types)
        elif isinstance(org_types, str):
            return org_types
        elif org_types is None:
            return 'short'
        else:
            return str(org_types)
