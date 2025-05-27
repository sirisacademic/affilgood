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
                home_dir = os.path.expanduser("~")
                cache_dir = os.path.join(home_dir, ".wikidata_cache")
            
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

    def __init__(self, verbose=False):
        self.endpoint_url = WIKIDATA_SPARQL_ENDPOINT
        self.headers = {"User-Agent": "WikidataDumpGeneratorBot/1.0 (your-email@example.com)"}
        
        # Initialize cache for WikiData queries
        self.cache = WikiDataCache()
        
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
            country: Country name or QID (e.g., 'United States' or 'Q30')
            
        Returns:
            Wikidata QID for the country
        """
        # If it's already a QID (starts with Q), return it
        if isinstance(country, str) and country.startswith('Q'):
            return country
            
        # Get the map of country names to QIDs
        country_map = self.get_country_qids_from_wikidata()
        
        # Try direct lookup
        if country in country_map:
            return country_map[country]
            
        # Try case-insensitive lookup
        for name, qid in country_map.items():
            if name.lower() == country.lower():
                return qid
                
        # Try with pycountry
        try:
            country_obj = pycountry.countries.lookup(country)
            country_name = country_obj.name
            
            # Try with the official name
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
                
                # Set SPARQL endpoint for count query with strict timeout
                sparql = SPARQLWrapper(WIKIDATA_SPARQL_ENDPOINT)
                sparql.setTimeout(10)  # Very short timeout for count query
                sparql.setQuery(count_query)
                sparql.setReturnFormat(JSON)
                
                try:
                    # Use a threading approach with timeout to avoid hanging
                    import threading
                    import queue
                    
                    result_queue = queue.Queue()
                    
                    def execute_query():
                        try:
                            count_results = sparql.query().convert()
                            result_queue.put(int(count_results['results']['bindings'][0]['count']['value']))
                        except Exception as e:
                            result_queue.put(None)
                    
                    # Start query in a thread
                    query_thread = threading.Thread(target=execute_query)
                    query_thread.daemon = True
                    query_thread.start()
                    
                    # Wait for result with timeout
                    try:
                        total_count = result_queue.get(timeout=15)  # 15 second timeout
                        if total_count is not None:
                            # Cache the count
                            self.cache.cache_count(type_qid, country_qid, total_count)
                            
                            if self.verbose:
                                logger.info(f"Found {total_count} organizations to fetch")
                            
                            # If small enough, fetch in one go to avoid multiple queries
                            if total_count <= batch_size:
                                batch_results = self._fetch_batch_with_retry(type_qid, country_qid, limit=total_count, offset=0)
                                return batch_results if batch_results else []
                    except queue.Empty:
                        if self.verbose:
                            logger.warning("Count query timed out. Proceeding with batched retrieval.")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Count query failed: {e}. Proceeding with batched retrieval.")
        except Exception as e:
            # If anything goes wrong with count, just proceed with batching
            if self.verbose:
                logger.warning(f"Count query exception: {e}. Proceeding without count information.")
        
        # Process in batches with timeouts to avoid hanging
        batch_num = 1
        
        while batch_num <= max_batches:
            try:
                # Check if batch is in cache first
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
                    # Not in cache, fetch from WikiData with timeout protection
                    try:
                        # Use a threading approach with timeout to avoid hanging
                        import threading
                        import queue
                        
                        result_queue = queue.Queue()
                        
                        def execute_batch_fetch():
                            try:
                                results = self._fetch_batch_with_retry(type_qid, country_qid, limit=batch_size, offset=offset)
                                result_queue.put(results)
                            except Exception as e:
                                result_queue.put([])
                        
                        # Start query in a thread
                        query_thread = threading.Thread(target=execute_batch_fetch)
                        query_thread.daemon = True
                        query_thread.start()
                        
                        # Wait for result with timeout
                        try:
                            batch_results = result_queue.get(timeout=60)  # 60 second timeout for batch
                        except queue.Empty:
                            if self.verbose:
                                logger.warning(f"Batch query at offset {offset} timed out. Proceeding to next batch.")
                            batch_results = []
                    except Exception as batch_error:
                        if self.verbose:
                            logger.error(f"Error during batch retrieval: {batch_error}")
                        batch_results = []
                    
                    # Cache the batch result if successful
                    if batch_results:
                        try:
                            self.cache.cache_batch(type_qid, country_qid, batch_size, offset, batch_results)
                        except Exception as cache_error:
                            if self.verbose:
                                logger.warning(f"Error caching batch: {cache_error}")
                
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

    def _fetch_batch_with_retry(self, type_qid, country_qid, limit=100, offset=0, max_retries=2):
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
        delay_base = 3  # Base delay in seconds
        
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

    def _fetch_batch(self, type_qid, country_qid, limit=100, offset=0):
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
        sparql.setTimeout(30)  # 30 seconds timeout
        
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
            'name': row['name_en'],
            'aliases': row['aliases'],
            'all_names': row['all_names'],
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
                                                if k in row and pd.notna(row[k])}, 
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

            return expanded_df
        except Exception as final_error:
            logger.error(f"Error in final processing: {final_error}")
            import traceback
            traceback.print_exc()
            # Return an empty DataFrame instead of None to avoid ambiguity
            return pd.DataFrame()
