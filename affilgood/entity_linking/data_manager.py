import os
import sys
import boto3
import gzip
import json
import requests
import zipfile
import pandas as pd
import logging

from io import BytesIO
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime, timedelta
from unidecode import unidecode
from pathlib import Path

from .constants import *
from .plugins import DataSourceRegistry
from .utils.text_utils import get_variants_list
from .utils.translation_mappings import translate_institution_name

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Ensure S2AFF is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH)))
from s2aff.consts import PATHS



class DataManager:
    """Manages data updates, S3 syncing, and file resolution for entity linking."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.data_s2aff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH, "data"))
        if not os.path.exists(self.data_s2aff_path):
            logger.info(f"Creating directory: {self.data_s2aff_path}")
            os.makedirs(self.data_s2aff_path, exist_ok=True)
            
        # Initialize data source mapping functions
        # Default built-in mappers
        self._source_mappers = {
            'ror': self._map_ror_organization,
            'wikidata': self._map_wikidata_organization,
        }
        
        # Add mappers from registered handlers
        for source_id, handler in DataSourceRegistry.get_all_handlers().items():
            self._source_mappers[source_id] = handler.map_organization
        
        # Create base directories for indices
        self.whoosh_indices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'whoosh_indices'))
        self.hnsw_indices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hnsw_indices'))
        os.makedirs(self.whoosh_indices_path, exist_ok=True)
        os.makedirs(self.hnsw_indices_path, exist_ok=True)
        
        # Lang codes for country mapping
        lang_codes_df = pd.read_csv(COUNTRY_LANGS_FILE, sep='\t').fillna("")
        self.lang_lookup = lang_codes_df.set_index('country_exonym')['lang_codes'].str.split('|').to_dict()

    def get_s3_client(self):
        """Returns an S3 client with unsigned configuration."""
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def list_s3_objects(self, bucket, prefix):
        """Lists objects in an S3 bucket with a given prefix."""
        s3 = self.get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        response_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        return [obj for page in response_iterator if "Contents" in page for obj in page["Contents"]]

    def should_skip_s3_key(self, s3_key):
        """Determines if an S3 key should be skipped based on OMIT_S2AFF substrings."""
        if not s3_key:
            return False  # If the key is None, it cannot match OMIT_S2AFF
        return any(substring in s3_key for substring in OMIT_S2AFF)

    def download_file(self, url_or_bucket, key_or_path, local_path, is_s3=False, omit_s2aff=False):
        """
        Downloads a file from S3 or a URL to a local path, skipping if it matches OMIT_S2AFF.
        """
        # Check for omission
        if omit_s2aff and self.should_skip_s3_key(key_or_path):
            logger.info(f"Skipping download of {key_or_path} due to OMIT_S2AFF rules.")
            return
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            if is_s3:
                # Validate key_or_path
                if not key_or_path:
                    logger.warning(f"Invalid key or path: {key_or_path}. Skipping download.")
                    return
                logger.info(f"Downloading from S3: {key_or_path} to {local_path}")
                s3 = self.get_s3_client()
                s3.download_file(url_or_bucket, key_or_path, local_path)
            else:
                # Check if file already exists
                if os.path.exists(local_path):
                    logger.info(f"Using existing local file {local_path}")
                    return
                # Validate url_or_bucket
                if not url_or_bucket:
                    logger.warning(f"Invalid url: {url_or_bucket}. Skipping download.")
                    return
                logger.info(f"Downloading from URL: {url_or_bucket} to {local_path}")
                response = requests.get(url_or_bucket)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            logger.error(f"Failed to download {key_or_path} to {local_path}: {e}")

    def resolve_s2aff_paths(self):
        """Converts missing file paths in PATHS to their corresponding URLs."""
        for key, path in PATHS.items():
            if not os.path.exists(path):
                logger.info(f"File '{path}' does not exist. Resolving to URL.")
                PATHS[key] = f"https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2aff-release/{os.path.basename(path)}"

    def ensure_s2aff_files(self):
        """Ensures all required files in PATHS exist locally."""
        for key, path in PATHS.items():
            if path.startswith("http"):
                local_path = os.path.join(self.data_s2aff_path, os.path.basename(path))
                if not os.path.exists(local_path):
                    logger.info(f"Ensuring file: {key}, URL: {path}")
                    self.download_file(path, os.path.basename(path), local_path, omit_s2aff=True)
            elif not os.path.exists(path):
                logger.error(f"File {path} is missing but not resolvable to a URL.")
                raise FileNotFoundError(f"Required file {key} not found at {path}.")
        # Update OpenAlex work counts
        self.update_openalex_works_counts()

    def format_id_url(self, org_id, source):
        """
        Format an organization ID into a proper URL based on the data source.
        
        Args:
            org_id: The organization identifier
            source: The data source ('ror', 'wikidata', or plugin source)
            
        Returns:
            str: Formatted URL
        """
        # Try to get handler for this source
        handler = DataSourceRegistry.get_handler(source)
        if handler:
            return handler.format_id_url(org_id)
        
        # Fallback to built-in sources
        if source == 'ror':
            # Handle ROR IDs - ensure they have the correct format
            if 'ror.org' in org_id:
                return org_id
            else:
                return f"{ROR_URL}{org_id}"
        elif source == 'wikidata':
            # Handle WikiData IDs
            if 'wikidata.org' in org_id:
                return org_id
            else:
                return f"{WIKIDATA_URL}{org_id}"
        else:
            # Default fallback - just return the ID as is
            return org_id

    def register_source_mapper(self, source, mapper_function):
        """
        Register a new data source mapper function.
        
        Args:
            source: String identifier for the data source
            mapper_function: Function that maps source-specific data to standard fields
        """
        self._source_mappers[source] = mapper_function

    def get_or_create_index(self, source, indices_type, org_types=None, countries=None, 
                           force_rebuild=False, encoder_path=None, use_wikidata_labels_with_ror=False, **kwargs):
        """
        Get or create an index for a specific data source.
        
        Args:
            source: Data source identifier ('ror', 'wikidata', or plugin source)
            indices_type: Type of index to create ('whoosh' or 'hnsw')
            org_types: Organization types to include (for filtering WikiData)
            countries: Countries to include (for filtering WikiData)
            force_rebuild: Whether to force rebuilding the index
            encoder_path: Path to encoder model (for HNSW indices)
            use_wikidata_labels_with_ror: Whether to enrich ROR organizations with WikiData labels
            **kwargs: Additional parameters for plugin handlers
            
        Returns:
            str: Path to the index
        """
        # Build configuration dictionary for handlers
        config = kwargs.get('config', {})
        # Add org_types and countries to config if provided
        if org_types is not None:
            config['org_types'] = org_types
        if countries is not None:
            config['countries'] = countries
        if 'encoder_path' not in config and encoder_path is not None:
            config['encoder_path'] = encoder_path
        # Add use_wikidata_labels_with_ror to config
        config['use_wikidata_labels_with_ror'] = use_wikidata_labels_with_ror
        
        # Get standard index path for built-in sources
        index_path = None
        index_id = None
        
        # Try to get handler for this source
        handler = DataSourceRegistry.get_handler(source)
        if handler:
            # Get data from the handler
            try:
                org_data, index_id = handler.get_data_for_indexing(
                    config,
                    indices_type=indices_type,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error getting data from handler {source}: {e}")
                return None
                
            # If we have an index ID, construct the full path
            if index_id:
                if indices_type == 'whoosh':
                    index_path = os.path.join(self.whoosh_indices_path, index_id)
                elif indices_type == 'hnsw':
                    index_path = os.path.join(self.hnsw_indices_path, index_id)
        else:
            # For built-in sources, use standard path calculation
            index_path = self.get_index_path(source, indices_type, org_types, countries)
        
        if not index_path:
            logger.error(f"Failed to determine index path for {source}")
            return None
        
        if self.verbose:
            logger.info(f"DataManager.get_or_create_index: index_path={index_path}")
        
        # Check if index exists and is valid
        if not force_rebuild:
            if indices_type == 'whoosh':
                if os.path.exists(index_path) and len(os.listdir(index_path)) > 0:
                    logger.info(f"Using existing {source} {indices_type} index: {index_path}")
                    return index_path
            elif indices_type == 'hnsw':
                if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "org_index.bin")):
                    logger.info(f"Using existing {source} {indices_type} index: {index_path}")
                    return index_path
        
        # Index doesn't exist or forced rebuild, create it
        logger.info(f"Creating {source} {indices_type} index: {index_path}")
        
        # Get organization data if not already obtained from handler
        if handler and org_data is not None:
            # Data already obtained from handler
            pass
        elif source == 'ror':
            # For ROR, get the latest dump
            ror_file = self.get_latest_ror()
            if not ror_file:
                logger.error(f"Failed to get ROR data for index creation")
                return None
                
            # Load ROR data
            with open(ror_file, 'r', encoding='utf-8') as f:
                org_data = json.load(f)
        elif source == 'wikidata':
            # For WikiData, use the existing retrieval function
            org_data = self.get_wikidata_organizations(org_types, countries)
        else:
            logger.error(f"Unknown source: {source}")
            return None
        
        # Check if we have data to index
        if org_data is None or (isinstance(org_data, pd.DataFrame) and org_data.empty):
            logger.error(f"No data available for {source} index creation")
            return None
        
        # Create the index
        success = False
        if indices_type == 'whoosh':
            success = self.create_whoosh_index(
                org_data, 
                index_path, 
                source=source, 
                use_wikidata_labels_with_ror=use_wikidata_labels_with_ror
            )
        elif indices_type == 'hnsw':
            # Pass encoder_path and use_wikidata_labels_with_ror to create_hnsw_index
            success = self.create_hnsw_index(
                org_data, 
                index_path, 
                source=source, 
                encoder_path=encoder_path if encoder_path else ENCODER_DEFAULT_MODEL,
                use_wikidata_labels_with_ror=use_wikidata_labels_with_ror
            )
        else:
            logger.error(f"Unknown indices type: {indices_type}")
            return None
        
        if not success:
            logger.error(f"Failed to create {source} {indices_type} index")
            return None
        
        return index_path


    # ROR DATA MANAGEMENT
    def get_local_ror_dumps(self):
        """Returns a sorted list of local ROR dump files."""
        files = [os.path.join(self.data_s2aff_path, f) for f in os.listdir(self.data_s2aff_path) if "ror-data.json" in f]
        return sorted(files, reverse=True)

    def get_latest_ror(self, ror_dump_path=None):
        """Fetches the latest ROR dump or returns the most recent local one."""
        if ror_dump_path is None and ROR_DUMP_PATH:
            ror_dump_path = ROR_DUMP_PATH
            if os.path.exists(ror_dump_path):
                logger.info(f"Using existing ROR dump from {ror_dump_path}")
                return ror_dump_path
            elif os.path.exists(os.path.join(self.data_s2aff_path, ror_dump_path)):
                return os.path.join(self.data_s2aff_path, ror_dump_path)
            else:
                logger.warning(f"ROR dump not found in {ror_dump_path} nor {os.path.join(self.data_s2aff_path, ror_dump_path)}.")
        logger.info(f"Fetching latest ROR dump from {ROR_DUMP_LINK}")
        try:
            response = requests.get(ROR_DUMP_LINK)
            response.raise_for_status()
            download_url = response.json()["hits"]["hits"][0]["files"][0]["links"]["self"]
            file_name = response.json()["hits"]["hits"][0]["files"][0]["key"]
            file_path = os.path.join(self.data_s2aff_path, file_name)
            self.download_file(download_url, file_path, file_path)
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(self.data_s2aff_path)
            return self.get_local_ror_dumps()[0]
        except Exception as e:
            logger.error(f"Failed to fetch ROR dump: {e}")
            local_dumps = self.get_local_ror_dumps()
            if local_dumps:
                logger.info(f"Using existing ROR dump: {local_dumps[0]}")
                return local_dumps[0]
            else:
                return None

    # WIKIDATA MANAGEMENT
    def get_index_path(self, source, indices_type, org_types=None, countries=None):
        """
        Get path for an index based on source, type, and filters.
        
        Args:
            source: Data source ('ror', 'wikidata')
            indices_type: Type of indices ('whoosh', 'hnsw')
            org_types: Organization types to include ('all', 'short', 'extended', or specific list)
            countries: Countries to include ('all' or specific list)
            
        Returns:
            Path to the index directory
        """
        # Handle default parameters
        if org_types is None:
            org_types = 'all'
        if countries is None:
            countries = 'all'
            
        # Format org_types string
        if isinstance(org_types, list):
            org_types_str = '_'.join(org_types)
        else:
            org_types_str = org_types
        
        # Format countries string
        if isinstance(countries, list):
            countries_str = '_'.join(countries) 
        else:
            countries_str = countries
        
        # Build index name
        index_name = f"{source}_{org_types_str}_{countries_str}"
        
        # Get base path depending on index type
        if indices_type == 'whoosh':
            base_path = self.whoosh_indices_path
        elif indices_type == 'hnsw':
            base_path = self.hnsw_indices_path
        else:
            raise ValueError(f"Unknown indices type: {indices_type}")
        
        # Return full path
        return os.path.join(base_path, index_name)
    
    def get_index_metadata(self, index_path):
        """
        Get metadata for an index.
        
        Args:
            index_path: Path to the index directory
            
        Returns:
            Dictionary of metadata or None if not found
        """
        metadata_path = os.path.join(index_path, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading index metadata: {e}")
                return None
        return None
    
    def save_index_metadata(self, index_path, metadata):
        """
        Save metadata for an index.
        
        Args:
            index_path: Path to the index directory
            metadata: Dictionary of metadata to save
        """
        os.makedirs(index_path, exist_ok=True)
        metadata_path = os.path.join(index_path, 'metadata.json')
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving index metadata: {e}")
            return False
    
    def index_exists(self, source, indices_type, org_types=None, countries=None):
        """
        Check if an index exists.
        
        Args:
            source: Data source ('ror', 'wikidata')
            indices_type: Type of indices ('whoosh', 'hnsw')
            org_types: Organization types to include
            countries: Countries to include
            
        Returns:
            Boolean indicating if the index exists
        """
        index_path = self.get_index_path(source, indices_type, org_types, countries)
        if indices_type == 'whoosh':
            # For Whoosh, check if directory has files
            return os.path.exists(index_path) and len(os.listdir(index_path)) > 0
        elif indices_type == 'hnsw':
            # For HNSW, check for specific files
            index_file = os.path.join(index_path, "org_index.bin")
            meta_file = os.path.join(index_path, "org_index_meta.json")
            return os.path.exists(index_file) and os.path.exists(meta_file)
        return False
    
    def list_available_indices(self, indices_type=None):
        """
        List all available indices.
        
        Args:
            indices_type: Optional filter for type ('whoosh', 'hnsw')
            
        Returns:
            Dictionary of available indices with metadata
        """
        indices = {}
        
        # Determine which directories to scan
        if indices_type == 'whoosh' or indices_type is None:
            if os.path.exists(self.whoosh_indices_path):
                for index_name in os.listdir(self.whoosh_indices_path):
                    index_path = os.path.join(self.whoosh_indices_path, index_name)
                    if os.path.isdir(index_path) and os.listdir(index_path):
                        metadata = self.get_index_metadata(index_path)
                        indices[f"whoosh/{index_name}"] = metadata or {"path": index_path}
        
        if indices_type == 'hnsw' or indices_type is None:
            if os.path.exists(self.hnsw_indices_path):
                for index_name in os.listdir(self.hnsw_indices_path):
                    index_path = os.path.join(self.hnsw_indices_path, index_name)
                    if os.path.isdir(index_path) and os.path.exists(os.path.join(index_path, "org_index.bin")):
                        metadata = self.get_index_metadata(index_path)
                        indices[f"hnsw/{index_name}"] = metadata or {"path": index_path}
        
        return indices

    def should_update_openalex(self, days_threshold=UPDATE_OPENALEX_WORK_COUNTS_OLDER_THAN):
        """Checks if OpenAlex work counts need updating."""
        path = PATHS.get("openalex_works_counts", "")
        if not os.path.exists(path):
            logger.info(f"File '{path}' not found. Updating...")
            return True
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        return age > timedelta(days=days_threshold)
   
    def update_openalex_works_counts(self):
        """Updates OpenAlex work counts."""
        if self.should_update_openalex():
            bucket = "openalex"
            prefix = "data/institutions/"
            s3 = self.get_s3_client()
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            works_count_dict = {}
            for page in pages:
                for obj in page["Contents"]:
                    if obj["Key"].startswith("part") and obj["Key"].endswith(".gz"):
                        if self.verbose:
                            logger.info(f"Processing {obj}")
                        obj = s3.get_object(Bucket=bucket, Key=obj["Key"])
                        with gzip.GzipFile(fileobj=BytesIO(obj["Body"].read())) as f:
                            for line in f:
                                line = json.loads(line)
                                ror_id = line["ror"]
                                works_count = line["works_count"]
                                works_count_dict[ror_id] = works_count
            # Convert works_count_dict to a dataframe and save it
            df = pd.DataFrame.from_dict(works_count_dict, orient="index", columns=["works_count"]).reset_index()
            df.columns = ["ror", "works_count"]
            df.to_csv(PATHS["openalex_works_counts"], index=False)
            logger.info(f"Updated works counts saved to {PATHS['openalex_works_counts']}")

    # WIKIDATA METHODS
    def get_wikidata_organizations(self, org_types=None, countries=None):
        """
        Get organizations from WikiData with specified filters.
        
        Args:
            org_types: Organization types to include
            countries: Countries to include
            
        Returns:
            DataFrame of organizations
        """
        # Create WikidataDumpGenerator to get WikiData organizations
        from .wikidata_dump_generator import WikidataDumpGenerator
        
        indexer = WikidataDumpGenerator(verbose=self.verbose)
        
        try:
            # Get WikiData organizations
            orgs_df = indexer.get_index(countries=countries, org_types=org_types)
            
            if orgs_df is None or orgs_df.empty:
                logger.warning("No organizations found in WikiData with the specified filters")
                return None
                
            logger.info(f"Retrieved {len(orgs_df)} organizations from WikiData")
            return orgs_df.fillna("")
            
        except Exception as e:
            logger.error(f"Error retrieving WikiData organizations: {e}")
            return None

    # WHOOSH INDEX METHODS
    def _map_ror_organization(self, org):
        """
        Map a ROR organization to standard fields.
        
        Args:
            org: ROR organization dictionary
            
        Returns:
            dict: Standardized organization fields
        """
        doc = {}
        
        # Core fields
        doc['id'] = org.get('id', '')
        doc['name'] = org.get('name', '')
        doc['name_length'] = len(doc['name'].split()) if doc['name'] else 0
        
        # Get aliases, labels
        doc['aliases'] = org.get('aliases', [])
        doc['labels'] = [label['label'] for label in org.get('labels', [])] if 'labels' in org else []
        doc['acronyms'] = org.get('acronyms', [])
        
        # Location information
        if 'addresses' in org and len(org['addresses']) > 0:
            address = org['addresses'][0]
            doc['city'] = address.get('city', '')
            
            # Try to get region from geonames
            if 'geonames_city' in address and 'geonames_admin1' in address['geonames_city']:
                doc['region'] = address['geonames_city']['geonames_admin1'].get('name', '')
        else:
            doc['city'] = ''
            doc['region'] = ''
            
        # Country information
        if 'country' in org:
            doc['country'] = org['country'].get('country_code', '')
            doc['country_name'] = org['country'].get('country_name', '')
        else:
            doc['country'] = ''
            doc['country_name'] = ''
        
        # Parent information
        doc['parent_organizations'] = [r['label'] for r in org.get('relationships', []) 
                                    if r.get('type') == 'Parent']
        doc['parent'] = ' '.join(doc['parent_organizations'])
        
        return doc
            
    def _map_wikidata_organization(self, org):
        """
        Map a WikiData organization to standard fields.
        
        Args:
            org: WikiData organization dictionary
            
        Returns:
            dict: Standardized organization fields
        """
        doc = {}
        
        # Core fields
        doc['id'] = org.get('id', '')
        doc['name'] = org.get('name', '')
        doc['name_length'] = len(doc['name'].split()) if doc['name'] else 0
        
        # Process aliases
        if 'aliases' in org:
            if isinstance(org['aliases'], list):
                doc['aliases'] = org['aliases']
            elif isinstance(org['aliases'], dict):
                doc['aliases'] = list(org['aliases'].values())
            elif isinstance(org['aliases'], str):
                try:
                    doc['aliases'] = json.loads(org['aliases'])
                except:
                    doc['aliases'] = org['aliases'].split('|')
            else:
                doc['aliases'] = []
        else:
            doc['aliases'] = []
            
        # Process all_names (labels)
        if 'all_names' in org:
            if isinstance(org['all_names'], list):
                doc['labels'] = org['all_names']
            elif isinstance(org['all_names'], dict):
                doc['labels'] = list(org['all_names'].values())
            elif isinstance(org['all_names'], str):
                try:
                    doc['labels'] = json.loads(org['all_names'])
                except:
                    doc['labels'] = org['all_names'].split('|')
            else:
                doc['labels'] = []
        else:
            doc['labels'] = []
            
        # Process acronyms
        if 'acronyms' in org:
            if isinstance(org['acronyms'], list):
                doc['acronyms'] = org['acronyms']
            elif isinstance(org['acronyms'], str):
                try:
                    doc['acronyms'] = json.loads(org['acronyms'])
                except:
                    doc['acronyms'] = org['acronyms'].split('|')
            else:
                doc['acronyms'] = []
        else:
            doc['acronyms'] = []
            
        # Location information
        doc['city'] = org.get('city', '')
        doc['region'] = org.get('region', '')
        doc['country'] = org.get('country_code', '')
        doc['country_name'] = org.get('country_name', '')
        
        # Parent information
        if 'relationships' in org:
            if isinstance(org['relationships'], str):
                doc['parent'] = org['relationships']
                doc['parent_organizations'] = org['relationships'].split('|')
            else:
                doc['parent'] = str(org['relationships'])
                doc['parent_organizations'] = [str(org['relationships'])]
        else:
            doc['parent'] = ''
            doc['parent_organizations'] = []
        
        return doc
            
    def process_organization_for_index(self, org, source='ror', use_wikidata_labels_with_ror=False, wikidata_labels=None):
        """
        Process organization for indexing with standardized field names.
        
        Args:
            org: Organization dictionary from specific source
            source: Data source identifier
            use_wikidata_labels_with_ror: Whether to enrich ROR indices with WikiData labels
            wikidata_labels: Dictionary of WikiData labels (if None, will be loaded if needed)
            
        Returns:
            Document dictionary with standardized fields
        """
        # Check if we have a mapper for this source
        if source not in self._source_mappers:
            raise ValueError(f"Unknown source: {source}. Registered sources: {list(self._source_mappers.keys())}")
        
        # Map source-specific fields to standard structure
        doc = self._source_mappers[source](org)
        
        # Extract and remove extra_fields to prevent schema conflicts
        extra_fields = {}
        if 'extra_fields' in doc:
            extra_fields = doc.pop('extra_fields')
        
        # Common processing for all sources after mapping
        # Add data source field
        doc['data_source'] = source
        
        # Process name normalization
        original_name = doc['name']
        normalized_name = unidecode(original_name)
        if normalized_name != original_name:
            doc['name_normalized'] = normalized_name
        
        # Generate translations for the institution name
        translated_names = translate_institution_name(original_name)
        
        # Translate aliases if they exist
        translated_aliases = []
        for alias in doc.get('aliases', []) + doc.get('labels', []):
            translated_aliases.extend(translate_institution_name(alias))
        
        # Add WikiData labels if enabled and source is ROR
        if use_wikidata_labels_with_ror and source == 'ror' and (wikidata_labels or wikidata_labels is None):
            # Load WikiData labels if not provided
            if wikidata_labels is None:
                wikidata_labels = self.load_wikidata_labels(verbose=hasattr(self, 'verbose') and self.verbose)
            
            # Get the clean ROR ID
            clean_ror_id = doc['id']
            if clean_ror_id.startswith(ROR_URL):
                clean_ror_id = clean_ror_id.replace(ROR_URL, "")
            
            # Add WikiData labels to aliases
            if clean_ror_id in wikidata_labels:
                for wikidata_label, lang in wikidata_labels[clean_ror_id]:
                    if wikidata_label and wikidata_label not in doc.get('aliases', []) and wikidata_label not in doc.get('labels', []):
                        # Add to aliases or labels list
                        if 'aliases' not in doc:
                            doc['aliases'] = []
                        doc['aliases'].append(wikidata_label)
                        
                        # Also add translations of WikiData labels
                        translated_aliases.extend(translate_institution_name(wikidata_label))
        
        # Use get_variants_list for aliases with translated names
        all_aliases = doc.get('aliases', []) + doc.get('labels', []) + translated_names + translated_aliases
        
        # Generate all variants including normalized ones
        all_alias_variants = get_variants_list(all_aliases)
        doc['aliases_text'] = ' ||| '.join(all_alias_variants)
        doc['aliases_list'] = all_alias_variants
        
        # Process acronyms
        acronyms = doc.get('acronyms', [])
        # Ensure acronyms is a list
        if not isinstance(acronyms, list):
            # Convert string to list if needed
            if isinstance(acronyms, str):
                acronyms = acronyms.split(' ||| ') if ' ||| ' in acronyms else [acronyms]
            else:
                acronyms = []
        
        all_acronyms_variants = get_variants_list(acronyms)
        doc['acronyms'] = ' ||| '.join(all_acronyms_variants)
        
        # Combined field with all names
        all_names = [original_name]
        if normalized_name != original_name:
            all_names.append(normalized_name)
        all_names.extend(acronyms)
        all_names.extend(all_aliases)
        doc['all_text'] = ' ||| '.join(filter(None, all_names))
        
        # Combined location field
        location_parts = [
            doc.get('city', ''),
            doc.get('region', ''),
            doc.get('country_name', '')
        ]
        doc['location_text'] = ' '.join(filter(None, location_parts))
        
        # Clean up internal fields that aren't part of the schema
        if 'aliases' in doc:
            del doc['aliases']
        if 'labels' in doc:
            del doc['labels']
        if 'parent_organizations' in doc:
            del doc['parent_organizations']
        
        # Store any extra_fields as JSON in a STORED field
        if extra_fields:
            try:
                doc['extra_data'] = json.dumps(extra_fields)
            except:
                # If serialization fails, store as string representation
                doc['extra_data'] = str(extra_fields)
        
        return doc
            
    def get_standardized_schema(self):
        """
        Returns a standardized Whoosh schema for organization indexing, 
        usable for any data source.
        
        Returns:
            Schema: A Whoosh schema with standardized fields
        """
        from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED
        from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
        
        return Schema(
            # Core fields - source agnostic
            id=ID(stored=True),
            name=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            name_normalized=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            
            # Names and aliases
            aliases_text=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            aliases_list=STORED,
            acronyms=KEYWORD(stored=True),
            
            # Location information
            city=TEXT(analyzer=StandardAnalyzer(), stored=True),
            region=TEXT(analyzer=StandardAnalyzer(), stored=True),
            country=TEXT(analyzer=StandardAnalyzer(), stored=True),
            country_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            
            # Combined fields for search
            all_text=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            location_text=TEXT(analyzer=StandardAnalyzer(), stored=True),
            
            # Parent organizations
            parent=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            
            # For boosting and filtering
            name_length=STORED,
            
            # Source field - explicitly track the data source
            data_source=ID(stored=True),
            
            # Store extra fields as JSON string
            extra_data=STORED
        )

    def load_wikidata_labels(self, verbose=False):
        """Load WikiData labels for ROR organizations."""
        wikidata_labels = {}
        if os.path.exists(WIKIDATA_LABELS_FILE):
            if verbose:
                logger.info(f"Loading WikiData labels from {WIKIDATA_LABELS_FILE}")
            
            # Determine if it's a gzipped file
            if WIKIDATA_LABELS_FILE.endswith('.gz'):
                import gzip
                open_func = gzip.open
            else:
                open_func = open
            
            try:
                # Load WikiData labels (tab-separated format: ror_id, wiki_id, label, lang)
                with open_func(WIKIDATA_LABELS_FILE, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('ror_id') or not line.strip():
                            continue  # Skip header or empty lines
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            ror_id = parts[0]
                            label = parts[1]
                            lang = parts[2]
                            if ror_id not in wikidata_labels:
                                wikidata_labels[ror_id] = []
                            wikidata_labels[ror_id].append((label, lang))
                
                if verbose:
                    logger.info(f"Loaded WikiData labels for {len(wikidata_labels)} organizations")
            except Exception as e:
                logger.error(f"Error loading WikiData labels: {e}")
        elif verbose:
            logger.info(f"WikiData labels file not found at {WIKIDATA_LABELS_FILE}")
        
        return wikidata_labels

    def create_whoosh_index(self, data, index_dir, source='ror', use_wikidata_labels_with_ror=False):
        """
        Create a Whoosh index from a list of organizations with standardized fields.
        
        Args:
            data: List or DataFrame of organizations
            index_dir: Directory to store the index
            source: Data source identifier ('ror', 'wikidata', etc.)
            use_wikidata_labels_with_ror: Whether to enrich ROR organizations with WikiData labels
            
        Returns:
            bool: True if successful, False otherwise
        """
        from whoosh.index import create_in
        
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            logger.error("No organization data provided for Whoosh index creation")
            return False
            
        # Create the index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Get standardized schema
        schema = self.get_standardized_schema()
        
        # Load WikiData labels if needed and source is ROR
        wikidata_labels = None
        if use_wikidata_labels_with_ror and source == 'ror':
            wikidata_labels = self.load_wikidata_labels(verbose=self.verbose)
        
        # Create the index
        try:
            ix = create_in(index_dir, schema)
            writer = ix.writer()
            
            # Process and add documents
            logger.info(f"Indexing organizations...")
            
            # Handle different data types (DataFrame or list)
            if isinstance(data, pd.DataFrame):
                for _, row in data.iterrows():
                    # Convert row to dict if it's a DataFrame
                    org_dict = row.to_dict()
                    doc = self.process_organization_for_index(
                        org_dict, 
                        source,
                        use_wikidata_labels_with_ror=use_wikidata_labels_with_ror,
                        wikidata_labels=wikidata_labels
                    )
                    
                    # Ensure all values are of appropriate types
                    cleaned_doc = {}
                    for key, value in doc.items():
                        # Skip None values
                        if value is None:
                            continue
                        
                        # Ensure ID and other text fields are strings
                        if key in ['id', 'name', 'city', 'country', 'country_name', 'region', 'parent']:
                            cleaned_doc[key] = str(value)
                        # Ensure numeric fields are integers
                        elif key == 'name_length':
                            cleaned_doc[key] = int(value) if value is not None else 0
                        # Pass other values as is
                        else:
                            cleaned_doc[key] = value
                            
                    writer.add_document(**cleaned_doc)
            else:
                # Assume list of dicts
                for org in data:
                    doc = self.process_organization_for_index(
                        org, 
                        source,
                        use_wikidata_labels_with_ror=use_wikidata_labels_with_ror,
                        wikidata_labels=wikidata_labels
                    )
                    
                    # Ensure all values are of appropriate types
                    cleaned_doc = {}
                    for key, value in doc.items():
                        # Skip None values
                        if value is None:
                            continue
                        
                        # Ensure ID and other text fields are strings
                        if key in ['id', 'name', 'city', 'country', 'country_name', 'region', 'parent']:
                            cleaned_doc[key] = str(value)
                        # Ensure numeric fields are integers
                        elif key == 'name_length':
                            cleaned_doc[key] = int(value) if value is not None else 0
                        # Pass other values as is
                        else:
                            cleaned_doc[key] = value
                            
                    writer.add_document(**cleaned_doc)
                    
            # Commit changes
            logger.info("Committing changes to Whoosh index...")
            writer.commit()
            
            # Save metadata
            metadata = {
                "source": source,
                "indices_type": "whoosh",
                "created_at": datetime.now().isoformat(),
                "organization_count": len(data),
                "use_wikidata_labels_with_ror": use_wikidata_labels_with_ror
            }
            self.save_index_metadata(index_dir, metadata)
            
            logger.info(f"Whoosh index created successfully in {index_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Whoosh index: {e}")
            import traceback
            traceback.print_exc()
            return False

    # HNSW INDEX METHODS 
    def create_hnsw_index(self, data, index_dir, encoder_path=ENCODER_DEFAULT_MODEL, source='ror', use_wikidata_labels_with_ror=False):
        """
        Create an HNSW index from a list of organizations.
        
        Args:
            data: List or DataFrame of organizations
            index_dir: Directory to store the index
            encoder_path: Path to the encoder model
            source: Data source identifier ('ror', 'wikidata', etc.)
            use_wikidata_labels_with_ror: Whether to enrich ROR organizations with WikiData labels
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import hnswlib
        except ImportError:
            logger.error("Failed to import hnswlib. Please install with 'pip install hnswlib'")
            return False
        
        if data is None or (isinstance(data, pd.DataFrame) and data.empty) or (isinstance(data, list) and len(data) == 0):
            logger.error("No organization data provided for HNSW index creation")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Initialize encoder
        try:
            from sentence_transformers import SentenceTransformer, util
            logger.info(f"Loading encoder model: {encoder_path}")
            encoder = SentenceTransformer(encoder_path)
            
            # Add special tokens if needed
            if hasattr(encoder, 'tokenizer') and all(token not in encoder.tokenizer.vocab for token in SPECIAL_TOKENS):
                logger.info("Adding special tokens to encoder tokenizer")
                encoder.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
                encoder._first_module().auto_model.resize_token_embeddings(len(encoder.tokenizer))
        except Exception as e:
            logger.error(f"Error initializing encoder: {e}")
            return False
        
        # Load WikiData labels if needed and source is ROR
        wikidata_labels = None
        if use_wikidata_labels_with_ror and source == 'ror':
            wikidata_labels = self.load_wikidata_labels(verbose=self.verbose)
        
        # Load country-language mappings if available
        try:
            country_languages = {}
            if os.path.exists(COUNTRY_LANGS_FILE):
                logger.info(f"Loading country-language mappings from {COUNTRY_LANGS_FILE}")
                df = pd.read_csv(COUNTRY_LANGS_FILE, sep='\t').fillna('')
                for _, row in df.iterrows():
                    country_name = row.get('country_exonym', '')
                    if country_name:
                        lang_codes = row.get('lang_codes', '').split('|')
                        country_languages[country_name] = [lc.strip() for lc in lang_codes if lc.strip()]
                logger.info(f"Loaded language mappings for {len(country_languages)} countries")
            else:
                logger.warning(f"Country languages file not found at {COUNTRY_LANGS_FILE}")
        except Exception as e:
            logger.error(f"Error loading country-language mappings: {e}")
            country_languages = {}
        
        try:
            # Process organizations and create text representations
            logger.info(f"Processing organizations with text representations")
            
            org_data = []
            org_texts = []  # Will store all text representations for embedding
            org_text_to_data_mapping = {}  # Maps text representation to org_data index
            
            # Process data (handle both DataFrame and list)
            orgs_to_process = data.to_dict('records') if isinstance(data, pd.DataFrame) else data
            
            for org_idx, org in enumerate(orgs_to_process):
                # Process the organization to standardized format with WikiData enrichment if enabled
                std_org = self.process_organization_for_index(
                    org, 
                    source,
                    use_wikidata_labels_with_ror=use_wikidata_labels_with_ror,
                    wikidata_labels=wikidata_labels
                )
                
                # Extract info for text representation
                org_id = std_org.get('id', '')
                name = std_org.get('name', '')
                acronyms = std_org.get('acronyms', '').split(' ||| ') if std_org.get('acronyms') else []
                city = std_org.get('city', '')
                country = std_org.get('country_name', '')
                parent = std_org.get('parent', '')
                
                # Get relevant languages for this organization
                relevant_langs = ['en']  # Always include English
                if country and country in country_languages:
                    relevant_langs.extend(country_languages[country])
                relevant_langs = list(set(relevant_langs))  # Remove duplicates
                
                # Create a basic organization data entry
                org_entry = {
                    "id": org_id,
                    "name": name,
                    "acronyms": acronyms,
                    "city": city,
                    "country": country,
                    "parent": parent,
                    "data_source": source,
                    "text_representations": []  # Will store all text formats
                }
                
                # 1. Create canonical text representation
                canonical_text = f"[MENTION] {name}"
                if acronyms and len(acronyms) > 0:
                    canonical_text += f" [ACRONYM] {acronyms[0]}"
                if parent:
                    canonical_text += f" [PARENT] {parent}"
                if city:
                    canonical_text += f" [CITY] {city}"
                if country:
                    canonical_text += f" [COUNTRY] {country}"
                    
                # Add canonical representation
                org_texts.append(canonical_text)
                org_text_to_data_mapping[canonical_text] = org_idx
                org_entry["text_representations"].append(canonical_text)
                
                # 2. Add variants from aliases_list if available
                if 'aliases_list' in std_org and std_org['aliases_list']:
                    for alias in std_org['aliases_list'][:5]:  # Limit to first 5 aliases
                        if alias and alias != name:
                            alias_text = f"[MENTION] {alias}"
                            if city:
                                alias_text += f" [CITY] {city}"
                            if country:
                                alias_text += f" [COUNTRY] {country}"
                            
                            org_texts.append(alias_text)
                            org_text_to_data_mapping[alias_text] = org_idx
                            org_entry["text_representations"].append(alias_text)
                
                # Add the organization entry to our data
                org_data.append(org_entry)
                
                # Progress indication for large datasets
                if org_idx % 5000 == 0 and org_idx > 0:
                    logger.info(f"Processed {org_idx} organizations...")
            
            # Compute embeddings in batches
            logger.info("Computing embeddings for all text representations...")
            batch_size = 32
            all_embeddings = []
            embedding_to_org_mapping = []  # Maps embedding index to org_data index
            
            for i in range(0, len(org_texts), batch_size):
                batch_texts = org_texts[i:i+batch_size]
                batch_embeddings = encoder.encode(
                    batch_texts, 
                    convert_to_tensor=True,
                    show_progress_bar=(i == 0)  # Only show progress for first batch
                )
                
                # Map these embeddings to organization indices
                for text in batch_texts:
                    embedding_to_org_mapping.append(org_text_to_data_mapping[text])
                
                # Convert to numpy
                batch_np = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_np)
            
            # Concatenate all embeddings
            import numpy as np
            all_embeddings_np = np.vstack(all_embeddings)
            
            # Normalize embeddings
            from sklearn.preprocessing import normalize
            all_embeddings_np = normalize(all_embeddings_np)
            
            # Get vector dimension
            vector_dim = all_embeddings_np.shape[1]
            
            # Create and build HNSW index
            logger.info(f"Building HNSW index with M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}...")
            hnsw_index = hnswlib.Index(space='cosine', dim=vector_dim)
            hnsw_index.init_index(
                max_elements=len(all_embeddings_np), 
                ef_construction=HNSW_EF_CONSTRUCTION, 
                M=HNSW_M
            )
            
            # Add items to index
            hnsw_index.add_items(all_embeddings_np, np.arange(len(all_embeddings_np)))
            
            # Set ef for search (higher ef leads to better accuracy but slower search)
            hnsw_index.set_ef(HNSW_EF_SEARCH)
            
            # Save index and metadata
            index_file = os.path.join(index_dir, "org_index.bin")
            meta_file = os.path.join(index_dir, "org_index_meta.json")
            
            hnsw_index.save_index(index_file)
            
            # Save metadata, including the mapping from embedding to organization
            metadata = {
                "data_source": source,
                "encoder_path": encoder_path,
                "created_at": datetime.now().isoformat(),
                "vector_dim": vector_dim,
                "hnsw_m": HNSW_M,
                "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
                "hnsw_ef_search": HNSW_EF_SEARCH,
                "num_orgs": len(org_data),
                "num_embeddings": len(all_embeddings_np),
                "embedding_to_org_mapping": embedding_to_org_mapping,
                "org_data": org_data,
                "use_wikidata_labels_with_ror": use_wikidata_labels_with_ror
            }
            
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            # Save additional metadata in the standard format
            self.save_index_metadata(index_dir, {
                "source": source,
                "indices_type": "hnsw",
                "created_at": datetime.now().isoformat(),
                "organization_count": len(org_data),
                "encoder_path": encoder_path,
                "use_wikidata_labels_with_ror": use_wikidata_labels_with_ror
            })
            
            logger.info(f"HNSW index created successfully at {index_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating HNSW index: {e}")
            import traceback
            traceback.print_exc()
            return False


    def load_hnsw_index(self, hnsw_index_dir=None, source='ror', org_types=None, countries=None):
        """
        Load HNSW index and metadata.
        
        Args:
            hnsw_index_dir: Optional specific directory for the index
            source: Data source ('ror', 'wikidata')
            org_types: Organization types to include
            countries: Countries to include
            
        Returns:
            Tuple of (index, metadata) or (None, None) if failed
        """
        try:
            import hnswlib
        except ImportError:
            logger.error("Failed to import hnswlib. Please install with 'pip install hnswlib'")
            return None, None
        
        # Get the index path
        if hnsw_index_dir is None:
            hnsw_index_dir = self.get_index_path(source, 'hnsw', org_types, countries)
        
        index_file = os.path.join(hnsw_index_dir, "org_index.bin")
        meta_file = os.path.join(hnsw_index_dir, "org_index_meta.json")
        
        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            logger.error(f"HNSW index files not found at {hnsw_index_dir}")
            return None, None
        
        try:
            # Load metadata
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Get vector dimension from metadata
            vector_dim = metadata.get("vector_dim")
            if not vector_dim:
                logger.error("Vector dimension not found in metadata")
                return None, None
            
            # Load index
            hnsw_index = hnswlib.Index(space='cosine', dim=vector_dim)
            hnsw_index.load_index(index_file, max_elements=metadata.get("num_orgs", 0))
            
            # Set search parameters
            hnsw_index.set_ef(metadata.get("hnsw_ef_search", HNSW_EF_SEARCH))
            
            logger.info(f"Loaded HNSW index with {metadata.get('num_orgs', 'unknown')} organizations")
            return hnsw_index, metadata
            
        except Exception as e:
            logger.error(f"Error loading HNSW index: {e}")
            return None, None
