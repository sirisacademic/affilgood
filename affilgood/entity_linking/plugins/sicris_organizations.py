import os
import pandas as pd
import logging
from . import DataSourceHandler, DataSourceRegistry
from ..constants import *

# Configure logger
logger = logging.getLogger(__name__)

# Default config
DEFAULT_CONFIG = {
    'file_path': f"{EL_DATA_PATH}/sicris_registry.tsv",
    'verbose': False,
    'force_rebuild_index': False,
    'encoder_path': ENCODER_DEFAULT_MODEL,
    'source_base_url': 'SICRIS:'
}

@DataSourceRegistry.register
class SicrisOrganizationHandler(DataSourceHandler):
    """
    Handler for Slovenian SICRIS organization data source.
    
    This handler provides functionality to load SICRIS organization data from
    TSV files and use it for entity linking.
    """
    
    @property
    def source_id(self):
        """Return the unique identifier for this data source."""
        return "sicris"
    
    def initialize(self, config={}):
        """
        Initialize with configuration.
        
        Args:
            config: Dictionary with configuration (must include 'file_path')
        """
        # Initialize with default values and update with passed config
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        
        file_path = self.config.get('file_path')
        if not file_path:
            logger.warning("SICRIS file path not provided")
        elif not os.path.exists(file_path):
            logger.error(f"SICRIS file not found: {file_path}")
    
    def load_data(self, config):
        """
        Load SICRIS organization data from a TSV file.
            
        Returns:
            DataFrame with organization data or None if loading fails
        """
        file_path = config.get('file_path', self.config.get('file_path'))
        verbose = config.get('verbose', self.config.get('verbose'))
        
        if not file_path:
            logger.error("SICRIS data file path not provided")
            return None
            
        if not os.path.exists(file_path):
            logger.error(f"SICRIS data file not found: {file_path}")
            return None
        
        try:
            # Read TSV file
            df = pd.read_csv(file_path, sep='\t')
            
            # Check that required columns exist
            required_columns = ['SICRIS ID', 'Organization Name', 'City']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {', '.join(missing_columns)}")
                return None
            
            # Fill NaN values with empty strings for string columns
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                df[col] = df[col].fillna('')
            
            # Fill NaN values for numeric columns
            numeric_columns = df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(0)
            
            if verbose:
                logger.info(f"Loaded {len(df)} SICRIS organization records from {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading SICRIS organization data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_data_for_indexing(self, config, indices_type='whoosh', **kwargs):
        """
        Prepare and return data for index creation.
        
        Args:
            config: Dictionary with configuration
            indices_type: Type of index to create ('whoosh' or 'hnsw')
            **kwargs: Additional parameters (kept for compatibility)
            
        Returns:
            tuple: (DataFrame with data, index_id)
        """
        file_path = config.get('file_path', self.config.get('file_path'))
        
        if not file_path:
            logger.error("SICRIS file path not provided for index creation")
            return None, None
            
        # Generate an index identifier based on the file name
        file_name = os.path.basename(file_path)
        file_base = os.path.splitext(file_name)[0]
        source = self.source_id
        
        # Create a simple index identifier
        index_id = f"{source}_{file_base}"
        
        # Load SICRIS data
        orgs_df = self.load_data(config)
        
        if orgs_df is None or orgs_df.empty:
            logger.error(f"No SICRIS data available for index creation")
            return None, None
        
        # Return the data and index identifier
        return orgs_df, index_id
    
    def map_organization(self, org):
        """
        Map a SICRIS organization to standard fields.
        
        Args:
            org: SICRIS organization dictionary
            
        Returns:
            dict: Standardized organization fields
        """
        doc = {}
        
        # Core fields - ensure ID is a string
        doc['id'] = str(org.get('SICRIS ID', ''))
        doc['name'] = str(org.get('Organization Name', ''))
        doc['name_length'] = len(doc['name'].split()) if doc['name'] else 0
        
        # No explicit aliases or labels in the data, but can be derived later if needed
        doc['aliases'] = []
        doc['labels'] = []
        doc['acronyms'] = []
        
        # Location information
        doc['city'] = str(org.get('City', ''))
        doc['region'] = str(org.get('nuts3_entity', ''))  # Use NUTS3 entity as region
        
        # Hard-code country information for Slovenia
        doc['country'] = 'SI'
        doc['country_name'] = 'Slovenia'
        
        # No parent organization info in the data
        doc['parent'] = ''
        
        # Store additional SICRIS-specific fields in a nested dictionary
        doc['extra_fields'] = {
            'status_form': str(org.get('Status Form (name)', '')),
            'tax_number': str(org.get('Tax Number', '')),
            'nuts3_code': str(org.get('NUTS 3 CODE', '')),
            'eu_lau_code': str(org.get('EU LAU CODE', '')),
        }
        
        return doc
    
    def format_id_url(self, org_id):
        """
        Format SICRIS ID as URL.
        
        Args:
            org_id: SICRIS organization identifier
            
        Returns:
            str: Formatted URL
        """
        base_url = self.config.get('source_base_url', '')
        return f"{base_url}{org_id}"
        
    def format_id_url(self, org_id):
        """
        Format SICRIS ID as URL, zero-padding to 4 digits.
        
        Args:
            org_id: SICRIS organization identifier
            
        Returns:
            str: Formatted URL with zero-padded ID
        """
        # Convert to string and zero-pad to 4 digits
        padded_id = str(org_id).zfill(4)
        base_url = self.config.get('source_base_url', '')
        return f"{base_url}{padded_id}"
