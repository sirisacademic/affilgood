import os
import pandas as pd
import logging
from . import DataSourceHandler, DataSourceRegistry
from ..constants import *

# Configure logger
logger = logging.getLogger(__name__)

# Default config if not passed. 
DEFAULT_CONFIG = {
    'file_path': f"{EL_DATA_PATH}/spanish_hospitals.xlsx", # EL_DATA_PATH set in constants.py (e.g.: "entity_linking/data")
    'verbose': False,
    'force_rebuild_index': False,
    'encoder_path': ENCODER_DEFAULT_MODEL,
    'source_base_url': 'https://www.sanidad.gob.es/ciudadanos/centros.do?metodo=realizarDetalle&tipo=hospital&numero='
}

@DataSourceRegistry.register
class SpanishHospitalHandler(DataSourceHandler):
    """
    Handler for Spanish hospital data source.
    
    This handler provides functionality to load Spanish hospital data from
    Excel, CSV, or TSV files and use it for entity linking.
    """
    
    @property
    def source_id(self):
        """Return the unique identifier for this data source."""
        return "spanish_hospitals"
    
    def initialize(self, config={}):
        """
        Initialize with configuration.
        
        Args:
            config: Dictionary with configuration (must include 'file_path')
        """
        # Initialize with default values and update the ones passed if any.
        self.config = DEFAULT_CONFIG
        for key in config:
            self.config[key] = config[key]
        file_path = self.config.get('file_path', DEFAULT_CONFIG['file_path'])
        if not file_path:
            logger.warning("Spanish hospital file path not provided")
        elif not os.path.exists(file_path):
            logger.error(f"Spanish hospital file not found: {file_path}")
    
    def load_data(self, config):
        """
        Load Spanish hospital data from a file.
            
        Returns:
            DataFrame with hospital data or None if loading fails
        """
        file_path = config.get('file_path', self.config.get('file_path'))
        verbose = config.get('verbose', self.config.get('verbose'))
        
        if not file_path:
            logger.error("Hospital data file path not provided")
            return None
            
        if not os.path.exists(file_path):
            logger.error(f"Hospital data file not found: {file_path}")
            return None
        
        try:
            # Determine file type based on extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.xlsx' or file_ext == '.xls':
                # Read Excel file
                df = pd.read_excel(file_path)
            elif file_ext == '.csv':
                # Read CSV file
                df = pd.read_csv(file_path)
            elif file_ext == '.tsv':
                # Read TSV file
                df = pd.read_csv(file_path, sep='\t')
            else:
                logger.error(f"Unsupported file extension: {file_ext}")
                return None
            
            # Check that required columns exist
            required_columns = ['CODCNH', 'Nombre Centro', 'Municipio', 'Provincia', 'CCAA']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {', '.join(missing_columns)}")
                return None
            
            # Fill NaN values with empty strings for string columns
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                df[col] = df[col].fillna('')
            
            if verbose:
                logger.info(f"Loaded {len(df)} Spanish hospital records from {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Spanish hospital data: {e}")
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
            logger.error("Hospital file path not provided for index creation")
            return None, None
            
        # Generate an index identifier based on the file name
        file_name = os.path.basename(file_path)
        file_base = os.path.splitext(file_name)[0]
        source = self.source_id
        
        # Create a simple index identifier
        index_id = f"{source}_{file_base}"
        
        # Load hospital data
        hospitals_df = self.load_data(config)
        
        if hospitals_df is None or hospitals_df.empty:
            logger.error(f"No hospital data available for index creation")
            return None, None
        
        # Return the data and index identifier
        return hospitals_df, index_id
    
    def map_organization(self, org):
        """
        Map a Spanish hospital organization to standard fields.
        
        Args:
            org: Spanish hospital organization dictionary
            
        Returns:
            dict: Standardized organization fields
        """
        doc = {}
        
        # Core fields - ensure ID is a string
        doc['id'] = str(org.get('CODCNH', ''))  # Convert to string
        doc['name'] = str(org.get('Nombre Centro', ''))
        doc['name_length'] = len(doc['name'].split()) if doc['name'] else 0
        
        # No explicit aliases or labels in the data, but can be derived later if needed
        doc['aliases'] = []
        doc['labels'] = []
        doc['acronyms'] = []
        
        # Location information - Spanish hierarchy: City → Province → Autonomous Community → Country
        doc['city'] = str(org.get('Municipio', ''))
        doc['region'] = str(org.get('Provincia', ''))
        
        # Hard-code country information
        doc['country'] = 'ES'
        doc['country_name'] = 'Spain'
        
        # Parent organization (hospital complex)
        if org.get('Forma parte Complejo') == 'S' and org.get('Nombre del Complejo'):
            doc['parent'] = str(org.get('Nombre del Complejo', ''))
        else:
            doc['parent'] = ''
        
        # Store additional hospital-specific fields in a nested dictionary to avoid schema conflicts
        # This won't be indexed by Whoosh directly but will be preserved in the processed data
        doc['extra_fields'] = {
            'address': str(org.get('Dirección', '')),
            'postal_code': str(org.get('Código Postal', '')),
            'hospital_type': str(org.get('Clase de Centro', '')),
            'autonomous_community': str(org.get('CCAA', '')),
        }
        
        return doc
    
    def format_id_url(self, org_id):
        """
        Format hospital ID as URL with zero-padding to 6 digits.
        
        Args:
            org_id: Hospital identifier (CODCNH)
            
        Returns:
            str: Formatted URL with properly padded ID
        """
        base_url = self.config.get('source_base_url', '')
        
        # Convert to string if it's not already
        id_str = str(org_id)
        
        # Zero-pad to 6 digits
        padded_id = id_str.zfill(6)
        
        # Return the full URL
        return f"{base_url}{padded_id}"
    

