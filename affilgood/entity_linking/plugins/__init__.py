# Core plugin architecture for data sources

from abc import ABC, abstractmethod

class DataSourceHandler(ABC):
    """
    Abstract base class for data source handlers.
    
    All data source handlers must implement these methods to provide
    consistent functionality for different data sources.
    """
    
    @property
    @abstractmethod
    def source_id(self):
        """
        Return the unique identifier for this data source.
        
        Returns:
            str: Unique identifier (e.g., 'ror', 'wikidata', 'spanish_hospitals')
        """
        pass
    
    @abstractmethod
    def load_data(self, config):
        """
        Load data from the source with the given configuration.
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            Data object (DataFrame or list of dictionaries)
        """
        pass
        
    @abstractmethod
    def get_data_for_indexing(self, config, indices_type='whoosh', **kwargs):
        """
        Prepare and return data for index creation.
        
        Args:
            config: Dictionary of configuration parameters
            indices_type: Type of index to create ('whoosh' or 'hnsw') 
            **kwargs: Additional parameters for compatibility
            
        Returns:
            tuple: (Data object (DataFrame or list), index_id)
        """
        pass
        
    @abstractmethod
    def map_organization(self, org):
        """
        Map organization fields to standard fields.
        
        Args:
            org: Organization dictionary with source-specific fields
            
        Returns:
            dict: Standardized organization fields
        """
        pass
        
    @abstractmethod
    def format_id_url(self, org_id):
        """
        Format organization ID as URL.
        
        Args:
            org_id: Organization identifier
            
        Returns:
            str: Formatted URL
        """
        pass
       
    def initialize(self, config):
        """
        Initialize the handler with configuration.
        
        Args:
            config: Dictionary of configuration parameters
            
        Returns:
            None
        """
        # Default implementation - may be overridden by subclasses
        pass


class DataSourceRegistry:
    """
    Registry for data source handlers.
    
    This class provides methods to register and retrieve data source handlers.
    """
    _handlers = {}
    
    @classmethod
    def register(cls, handler_class):
        """
        Register a new data source handler.
        
        This method can be used as a decorator on handler classes.
        
        Args:
            handler_class: DataSourceHandler subclass
            
        Returns:
            handler_class: The same class, for use as decorator
        """
        handler = handler_class()
        cls._handlers[handler.source_id] = handler
        return handler_class
        
    @classmethod
    def get_handler(cls, source_id):
        """
        Get handler for a specific data source.
        
        Args:
            source_id: Unique identifier for the data source
            
        Returns:
            DataSourceHandler: Handler for the data source, or None if not found
        """
        return cls._handlers.get(source_id)
        
    @classmethod
    def get_all_handlers(cls):
        """
        Get all registered handlers.
        
        Returns:
            dict: Dictionary mapping source_id to handler
        """
        return cls._handlers
