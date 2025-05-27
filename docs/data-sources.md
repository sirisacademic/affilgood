# Data Sources in AffilGood

AffilGood supports multiple data sources for entity linking, allowing you to match organizations against various registries. This document explains the available data sources, how to configure them, and how to implement custom data sources.

## Table of Contents

1. [Data Source Architecture](#data-source-architecture)
2. [Available Data Sources](#available-data-sources)
3. [Data Source Configuration](#data-source-configuration)
4. [Custom Data Sources](#custom-data-sources)
5. [Using Multiple Data Sources](#using-multiple-data-sources)
6. [Best Practices](#best-practices)

## Data Source Architecture

AffilGood uses a plugin-based architecture for data sources:

1. `DataSourceHandler`: Abstract base class that defines the interface for all data sources
2. `DataSourceRegistry`: Registry that manages available data source handlers

This architecture allows easy extension with new data sources without modifying the core pipeline.

### DataSourceHandler Interface

The `DataSourceHandler` interface defines the following methods:

```python
class DataSourceHandler(ABC):
    @property
    @abstractmethod
    def source_id(self):
        """Return the unique identifier for this data source."""
        pass
    
    @abstractmethod
    def load_data(self, config):
        """Load data from the source with the given configuration."""
        pass
        
    @abstractmethod
    def get_data_for_indexing(self, config, indices_type='whoosh', **kwargs):
        """Prepare and return data for index creation."""
        pass
        
    @abstractmethod
    def map_organization(self, org):
        """Map organization fields to standard fields."""
        pass
        
    @abstractmethod
    def format_id_url(self, org_id):
        """Format organization ID as URL."""
        pass
       
    def initialize(self, config):
        """Initialize the handler with configuration."""
        pass
```

### DataSourceRegistry

The `DataSourceRegistry` manages registered data source handlers:

```python
class DataSourceRegistry:
    @classmethod
    def register(cls, handler_class):
        """Register a new data source handler."""
        pass
        
    @classmethod
    def get_handler(cls, source_id):
        """Get handler for a specific data source."""
        pass
        
    @classmethod
    def get_all_handlers(cls):
        """Get all registered handlers."""
        pass
```

## Available Data Sources

### Research Organization Registry (ROR)

ROR is the primary data source for academic and research institutions. It provides a comprehensive registry of research organizations worldwide.

```python
from affilgood.entity_linking import DataSourceRegistry

# Get ROR handler
ror_handler = DataSourceRegistry.get_handler("ror")

# Use with a linker
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(data_source="ror")

# Or with AffilGood
from affilgood import AffilGood
affil_good = AffilGood(entity_linkers='Dense')  # Uses ROR by default
```

### WikiData

WikiData provides access to a broader range of organizations beyond academic institutions, including companies, hospitals, and other entities.

```python
from affilgood.entity_linking.wikidata_dump_generator import WikidataDumpGenerator

# Initialize WikiData generator
wikidata_generator = WikidataDumpGenerator(verbose=True)

# Generate index for specific countries and organization types
df = wikidata_generator.get_index(
    countries=["Spain", "Germany"],
    org_types=["university", "hospital"]
)

# Use WikiData as a data source
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(data_source="wikidata")
```

#### WikiData Organization Types

WikiData supports various organization types, including:

- `university`
- `hospital`
- `research_institute`
- `company`
- `museum`
- `library`
- `school`
- `government_agency`

You can specify the types when generating the index:

```python
df = wikidata_generator.get_index(
    countries=["France", "Italy"],
    org_types=["university", "research_institute"]
)
```

#### WikiData Country Selection

You can filter organizations by country:

```python
df = wikidata_generator.get_index(
    countries=["Spain", "Germany", "United States"],
    org_types="university"
)
```

### Spanish Hospitals

A specialized data source for Spanish hospitals.

```python
from affilgood.entity_linking import DataSourceRegistry

# Get Spanish hospitals handler
spanish_hospitals_handler = DataSourceRegistry.get_handler("spanish_hospitals")

# Configure with custom settings
spanish_hospitals_handler.initialize({
    'file_path': "path/to/spanish_hospitals.xlsx",
    'verbose': True
})

# Use with a linker
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(data_source="spanish_hospitals")
```

### SICRIS (Slovenian Research Organizations)

A specialized data source for Slovenian research organizations.

```python
from affilgood.entity_linking import DataSourceRegistry

# Get SICRIS handler
sicris_handler = DataSourceRegistry.get_handler("sicris")

# Configure with custom settings
sicris_handler.initialize({
    'file_path': "path/to/sicris_registry.tsv",
    'verbose': True
})

# Use with a linker
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(data_source="sicris")
```

## Data Source Configuration

### ROR Configuration

ROR requires minimal configuration as it uses pre-built indices:

```python
from affilgood.entity_linking import DataSourceRegistry
from affilgood.entity_linking.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Get or download the latest ROR dump
ror_dump_path = data_manager.get_latest_ror()

# Configure a linker with ROR
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(
    data_manager=data_manager,
    data_source="ror"
)
```

### WikiData Configuration

WikiData requires generating an index for the desired organization types and countries:

```python
from affilgood.entity_linking.wikidata_dump_generator import WikidataDumpGenerator
from affilgood.entity_linking.data_manager import DataManager

# Initialize data manager
data_manager = DataManager()

# Initialize WikiData generator
wikidata_generator = WikidataDumpGenerator(verbose=True)

# Generate index
df = wikidata_generator.get_index(
    countries=["Spain", "Germany"],
    org_types=["university", "hospital"]
)

# Configure a linker with WikiData
from affilgood.entity_linking.dense_linker import DenseLinker
dense_linker = DenseLinker(
    data_manager=data_manager,
    data_source="wikidata"
)
```

#### WikiData Cache Configuration

WikiData uses caching to improve performance:

```python
from affilgood.entity_linking.wikidata_dump_generator import WikiDataCache

# Configure cache
cache = WikiDataCache(
    cache_dir="path/to/cache",
    cache_expiration_days=30
)

# Clear expired cache entries
cleared_count = cache.clear_expired()
print(f"Cleared {cleared_count} expired cache entries")

# Get cache statistics
stats = cache.get_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
```

### Spanish Hospitals Configuration

Spanish hospitals data source requires a file path to the hospital registry:

```python
from affilgood.entity_linking import DataSourceRegistry

# Get Spanish hospitals handler
spanish_hospitals_handler = DataSourceRegistry.get_handler("spanish_hospitals")

# Configure with custom settings
spanish_hospitals_handler.initialize({
    'file_path': "path/to/spanish_hospitals.xlsx",  # Excel file path
    'verbose': True,                               # Enable logging
    'force_rebuild_index': False,                  # Rebuild index if exists
    'source_base_url': "https://www.sanidad.gob.es/ciudadanos/centros.do?metodo=realizarDetalle&tipo=hospital&numero="
})
```

### SICRIS Configuration

SICRIS data source requires a file path to the registry:

```python
from affilgood.entity_linking import DataSourceRegistry

# Get SICRIS handler
sicris_handler = DataSourceRegistry.get_handler("sicris")

# Configure with custom settings
sicris_handler.initialize({
    'file_path': "path/to/sicris_registry.tsv",  # TSV file path
    'verbose': True,                            # Enable logging
    'force_rebuild_index': False,               # Rebuild index if exists
    'source_base_url': "SICRIS:"                # Base URL for IDs
})
```

## Custom Data Sources

You can implement custom data sources by creating a class that extends `DataSourceHandler` and registering it with `DataSourceRegistry`.

### Example: Custom CSV Data Source

```python
from affilgood.entity_linking import DataSourceHandler, DataSourceRegistry
import pandas as pd
import os

@DataSourceRegistry.register
class CustomCSVDataSourceHandler(DataSourceHandler):
    @property
    def source_id(self):
        """Return the unique identifier for this data source."""
        return "custom_csv"
    
    def initialize(self, config):
        """Initialize with configuration."""
        self.config = config or {}
        self.file_path = self.config.get('file_path')
        self.verbose = self.config.get('verbose', False)
        
        if not self.file_path:
            raise ValueError("File path not provided for custom CSV data source")
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
    
    def load_data(self, config):
        """Load data from the CSV file."""
        file_path = config.get('file_path', self.file_path)
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check that required columns exist
            required_columns = ['id', 'name', 'city', 'country']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Fill NaN values
            df = df.fillna('')
            
            if self.verbose:
                print(f"Loaded {len(df)} records from {file_path}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_data_for_indexing(self, config, indices_type='whoosh', **kwargs):
        """Prepare and return data for index creation."""
        df = self.load_data(config)
        
        if df is None or df.empty:
            return None, None
        
        # Generate an index identifier
        file_name = os.path.basename(self.file_path)
        file_base = os.path.splitext(file_name)[0]
        index_id = f"{self.source_id}_{file_base}"
        
        return df, index_id
    
    def map_organization(self, org):
        """Map organization fields to standard fields."""
        return {
            'id': str(org.get('id', '')),
            'name': str(org.get('name', '')),
            'name_length': len(str(org.get('name', '')).split()),
            'aliases': org.get('aliases', '').split('|') if org.get('aliases') else [],
            'acronyms': org.get('acronyms', '').split('|') if org.get('acronyms') else [],
            'city': str(org.get('city', '')),
            'region': str(org.get('region', '')),
            'country': str(org.get('country', '')),
            'country_name': str(org.get('country_name', org.get('country', ''))),
            'parent': str(org.get('parent', ''))
        }
    
    def format_id_url(self, org_id):
        """Format organization ID as URL."""
        base_url = self.config.get('source_base_url', '')
        return f"{base_url}{org_id}"
```

### Example: Custom API Data Source

```python
from affilgood.entity_linking import DataSourceHandler, DataSourceRegistry
import requests
import pandas as pd
import os
import json

@DataSourceRegistry.register
class CustomAPIDataSourceHandler(DataSourceHandler):
    @property
    def source_id(self):
        """Return the unique identifier for this data source."""
        return "custom_api"
    
    def initialize(self, config):
        """Initialize with configuration."""
        self.config = config or {}
        self.api_url = self.config.get('api_url')
        self.api_key = self.config.get('api_key')
        self.cache_file = self.config.get('cache_file')
        self.verbose = self.config.get('verbose', False)
        
        if not self.api_url:
            raise ValueError("API URL not provided for custom API data source")
    
    def load_data(self, config):
        """Load data from the API or cache."""
        api_url = config.get('api_url', self.api_url)
        api_key = config.get('api_key', self.api_key)
        cache_file = config.get('cache_file', self.cache_file)
        force_refresh = config.get('force_refresh', False)
        
        # Use cache if available and not forcing refresh
        if cache_file and os.path.exists(cache_file) and not force_refresh:
            try:
                if self.verbose:
                    print(f"Loading data from cache: {cache_file}")
                return pd.read_json(cache_file)
            except Exception as e:
                print(f"Error loading cache: {e}")
                # Fall back to API if cache fails
        
        try:
            # Make API request
            headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Save to cache if cache_file provided
            if cache_file:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_json(cache_file)
                if self.verbose:
                    print(f"Saved data to cache: {cache_file}")
            
            if self.verbose:
                print(f"Loaded {len(df)} records from API")
            
            return df
            
        except Exception as e:
            print(f"Error loading data from API: {e}")
            return None
    
    def get_data_for_indexing(self, config, indices_type='whoosh', **kwargs):
        """Prepare and return data for index creation."""
        df = self.load_data(config)
        
        if df is None or df.empty:
            return None, None
        
        # Generate an index identifier
        api_name = self.api_url.split('/')[-1].split('?')[0]
        index_id = f"{self.source_id}_{api_name}"
        
        return df, index_id
    
    def map_organization(self, org):
        """Map organization fields to standard fields."""
        return {
            'id': str(org.get('id', '')),
            'name': str(org.get('name', '')),
            'name_length': len(str(org.get('name', '')).split()),
            'aliases': org.get('alternative_names', []),
            'acronyms': org.get('acronyms', []),
            'city': str(org.get('city', '')),
            'region': str(org.get('region', '')),
            'country': str(org.get('country_code', '')),
            'country_name': str(org.get('country', '')),
            'parent': str(org.get('parent_organization', ''))
        }
    
    def format_id_url(self, org_id):
        """Format organization ID as URL."""
        base_url = self.config.get('source_base_url', '')
        return f"{base_url}{org_id}"
```

## Using Multiple Data Sources

AffilGood supports using multiple data sources simultaneously to improve coverage and accuracy.

### Within a Single Linker

You can configure a linker to use a specific data source:

```python
from affilgood.entity_linking.dense_linker import DenseLinker

# Create linkers for different data sources
ror_linker = DenseLinker(data_source="ror")
wikidata_linker = DenseLinker(data_source="wikidata")
custom_linker = DenseLinker(data_source="custom_csv")
```

### With Multiple Linkers

You can use multiple linkers with different data sources:

```python
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Create linkers for different data sources
ror_linker = DenseLinker(data_source="ror")
wikidata_linker = DenseLinker(data_source="wikidata")
hospitals_linker = WhooshLinker(data_source="spanish_hospitals")

# Create entity linker with all linkers
entity_linker = EntityLinker(
    linkers=[ror_linker, wikidata_linker, hospitals_linker],
    return_scores=True
)

# Process entities
results = entity_linker.process_in_chunks(entities)
```

### With AffilGood

You can configure AffilGood to use multiple linkers with different data sources:

```python
from affilgood import AffilGood
from affilgood.entity_linking.dense_linker import DenseLinker
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Create linkers for different data sources
ror_linker = DenseLinker(data_source="ror")
wikidata_linker = DenseLinker(data_source="wikidata")
hospitals_linker = WhooshLinker(data_source="spanish_hospitals")

# Create AffilGood with all linkers
affil_good = AffilGood(
    entity_linkers=[ror_linker, wikidata_linker, hospitals_linker],
    return_scores=True
)

# Process affiliations
results = affil_good.process(affiliations)
```

## Best Practices

### Data Source Selection

- **ROR**: Best for academic and research organizations
- **WikiData**: Better for a broader range of organizations, including companies, hospitals, etc.
- **Specialized Sources**: Best for domain-specific needs (e.g., Spanish hospitals)

### Performance Considerations

- Use caching whenever possible to improve performance
- For WikiData, limit the countries and organization types to reduce index size
- Consider pre-generating indices for commonly used data sources

### Custom Data Sources

- Implement all required methods of the `DataSourceHandler` interface
- Use standardized field mappings for consistent results
- Cache API responses when possible to reduce external dependencies
- Register your handler with `DataSourceRegistry` for seamless integration

### Troubleshooting

- If a data source is not working, check that the required data file exists
- For WikiData, ensure the SPARQL endpoint is accessible
- For custom data sources, validate the data format and required fields
