# AffilGood: Usage Examples

This document provides detailed examples of how to use the AffilGood library for different scenarios.

## Basic Usage

### Complete Pipeline

The simplest way to use AffilGood is to run the complete pipeline with default settings:

```python
from affilgood import AffilGood

# Initialize with default settings
affil_good = AffilGood()

# Example affiliation strings
affiliations = [
    "Department of Computer Science, University of California, Berkeley, CA 94720, USA",
    "Max Planck Institute for Intelligent Systems, 72076 Tübingen, Germany"
]

# Process affiliations through the complete pipeline
results = affil_good.process(affiliations)

# Print identified organizations and their ROR IDs
for item in results:
    print(f"Original: {item['raw_text']}")
    print(f"Spans: {item['span_entities']}")
    print(f"Organizations: {[ner.get('ORG', []) for ner in item['ner']]}")
    print(f"ROR: {item['ror']}")
    print("---")
```

### Step-by-Step Processing

You can also run each step of the pipeline separately:

```python
from affilgood import AffilGood

affil_good = AffilGood()

# Example affiliation strings
affiliations = [
    "Department of Computer Science, University of California, Berkeley, CA 94720, USA",
    "Max Planck Institute for Intelligent Systems, 72076 Tübingen, Germany"
]

# Step 1: Identify spans
spans = affil_good.get_span(affiliations)
print("After span identification:", spans[0]['span_entities'])

# Step 2: Perform named entity recognition
entities = affil_good.get_ner(spans)
print("Entities recognized:", entities[0]['ner'][0])

# Step 3: Normalize location metadata
normalized = affil_good.get_normalization(entities)
print("Normalized locations:", normalized[0]['osm'][0])

# Step 4: Link to ROR identifiers
linked = affil_good.get_entity_linking(normalized)
print("ROR IDs:", linked[0]['ror'][0])
```

## Advanced Usage

### Custom Configuration

Customize AffilGood components for your specific needs:

```python
from affilgood import AffilGood

# Configure with specific models and settings
affil_good = AffilGood(
    span_separator=';',  # Split spans by semicolons instead of using model
    ner_model_path='nicolauduran45/affilgood-ner-multilingual-v2',
    entity_linkers='S2AFF',  # Use S2AFF instead of Whoosh for entity linking
    return_scores=True,  # Include confidence scores in results
    metadata_normalization=True,
    verbose=True,  # Enable detailed logging
    device='cuda:0'  # Use specific GPU
)

# Process affiliations with custom configuration
affiliations = [
    "Department of Physics; Faculty of Engineering, University of XYZ, London, UK",
    "Institute of Quantum Computing, ETH Zürich, Switzerland"
]

results = affil_good.process(affiliations)
```

### Multiple Entity Linkers

Use multiple entity linkers and combine their results:

```python
from affilgood import AffilGood
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.entity_linking.s2aff_linker import S2AFFLinker
from affilgood.entity_linking.entity_linker import EntityLinker

# Initialize linkers with custom settings
whoosh_linker = WhooshLinker(
    threshold_score=0.3,
    rerank=True,
    number_candidates_rerank=7,
    debug=False
)

s2aff_linker = S2AFFLinker(
    device="cpu",
    debug=False
)

# Create a combined entity linker
entity_linker = EntityLinker(
    linkers=[whoosh_linker, s2aff_linker],
    return_scores=True
)

# Initialize AffilGood with the custom entity linker
affil_good = AffilGood(
    entity_linkers=entity_linker
)

# Process affiliations
affiliations = ["National Institute of Standards and Technology, Boulder, CO, USA"]
results = affil_good.process(affiliations)
```

### Processing Large Datasets

Efficiently process large datasets with parallel processing:

```python
from affilgood import AffilGood
import pandas as pd
from tqdm import tqdm
import concurrent.futures

# Load a large dataset
df = pd.read_csv("large_affiliation_dataset.csv")
affiliations = df['affiliation'].tolist()

# Initialize AffilGood
affil_good = AffilGood()

# Process in chunks for better memory management
def process_chunk(chunk):
    return affil_good.process(chunk)

# Split into chunks
chunk_size = 100
chunks = [affiliations[i:i+chunk_size] for i in range(0, len(affiliations), chunk_size)]

# Process chunks in parallel
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(chunks)):
        results.extend(future.result())

# Convert results back to DataFrame
results_df = pd.DataFrame({
    'raw_text': [item['raw_text'] for item in results],
    'spans': [item['span_entities'] for item in results],
    'organizations': [[ner.get('ORG', []) for ner in item['ner']] for item in results],
    'ror_ids': [item['ror'] for item in results]
})

# Save the results
results_df.to_csv("processed_affiliations.csv", index=False)
```

## Specialized Use Cases

### Only Entity Linking

If you already have structured affiliation data with organizations and locations identified:

```python
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.entity_linking.whoosh_linker import WhooshLinker

# Initialize just the entity linker
linker = EntityLinker(linkers=[WhooshLinker()])

# Prepare structured input data
structured_data = [
    {
        "raw_text": "University of Oxford, UK",
        "span_entities": ["University of Oxford, UK"],
        "ner": [
            {
                "ORG": ["University of Oxford"],
                "COUNTRY": ["UK"]
            }
        ],
        "ner_raw": [...],  # Raw NER output
        "osm": [None]  # No OSM data yet
    }
]

# Link to ROR IDs
linked_results = linker.process_in_chunks(structured_data)
print(linked_results[0]['ror'])
```

### Only Named Entity Recognition

Extract organizations and locations from text without linking:

```python
from affilgood.ner.ner import NER
from affilgood.span_identification.simple_span_identifier import SimpleSpanIdentifier

# Initialize components
span_identifier = SimpleSpanIdentifier(separator=None)  # Treat each text as a single span
ner = NER()

# Example affiliations
affiliations = [
    "Department of Physics, Harvard University, Cambridge, MA, USA",
    "CERN, Geneva, Switzerland"
]

# Identify spans and run NER
spans = span_identifier.identify_spans(affiliations)
entities = ner.recognize_entities(spans)

# Print recognized entities
for item in entities:
    print(f"Text: {item['raw_text']}")
    for i, span in enumerate(item['span_entities']):
        print(f"  Span {i+1}: {span}")
        print(f"    Organizations: {item['ner'][i].get('ORG', [])}")
        print(f"    Departments: {item['ner'][i].get('SUBORG', [])}")
        print(f"    City: {item['ner'][i].get('CITY', [])}")
        print(f"    Region: {item['ner'][i].get('REGION', [])}")
        print(f"    Country: {item['ner'][i].get('COUNTRY', [])}")
    print("---")
```

### Geocoding Locations

Use just the normalization module to standardize and geocode locations:

```python
from affilgood.metadata_normalization.normalizer import GeoNormalizer

# Initialize normalizer
normalizer = GeoNormalizer()

# Example data
entities = [
    {
        "raw_text": "Example University, New York City, USA",
        "span_entities": ["Example University, New York City, USA"],
        "ner": [
            {
                "ORG": ["Example University"],
                "CITY": ["New York City"],
                "COUNTRY": ["USA"]
            }
        ],
        "ner_raw": []
    }
]

# Normalize location data
normalized = normalizer.normalize(entities)

# Print geocoded information
for item in normalized:
    print(f"Original: {item['raw_text']}")
    for i, osm_data in enumerate(item['osm']):
        if osm_data:
            print(f"  Normalized city: {osm_data.get('CITY')}")
            print(f"  Normalized country: {osm_data.get('COUNTRY')}")
            print(f"  Coordinates: {osm_data.get('COORDS')}")
            print(f"  OSM ID: {osm_data.get('OSM_ID')}")
```

### Custom Whoosh Index

Create and use a custom Whoosh index for entity linking:

```python
from affilgood import AffilGood
from affilgood.entity_linking.whoosh_linker import WhooshLinker
from affilgood.utils.data_manager import DataManager

# Initialize a data manager
data_manager = DataManager()

# Create a custom Whoosh index
custom_index_dir = "/path/to/custom_index"
data_manager.create_whoosh_index(custom_index_dir)

# Create a Whoosh linker with the custom index
whoosh_linker = WhooshLinker(
    data_manager=data_manager,
    index_dir=custom_index_dir,
    rebuild_index=False  # Use existing index
)

# Initialize AffilGood with the custom linker
affil_good = AffilGood(entity_linkers=whoosh_linker)

# Process affiliations
affiliations = ["Stanford University, California, USA"]
results = affil_good.process(affiliations)
```

### Multilingual Support

Process affiliations in multiple languages:

```python
from affilgood import AffilGood

# Initialize with multilingual models
affil_good = AffilGood(
    span_model_path="nicolauduran45/affilgood-span-multilingual-v2",
    ner_model_path="nicolauduran45/affilgood-ner-multilingual-v2"
)

# Example affiliations in different languages
affiliations = [
    "Universidad de Barcelona, Departamento de Física, Barcelona, España",
    "Max-Planck-Institut für Intelligente Systeme, Tübingen, Deutschland",
    "Университет Ломоносова, Москва, Россия",
    "東京大学, 物理学科, 東京, 日本"
]

# Process multilingual affiliations
results = affil_good.process(affiliations)

# Print results
for item in results:
    print(f"Original: {item['raw_text']}")
    print(f"Organizations: {[ner.get('ORG', []) for ner in item['ner']]}")
    print(f"ROR: {item['ror']}")
    print("---")
```

## Integration Examples

### Integration with Pandas

Process affiliations in a pandas DataFrame:

```python
import pandas as pd
from affilgood import AffilGood

# Initialize AffilGood
affil_good = AffilGood()

# Create a sample DataFrame
df = pd.DataFrame({
    'author': ['John Smith', 'Jane Doe', 'Alice Johnson'],
    'affiliation': [
        'Harvard Medical School, Boston, MA, United States',
        'Institut Pasteur, Paris, France',
        'University of Oxford, Department of Chemistry, Oxford, UK'
    ]
})

# Define a function to extract ROR IDs
def extract_ror_ids(affiliation):
    results = affil_good.process([affiliation])
    if results and results[0]['ror']:
        return results[0]['ror'][0]
    return None

# Apply the function to the DataFrame
df['ror_id'] = df['affiliation'].apply(extract_ror_ids)

print(df)
```

### Integration with SPARQL and RDF

Convert AffilGood results to RDF and query with SPARQL:

```python
from affilgood import AffilGood
import rdflib
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

# Initialize AffilGood
affil_good = AffilGood()

# Process example affiliations
affiliations = [
    "Stanford University, Department of Physics, Stanford, CA, USA",
    "University of Cambridge, Cambridge, UK"
]
results = affil_good.process(affiliations)

# Create RDF graph
g = Graph()

# Define namespaces
SCHEMA = Namespace("http://schema.org/")
ROR = Namespace("https://ror.org/")
g.bind("schema", SCHEMA)
g.bind("ror", ROR)

# Create RDF triples from results
for i, item in enumerate(results):
    # Create a node for the affiliation string
    affiliation_node = URIRef(f"http://example.org/affiliation/{i}")
    g.add((affiliation_node, RDF.type, SCHEMA.Organization))
    g.add((affiliation_node, SCHEMA.name, Literal(item['raw_text'])))
    
    # Add organization and location information
    for j, ner in enumerate(item['ner']):
        if 'ORG' in ner and ner['ORG']:
            g.add((affiliation_node, SCHEMA.legalName, Literal(ner['ORG'][0])))
        
        if 'CITY' in ner and ner['CITY']:
            g.add((affiliation_node, SCHEMA.location, Literal(ner['CITY'][0])))
        
        if 'COUNTRY' in ner and ner['COUNTRY']:
            g.add((affiliation_node, SCHEMA.addressCountry, Literal(ner['COUNTRY'][0])))
    
    # Add ROR identifiers
    if item['ror'] and item['ror'][0]:
        # Extract ROR ID from the formatted string
        ror_str = item['ror'][0]
        import re
        ror_id_match = re.search(r'https://ror.org/([0-9a-z]+)', ror_str)
        if ror_id_match:
            ror_id = ror_id_match.group(1)
            ror_uri = ROR[ror_id]
            g.add((affiliation_node, SCHEMA.sameAs, ror_uri))

# Example SPARQL query to find all organizations with their ROR IDs
query = """
SELECT ?org ?name ?ror
WHERE {
    ?org a schema:Organization ;
         schema:name ?name ;
         schema:sameAs ?ror .
    FILTER(STRSTARTS(STR(?ror), "https://ror.org/"))
}
"""

# Execute the query
results = g.query(query)

# Print results
for row in results:
    print(f"Organization: {row.name}")
    print(f"ROR URI: {row.ror}")
    print("---")

# Serialize graph to Turtle format
turtle_data = g.serialize(format="turtle")
print(turtle_data)
```

### Integration with a Web API (FastAPI)

Create a simple API for affiliation processing:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from affilgood import AffilGood
import uvicorn

# Initialize AffilGood
affil_good = AffilGood()

# Create FastAPI app
app = FastAPI(title="AffilGood API")

# Define request and response models
class AffiliationRequest(BaseModel):
    affiliations: List[str]
    
class EntityModel(BaseModel):
    org: Optional[List[str]] = None
    suborg: Optional[List[str]] = None
    city: Optional[List[str]] = None
    region: Optional[List[str]] = None
    country: Optional[List[str]] = None
    
class GeoModel(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    coords: Optional[List[float]] = None
    
class AffiliationResult(BaseModel):
    raw_text: str
    spans: List[str]
    entities: List[EntityModel]
    geo: List[Optional[GeoModel]]
    ror_ids: List[str]

class AffiliationResponse(BaseModel):
    results: List[AffiliationResult]

# Define API endpoint
@app.post("/process", response_model=AffiliationResponse)
async def process_affiliations(request: AffiliationRequest):
    try:
        # Process affiliations
        results = affil_good.process(request.affiliations)
        
        # Transform results to response format
        response_results = []
        for item in results:
            entities = []
            geo = []
            
            for i, ner in enumerate(item.get('ner', [])):
                entity = EntityModel(
                    org=ner.get('ORG'),
                    suborg=ner.get('SUBORG'),
                    city=ner.get('CITY'),
                    region=ner.get('REGION'),
                    country=ner.get('COUNTRY')
                )
                entities.append(entity)
                
                osm_data = item.get('osm', [])[i] if i < len(item.get('osm', [])) else None
                if osm_data:
                    geo_item = GeoModel(
                        city=osm_data.get('CITY'),
                        state=osm_data.get('STATE'),
                        country=osm_data.get('COUNTRY'),
                        coords=osm_data.get('COORDS')
                    )
                else:
                    geo_item = None
                geo.append(geo_item)
            
            result = AffiliationResult(
                raw_text=item['raw_text'],
                spans=item['span_entities'],
                entities=entities,
                geo=geo,
                ror_ids=item.get('ror', [])
            )
            response_results.append(result)
        
        return AffiliationResponse(results=response_results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing affiliations: {str(e)}")

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
