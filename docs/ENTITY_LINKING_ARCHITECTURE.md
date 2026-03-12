# Entity Linking Module — Architecture Draft

## Overview

The entity linking module matches NER-extracted ORG and SUB entities
against known registries (primarily ROR) to produce stable identifiers.

**Pipeline position:** after NER, before geocoding.

```
Span → Language → NER → **Entity Linking** → Geocoding → Output
```

## Design Principles

1. **Retrieval + Reranking** — two-stage architecture (like RAG)
2. **Pluggable retrievers** — sparse (Whoosh/BM25), dense (FAISS/HNSW), or both
3. **Pluggable rerankers** — cross-encoder, LLM-based, or none
4. **Registry-agnostic** — ROR is default, but architecture supports any registry
5. **Index-once, query-many** — indices are built once and cached to disk
6. **Context-aware** — linking uses NER context (country, city) to disambiguate


## Module Structure

```
affilgood/
├── components/
│   ├── entity_linking/
│   │   ├── __init__.py              # Public imports
│   │   │
│   │   ├── linker.py                # EntityLinker — orchestrator
│   │   │                            #   Manages retrieval → merge → rerank → select
│   │   │
│   │   ├── registry.py              # RegistryManager — data loading
│   │   │                            #   Downloads/loads ROR dump
│   │   │                            #   Normalizes to common schema
│   │   │                            #   Extensible for Wikidata, SICRIS, etc.
│   │   │
│   │   ├── index.py                 # IndexBuilder — builds & loads search indices
│   │   │                            #   FAISS flat / HNSW for dense
│   │   │                            #   Whoosh for sparse
│   │   │                            #   Handles encoding (sentence-transformers)
│   │   │
│   │   ├── retrievers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # BaseRetriever (protocol/ABC)
│   │   │   ├── dense.py             # DenseRetriever (FAISS/HNSW)
│   │   │   ├── sparse.py            # SparseRetriever (Whoosh/BM25)
│   │   │   └── combined.py          # CombinedRetriever (merge + dedupe)
│   │   │
│   │   ├── rerankers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py              # BaseReranker (protocol/ABC)
│   │   │   ├── cross_encoder.py     # CrossEncoderReranker
│   │   │   ├── direct_pair.py       # DirectPairReranker (text similarity)
│   │   │   └── llm.py               # LLMReranker (generative, future)
│   │   │
│   │   └── data/
│   │       ├── ror/                  # Auto-downloaded ROR dump + indices
│   │       │   ├── ror_dump.json     # Raw ROR data (auto-downloaded)
│   │       │   ├── ror_records.jsonl  # Normalized records
│   │       │   ├── faiss.index       # Dense index
│   │       │   ├── faiss_ids.json    # ID mapping for FAISS
│   │       │   └── whoosh_index/     # Sparse index directory
│   │       └── cache/
│   │           └── linking_cache.sqlite  # Query → result cache
│   │
│   ├── geocode.py
│   ├── language.py
│   ├── ner.py
│   └── span.py
```


## Core Classes

### 1. RegistryManager — Data Layer

Responsible for downloading, loading, and normalizing registry data.

```python
class RegistryManager:
    """
    Manages registry data (ROR, Wikidata, custom).

    Downloads ROR dump on first use, caches locally.
    Normalizes all registries to a common record schema.
    """

    def __init__(self, data_dir=None, registries=("ror",)):
        ...

    def get_records(self, registry="ror") -> List[RegistryRecord]:
        """Return normalized records for indexing/search."""
        ...

    def download_ror(self, force=False):
        """Download latest ROR dump from Zenodo."""
        ...
```

**Common record schema** (what every registry normalizes to):

```python
@dataclass
class RegistryRecord:
    id: str                    # "https://ror.org/052gg0110"
    name: str                  # "University of Oxford"
    aliases: List[str]         # ["Oxford University", "Oxon"]
    acronyms: List[str]        # ["UOxf"]
    labels: List[str]          # Translated names: ["Université d'Oxford"]
    country: str               # "United Kingdom"
    country_code: str          # "GB"
    city: str                  # "Oxford"
    types: List[str]           # ["Education", "Facility"]
    status: str                # "active"
    source: str                # "ror"
    url: str                   # "http://www.ox.ac.uk"
    parent_id: Optional[str]   # For hierarchical orgs
```

**ROR-specific notes:**
- Download from Zenodo API (latest release)
- ~110k records, ~200MB uncompressed JSON
- Contains: names, aliases, acronyms, labels (translations),
  country, city, types, relationships, URLs, external IDs
- Update frequency: ~monthly


### 2. IndexBuilder — Index Layer

Builds search indices from registry records.

```python
class IndexBuilder:
    """
    Builds and manages search indices for entity linking.

    Index types:
    - Dense: FAISS (flat or HNSW) with sentence-transformer embeddings
    - Sparse: Whoosh full-text with BM25 scoring

    Indices are built once and cached to disk.
    Rebuild triggered by: new ROR dump, config change, or explicit request.
    """

    def __init__(self, data_dir, encoder_model=None):
        ...

    def build_dense_index(
        self,
        records: List[RegistryRecord],
        index_type: str = "hnsw",   # "flat", "hnsw", "ivf"
        **kwargs,
    ) -> DenseIndex:
        ...

    def build_sparse_index(
        self,
        records: List[RegistryRecord],
    ) -> SparseIndex:
        ...

    def load_or_build(self, records, index_type, rebuild=False):
        """Load existing index from disk, or build if missing."""
        ...
```

**Dense index details:**

```
Encoding pipeline:
  record.name + " " + ", ".join(record.aliases)
    → sentence-transformer encoder (default: all-MiniLM-L6-v2)
    → 384-dim float32 vector
    → FAISS index

Index types:
  - IndexFlatIP:    exact inner product (small registries, <50k)
  - IndexHNSWFlat:  approximate NN (default for ROR, ~110k records)
                    params: M=32, efConstruction=200, efSearch=128
  - IndexIVFFlat:   inverted file (very large registries, >500k)
                    params: nlist=sqrt(n), nprobe=10

Stored on disk:
  - faiss.index          — the FAISS index file
  - faiss_ids.json       — list[str] mapping FAISS row → record.id
  - faiss_meta.json      — encoder model, dim, index type, record count, build date
```

**Sparse index details:**

```
Whoosh schema:
  - id:       STORED (registry ID)
  - name:     TEXT (analyzed, BM25 scoring)
  - aliases:  TEXT (analyzed)
  - acronyms: KEYWORD (exact + lowercase)
  - country:  TEXT (for context filtering)
  - city:     TEXT (for context filtering)

Stored on disk:
  - whoosh_index/       — Whoosh index directory
```

**Index text composition for dense encoding:**

For each record, we create multiple searchable texts:
```python
texts_to_encode = []
# Primary: official name
texts_to_encode.append(record.name)
# Each alias as separate entry (same ID)
for alias in record.aliases + record.labels:
    texts_to_encode.append(alias)
# Acronyms (if ≥2 chars)
for acr in record.acronyms:
    if len(acr) >= 2:
        texts_to_encode.append(acr)
```

This means FAISS will have MORE rows than records (one per name variant).
The `faiss_ids.json` maps each row back to the record ID.


### 3. BaseRetriever / Retrievers — Retrieval Layer

```python
class BaseRetriever(ABC):
    """Protocol for all retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        context: Optional[Dict] = None,  # {"country": "Spain", "city": "Barcelona"}
    ) -> List[Candidate]:
        ...

@dataclass
class Candidate:
    id: str              # "https://ror.org/052gg0110"
    name: str            # "University of Oxford"
    score: float         # retrieval score (0–1 normalized)
    source: str          # "dense", "sparse", "combined"
    matched_text: str    # which name variant matched
    metadata: Dict       # country, city, types, etc.
```

**DenseRetriever:**
```python
class DenseRetriever(BaseRetriever):
    """
    Dense retrieval using FAISS.

    Encodes query with same sentence-transformer used at index time.
    Returns top-k by cosine/inner-product similarity.
    """
    def __init__(self, index, encoder, id_mapping, threshold=0.3):
        ...

    def retrieve(self, query, top_k=10, context=None):
        # 1. Encode query → vector
        # 2. Search FAISS index
        # 3. Map FAISS IDs → record IDs
        # 4. Filter by threshold
        # 5. (Optional) boost score if context.country matches
        ...
```

**SparseRetriever:**
```python
class SparseRetriever(BaseRetriever):
    """
    Sparse retrieval using Whoosh/BM25.

    Good for exact/partial name matches, acronyms.
    """
    def __init__(self, index_dir, threshold=0.25):
        ...

    def retrieve(self, query, top_k=10, context=None):
        # 1. Build Whoosh query (name OR aliases OR acronyms)
        # 2. If context has country → add country filter/boost
        # 3. Search and score
        ...
```

**CombinedRetriever:**
```python
class CombinedRetriever(BaseRetriever):
    """
    Merges results from multiple retrievers.

    Deduplicates by record ID, combines scores.
    """
    def __init__(self, retrievers: List[BaseRetriever], weights=None):
        ...

    def retrieve(self, query, top_k=10, context=None):
        # 1. Call each retriever
        # 2. Merge by record ID
        # 3. Weighted score combination
        # 4. Sort by combined score
        # 5. Return top_k
        ...
```


### 4. BaseReranker / Rerankers — Reranking Layer

```python
class BaseReranker(ABC):
    """Protocol for all rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        context: Optional[Dict] = None,
    ) -> List[Candidate]:
        """Rerank candidates, return sorted by new score."""
        ...
```

**CrossEncoderReranker:**
```python
class CrossEncoderReranker(BaseReranker):
    """
    Rerank using a cross-encoder model.

    Scores (query, candidate_name) pairs jointly.
    More accurate than bi-encoder but slower.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """
    def __init__(self, model_name=None, threshold=0.5, batch_size=32):
        ...

    def rerank(self, query, candidates, context=None):
        # 1. Create pairs: [(query, cand.name) for cand in candidates]
        # 2. Score all pairs with cross-encoder
        # 3. Update candidate scores
        # 4. Filter by threshold
        # 5. Sort descending
        ...
```

**DirectPairReranker:**
```python
class DirectPairReranker(BaseReranker):
    """
    Rerank using direct text comparison.

    Generates name variants (lowercase, no punct, etc.)
    and compares with candidate names.
    Lightweight, no model needed.
    """
    ...
```

**LLMReranker (future):**
```python
class LLMReranker(BaseReranker):
    """
    Rerank using an LLM to select the best candidate.

    Presents candidates to an LLM with context and asks
    it to pick the best match. Most accurate but slowest.
    """
    ...
```


### 5. EntityLinker — Orchestrator

```python
class EntityLinker:
    """
    Entity linking component for the AffilGood pipeline.

    Orchestrates: retrieve → merge → rerank → select.

    Parameters
    ----------
    retriever : str or BaseRetriever or list
        "dense" (default), "sparse", "combined", ["dense", "sparse"],
        or a custom BaseRetriever instance.
    reranker : str or BaseReranker or None
        "cross_encoder" (default), "direct_pair", None,
        or a custom BaseReranker instance.
    data_source : str
        "ror" (default), "wikidata", etc.
    threshold : float
        Minimum score to accept a match (default: 0.5).
    top_k_retrieve : int
        Candidates to retrieve before reranking (default: 10).
    top_k_rerank : int
        Candidates to rerank (default: 5).
    """

    def __init__(
        self,
        *,
        retriever="dense",
        reranker="cross_encoder",
        data_source="ror",
        threshold=0.5,
        top_k_retrieve=10,
        top_k_rerank=5,
        data_dir=None,
        encoder_model=None,
        index_type="hnsw",
        rebuild_index=False,
        verbose=False,
    ):
        # 1. Load registry data via RegistryManager
        # 2. Build or load indices via IndexBuilder
        # 3. Initialize retriever(s)
        # 4. Initialize reranker (if any)
        ...

    def link(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Pipeline interface.

        For each item, links ORG and SUB entities to registry IDs.

        Populates item["entity_linking"] with:
            {
                "institutions": [
                    {
                        "query": "Universitat Autònoma de Barcelona",
                        "id": "https://ror.org/052gg0110",
                        "name": "Autonomous University of Barcelona",
                        "score": 0.92,
                        "source": "ror",
                        "retriever": "dense",
                        "reranker": "cross_encoder",
                    }
                ],
                "subunits": [
                    {
                        "query": "Facultat de Biomedicina",
                        "id": None,           # subunits often don't have ROR
                        "score": 0.0,
                        "source": None,
                    }
                ]
            }
        """
        ...

    def link_one(
        self,
        entity_name: str,
        context: Optional[Dict] = None,
    ) -> Optional[Candidate]:
        """
        Link a single entity name (standalone API).

        context: {"country": "Spain", "city": "Barcelona"}
        """
        # 1. Retrieve candidates
        # 2. Rerank (if reranker configured)
        # 3. Return top candidate if above threshold, else None
        ...
```


## Pipeline Integration

### In pipeline.py

```python
# After NER, before geocoding:

if self.entity_linker is not None:
    if self.verbose:
        print("[Pipeline] Entity Linking")
    flat_items = self.entity_linker.link(flat_items)
```

### Context passing

The linker extracts NER context for disambiguation:

```python
# Inside EntityLinker.link(), for each item:
ner = item.get("ner", [{}])[0]
context = {
    "country": (ner.get("COUNTRY") or [None])[0],
    "city": (ner.get("CITY") or [None])[0],
    "region": (ner.get("REGION") or [None])[0],
}

# This context helps disambiguate:
#   "University of Barcelona" + country="Spain"
#   → UAB (not University of Barcelona in Venezuela)
```


## API Configuration

```python
# Default: dense retriever + cross-encoder reranker on ROR
ag = AffilGood(enable_entity_linking=True)

# Sparse only (lightweight)
ag = AffilGood(
    enable_entity_linking=True,
    linking_config={
        "retriever": "sparse",
        "reranker": None,
    },
)

# Combined retrieval + cross-encoder
ag = AffilGood(
    enable_entity_linking=True,
    linking_config={
        "retriever": "combined",   # dense + sparse
        "reranker": "cross_encoder",
        "threshold": 0.6,
    },
)

# Custom retriever
from my_module import MyRetriever
ag = AffilGood(
    enable_entity_linking=True,
    linking_config={
        "retriever": MyRetriever(my_params=True),
        "reranker": "cross_encoder",
    },
)
```


## Output Schema

### In normalized output (output.py)

```python
"institutions": [
    {
        "name": "Autonomous University of Barcelona",  # from registry
        "id": "https://ror.org/052gg0110",
        "confidence": 0.92,
        "source": "ror",
    }
],
"subunits": [
    {
        "name": "Facultat de Biomedicina",             # from NER (no registry match)
        "confidence": 0.999,
        "source": "ner",
    }
]
```

When entity linking is enabled AND finds a match:
- `institutions[].name` → registry canonical name (not NER text)
- `institutions[].id` → registry ID (ROR URL)
- `institutions[].confidence` → linking score
- `institutions[].source` → "ror", "wikidata", etc.

When linking is enabled but no match found:
- Falls back to NER data (same as current behavior)
- `id` → None
- `source` → "ner"


## Index Sizing & Performance Estimates

```
ROR (~110k records, ~300k name variants):

Dense (FAISS HNSW, 384-dim):
  - Index size on disk:   ~500 MB
  - Build time:           ~5-10 min (CPU), ~1-2 min (GPU)
  - Query time:           ~5-20 ms per query
  - Memory at runtime:    ~600 MB

Sparse (Whoosh):
  - Index size on disk:   ~50-100 MB
  - Build time:           ~1-2 min
  - Query time:           ~1-5 ms per query
  - Memory at runtime:    ~200 MB

Cross-encoder reranking (5 candidates):
  - Model size:           ~80 MB
  - Time per query:       ~20-50 ms (GPU), ~100-300 ms (CPU)
```


## Implementation Order

### Phase 1: Core (this session)
1. RegistryRecord dataclass + RegistryManager (ROR download + normalize)
2. BaseRetriever protocol + DenseRetriever (FAISS HNSW)
3. IndexBuilder (encode + build FAISS)
4. EntityLinker orchestrator (retrieve only, no reranking)
5. Wire into pipeline

### Phase 2: Sparse + Combined
6. SparseRetriever (Whoosh)
7. CombinedRetriever (merge logic)

### Phase 3: Reranking
8. BaseReranker protocol
9. CrossEncoderReranker
10. DirectPairReranker

### Phase 4: Extensions
11. LLMReranker
12. Wikidata data source
13. Custom data sources