"""
Index building and management for entity linking.

Handles:
- Encoding registry records with sentence-transformers
- Building FAISS indices (flat, HNSW, IVF)
- Persisting indices to disk
- Loading pre-built indices

Text formatting strategy (matching SIRIS-Lab/affilgood-dense-retriever):
    The encoder was trained with structured tokens:
        [MENTION] name [ACRONYM] acr [CITY] city [COUNTRY] country [PARENT] parent

    Index side (from registry records):
        Each record produces one FULL entry (canonical name + all metadata tokens)
        plus additional entries for each alias/label (with same metadata).

    Query side (from NER output):
        "[MENTION] org_name [CITY] Barcelona [COUNTRY] Spain"
        Context tokens are added from NER entities.
        Multiple geo variants queried and merged by max score.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import RegistryRecord

logger = logging.getLogger(__name__)

# Default encoder: SIRIS-Lab affilgood dense retriever (1024-dim, XLM-RoBERTa)
DEFAULT_ENCODER = "SIRIS-Lab/affilgood-dense-retriever"


# ------------------------------------------------------------------
# Structured token formatting
# ------------------------------------------------------------------

def format_record_text(name: str, meta_suffix: str = "") -> str:
    """
    Format a registry record name for encoding.

    Produces: "[MENTION] University of Oxford [ACRONYM] UOxf [CITY] Oxford [COUNTRY] United Kingdom"

    Parameters
    ----------
    name : str
        The name variant to encode (canonical, alias, label, or acronym).
    meta_suffix : str
        Pre-built metadata string: "[ACRONYM] X [CITY] Y [COUNTRY] Z"
    """
    text = f"[MENTION] {name.strip()}"
    if meta_suffix:
        text = f"{text} {meta_suffix}"
    return text


def format_query_text(
    entity_name: str,
    context: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format a query (NER entity + context) for encoding.

    Token order matches SIRIS-Lab/affilgood-dense-retriever training:
        [MENTION] org [CITY] city [COUNTRY] country

    Parameters
    ----------
    entity_name : str
        Organization name from NER.
    context : dict, optional
        NER context: {"country": "Spain", "city": "Barcelona"}
    """
    parts = [f"[MENTION] {entity_name.strip()}"]

    if context:
        if context.get("city"):
            parts.append(f"[CITY] {context['city']}")
        if context.get("country"):
            parts.append(f"[COUNTRY] {context['country']}")

    return " ".join(parts)


def build_query_variants(
    entity_name: str,
    context: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Build multiple geo-variant queries for a single entity.

    Multi-variant strategy (from experiments — critical for R@1=0.905):
      1. ORG + CITY + COUNTRY  (most specific)
      2. ORG + COUNTRY         (catches orgs whose ROR city ≠ affiliation city)
      3. ORG + REGION + COUNTRY (if region available)
      4. ORG only              (fallback when no geo info)

    Each variant is formatted with structured tokens.
    Retrieval results are merged by max score per record ID.

    Returns
    -------
    list of str
        Formatted query strings (deduplicated).
    """
    ctx = context or {}
    city = (ctx.get("city") or "").strip()
    country = (ctx.get("country") or "").strip()
    region = (ctx.get("region") or "").strip()

    variants = []
    seen = set()

    def _add(c=None, co=None):
        q_ctx = {}
        if c:
            q_ctx["city"] = c
        if co:
            q_ctx["country"] = co
        text = format_query_text(entity_name, q_ctx if q_ctx else None)
        if text not in seen:
            seen.add(text)
            variants.append(text)

    # Most specific: ORG + CITY + COUNTRY
    if city and country:
        _add(c=city, co=country)

    # ORG + COUNTRY only (critical: CNRS→Paris, INSERM→Paris)
    if country:
        _add(co=country)

    # ORG + REGION + COUNTRY
    if region and country:
        _add(c=region, co=country)

    # ORG + CITY only (when no country)
    if city and not country:
        _add(c=city)

    # ORG only (fallback)
    if not city and not country:
        _add()

    return variants


class IndexBuilder:
    """
    Builds and manages search indices from registry records.

    Parameters
    ----------
    data_dir : Path
        Directory for storing indices.
    encoder_model : str or None
        Sentence-transformer model name.
        Default: SIRIS-Lab/affilgood-dense-retriever (1024-dim).
    device : str or None
        Device for encoding ("cpu", "cuda", None=auto).
    batch_size : int
        Batch size for encoding.
    verbose : bool
        Verbose logging.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        encoder_model: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 256,
        verbose: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_model_name = encoder_model or DEFAULT_ENCODER
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        # Lazy-loaded encoder
        self._encoder = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_dense_index(
        self,
        records: List[RegistryRecord],
        *,
        index_type: str = "hnsw",
        rebuild: bool = False,
    ) -> "DenseIndex":
        """
        Build or load a FAISS dense index.

        Parameters
        ----------
        records : list of RegistryRecord
        index_type : str
            "flat" (exact), "hnsw" (default, approximate), "ivf"
        rebuild : bool
            Force rebuild even if index exists on disk.

        Returns
        -------
        DenseIndex
        """
        index_dir = self.data_dir / "dense"
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / "faiss.index"
        ids_path = index_dir / "faiss_ids.json"
        texts_path = index_dir / "faiss_texts.json"
        meta_path = index_dir / "faiss_meta.json"

        # Check for existing index
        if not rebuild and all(
            p.exists() for p in (index_path, ids_path, texts_path, meta_path)
        ):
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("encoder") == self.encoder_model_name
                and meta.get("index_type") == index_type
            ):
                if self.verbose:
                    print(
                        f"[Index] Loading existing dense index: "
                        f"{meta.get('num_vectors', '?')} vectors"
                    )
                return DenseIndex.load(index_dir)

        # Build from scratch
        if self.verbose:
            print(f"[Index] Building dense index ({index_type}) from {len(records)} records...")

        t0 = time.time()

        # 1. Expand records into (text, record_id) pairs with structured tokens
        texts, record_ids, record_map = self._expand_records(records)

        if self.verbose:
            print(f"[Index]   {len(texts)} name variants from {len(records)} records")
            if texts:
                print(f"[Index]   Sample: {texts[0][:120]}...")

        # 2. Encode all texts
        embeddings = self._encode(texts)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"[Index]   Encoded in {elapsed:.1f}s -> shape {embeddings.shape}")

        # 3. Build FAISS index
        import faiss

        dim = embeddings.shape[1]

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
        elif index_type == "ivf":
            n = embeddings.shape[0]
            nlist = max(1, int(np.sqrt(n)))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.nprobe = min(10, nlist)
        else:
            raise ValueError(f"Unknown index_type: '{index_type}'")

        # Normalize for cosine similarity (inner product on unit vectors)
        # Note: SIRIS encoder already has Normalize() layer, but
        # faiss.normalize_L2 is idempotent on unit vectors, so safe to call.
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"[Index]   Built {index_type} index in {elapsed:.1f}s total")

        # 4. Save to disk
        faiss.write_index(index, str(index_path))

        ids_path.write_text(json.dumps(record_ids, ensure_ascii=False))
        texts_path.write_text(json.dumps(texts, ensure_ascii=False))

        meta = {
            "encoder": self.encoder_model_name,
            "dim": dim,
            "index_type": index_type,
            "num_vectors": len(texts),
            "num_records": len(records),
            "build_time": round(time.time() - t0, 1),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        if self.verbose:
            print(f"[Index]   Saved to {index_dir}")

        return DenseIndex(
            index=index,
            record_ids=record_ids,
            texts=texts,
            record_map=record_map,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Record expansion with structured tokens
    # ------------------------------------------------------------------

    def _expand_records(
        self, records: List[RegistryRecord],
    ) -> Tuple[List[str], List[str], Dict[str, RegistryRecord]]:
        """
        Expand records into (texts, record_ids, record_map).

        Uses structured token format matching the SIRIS encoder training::

            [MENTION] name [ACRONYM] acr [CITY] city [COUNTRY] country

        Each record produces:
        1. One FULL entry: canonical name + all metadata tokens
        2. One entry per alias/label: alias + same metadata tokens
        3. One entry per acronym (>=2 chars): acronym + metadata tokens
        """
        texts: List[str] = []
        record_ids: List[str] = []
        record_map: Dict[str, RegistryRecord] = {}

        # First pass: build record_map so parent lookups work
        for record in records:
            record_map[record.id] = record

        # Second pass: expand
        for record in records:
            # Build shared metadata suffix
            meta_parts = []
            for acr in record.acronyms:
                if len(acr) >= 2:
                    meta_parts.append(f"[ACRONYM] {acr}")
            if record.city:
                meta_parts.append(f"[CITY] {record.city}")
            if record.country:
                meta_parts.append(f"[COUNTRY] {record.country}")
            if record.parent_id:
                parent = record_map.get(record.parent_id)
                if parent:
                    meta_parts.append(f"[PARENT] {parent.name}")

            meta_suffix = " ".join(meta_parts)

            # 1. Canonical name (full entry)
            texts.append(format_record_text(record.name, meta_suffix))
            record_ids.append(record.id)

            # Track seen names to avoid duplicates
            seen = {record.name.strip().lower()}

            # 2. Each alias
            for alias in record.aliases:
                alias_lower = alias.strip().lower()
                if alias_lower and alias_lower not in seen:
                    seen.add(alias_lower)
                    texts.append(format_record_text(alias, meta_suffix))
                    record_ids.append(record.id)

            # 3. Each translated label
            for label in record.labels:
                label_lower = label.strip().lower()
                if label_lower and label_lower not in seen:
                    seen.add(label_lower)
                    texts.append(format_record_text(label, meta_suffix))
                    record_ids.append(record.id)

            # 4. Acronyms as standalone entries
            for acr in record.acronyms:
                acr_lower = acr.strip().lower()
                if len(acr) >= 2 and acr_lower not in seen:
                    seen.add(acr_lower)
                    texts.append(format_record_text(acr, meta_suffix))
                    record_ids.append(record.id)

        return texts, record_ids, record_map

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _get_encoder(self):
        """Lazy-load sentence-transformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            device = self.device
            if device is None:
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            if self.verbose:
                print(f"[Index] Loading encoder: {self.encoder_model_name} on {device}")

            self._encoder = SentenceTransformer(
                self.encoder_model_name, device=device,
            )

        return self._encoder

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors using sentence-transformer."""
        encoder = self._get_encoder()

        embeddings = encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.verbose,
            normalize_embeddings=False,  # We normalize after for FAISS
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string (already formatted with tokens).
        Used at search time.
        """
        encoder = self._get_encoder()

        embedding = encoder.encode(
            [query],
            normalize_embeddings=False,
            convert_to_numpy=True,
        )

        vec = embedding.astype(np.float32)

        import faiss
        faiss.normalize_L2(vec)

        return vec


# ------------------------------------------------------------------
# Dense Index wrapper
# ------------------------------------------------------------------

class DenseIndex:
    """
    Wrapper around a FAISS index with ID mapping.
    """

    def __init__(
        self,
        index,
        record_ids: List[str],
        texts: List[str],
        record_map: Dict[str, RegistryRecord],
        meta: Dict[str, Any],
    ):
        self.index = index
        self.record_ids = record_ids
        self.texts = texts
        self.record_map = record_map
        self.meta = meta

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """
        Search the index.

        Returns list of (record_id, matched_text, score).
        Deduplicates by record_id (keeps highest score).
        """
        k = min(top_k * 3, len(self.record_ids))
        scores, indices = self.index.search(query_vector, k)

        seen: Dict[str, Tuple[str, float]] = {}

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            record_id = self.record_ids[idx]
            matched_text = self.texts[idx]
            score_val = float(score)

            if record_id not in seen or score_val > seen[record_id][1]:
                seen[record_id] = (matched_text, score_val)

        results = [
            (rid, text, score)
            for rid, (text, score) in seen.items()
        ]
        results.sort(key=lambda x: x[2], reverse=True)

        return results[:top_k]

    def get_record(self, record_id: str) -> Optional[RegistryRecord]:
        """Look up a record by ID."""
        return self.record_map.get(record_id)

    @classmethod
    def load(cls, index_dir: Path) -> "DenseIndex":
        """Load a pre-built index from disk."""
        import faiss

        index_dir = Path(index_dir)

        index = faiss.read_index(str(index_dir / "faiss.index"))
        record_ids = json.loads((index_dir / "faiss_ids.json").read_text())
        texts = json.loads((index_dir / "faiss_texts.json").read_text())
        meta = json.loads((index_dir / "faiss_meta.json").read_text())

        return cls(
            index=index,
            record_ids=record_ids,
            texts=texts,
            record_map={},
            meta=meta,
        )

    @property
    def num_vectors(self) -> int:
        return len(self.record_ids)

    @property
    def num_records(self) -> int:
        return len(set(self.record_ids))