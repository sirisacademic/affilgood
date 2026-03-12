"""
Dense retriever for entity linking.

Uses FAISS index with SIRIS-Lab/affilgood-dense-retriever embeddings.

Multi-variant query strategy (from experiments — key to R@1=0.905):
  For each entity, multiple geographic variants are queried and results
  merged by max score per record ID.  This catches both:
  - Specific matches:  "CNRS [CITY] Rouen [COUNTRY] France"
  - HQ-based matches:  "CNRS [COUNTRY] France"  (ROR has city=Paris)
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseRetriever, Candidate
from ..index import DenseIndex, IndexBuilder, format_query_text, build_query_variants
from ..registry import RegistryRecord

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using FAISS with multi-variant queries.

    Parameters
    ----------
    dense_index : DenseIndex
        Pre-built FAISS index (from IndexBuilder).
    index_builder : IndexBuilder
        For encoding queries at search time.
    threshold : float
        Minimum similarity score to return (default: 0.3).
    country_boost : float
        Score boost when country matches NER context (default: 0.05).
    city_boost : float
        Score boost when city matches NER context (default: 0.03).
    verbose : bool
        Verbose logging.
    """

    def __init__(
        self,
        dense_index: DenseIndex,
        index_builder: IndexBuilder,
        *,
        threshold: float = 0.3,
        country_boost: float = 0.05,
        city_boost: float = 0.03,
        verbose: bool = False,
    ):
        self.dense_index = dense_index
        self.index_builder = index_builder
        self.threshold = threshold
        self.country_boost = country_boost
        self.city_boost = city_boost
        self.verbose = verbose

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Candidate]:
        """
        Retrieve candidates via multi-variant dense similarity.

        Builds multiple geo-variant queries (ORG+CITY+CO, ORG+CO, etc.)
        and merges results by max score per record ID.

        Parameters
        ----------
        query : str
            Organization name to match (raw, without tokens).
        top_k : int
            Max candidates to return.
        context : dict, optional
            NER context: {"country": "Spain", "city": "Barcelona"}

        Returns
        -------
        list of Candidate
            Sorted by score descending, filtered by threshold.
        """
        if not query or not query.strip():
            return []

        # Build multiple geo-variant queries
        variant_queries = build_query_variants(query, context)

        if self.verbose:
            for vq in variant_queries:
                logger.debug("DenseRetriever variant: %s", vq)

        # Over-fetch per variant (3x), then merge
        internal_k = top_k * 3

        # Merge across all variants: keep max score per record_id
        score_map: Dict[str, tuple] = {}  # record_id → (matched_text, score)

        for formatted_query in variant_queries:
            query_vec = self.index_builder.encode_query(formatted_query)
            raw_results = self.dense_index.search(query_vec, top_k=internal_k)

            for record_id, matched_text, score in raw_results:
                if record_id not in score_map or score > score_map[record_id][1]:
                    score_map[record_id] = (matched_text, score)

        # Convert to Candidates with context boosting
        candidates = []
        ctx = context or {}
        ctx_country = (ctx.get("country") or "").lower().strip()
        ctx_city = (ctx.get("city") or "").lower().strip()

        for record_id, (matched_text, score) in score_map.items():
            if score < self.threshold:
                continue

            record = self.dense_index.get_record(record_id)

            boosted_score = score
            metadata = {}

            if record is not None:
                metadata = {
                    "country": record.country,
                    "country_code": record.country_code,
                    "city": record.city,
                    "types": record.types,
                }

                if ctx_country and record.country.lower() == ctx_country:
                    boosted_score += self.country_boost
                elif ctx_country and record.country_code.lower() == ctx_country:
                    boosted_score += self.country_boost

                if ctx_city and record.city.lower() == ctx_city:
                    boosted_score += self.city_boost

            candidates.append(Candidate(
                id=record_id,
                name=record.name if record else matched_text,
                score=min(boosted_score, 1.0),
                source=record.source if record else "ror",
                matched_text=matched_text,
                retriever="dense",
                metadata=metadata,
            ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[:top_k]

        if self.verbose and candidates:
            top = candidates[0]
            logger.debug(
                "DenseRetriever -> '%s' (score=%.3f, id=%s) [%d variants, %d candidates]",
                top.name, top.score, top.id, len(variant_queries), len(candidates),
            )

        return candidates
