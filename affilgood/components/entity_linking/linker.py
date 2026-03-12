"""
Entity linking orchestrator for AffilGood.

Implements a three-stage cascade pipeline derived from extensive
experiments on ROR entity linking (Acc@1 0.91 → 0.93+ target):

  Stage 1 — Direct Match
    Exact name/alias/acronym + country lookup.
    ~35% coverage at ~98% precision.  Instant, no model needed.

  Stage 2 — Dense Retrieval + Cross-Encoder Reranking
    AffilGood dense retriever (FAISS HNSW) → cross-encoder reranker.
    Retrieval and reranker scores are fused (weighted combination)
    to prevent the reranker from overriding correct retriever results.

  Stage 3 — LLM Judge (optional, for low-confidence cases)
    Small instruction-following LLM sees all candidates simultaneously
    (listwise comparison).  Handles acronym confusion, same-name
    disambiguation, and French UMR chains.

Pipeline position: after NER, before geocoding.
"""

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .registry import RegistryManager, RegistryRecord
from .index import IndexBuilder, DenseIndex, format_query_text
from .retrievers.base import BaseRetriever, Candidate
from .retrievers.dense import DenseRetriever
from .rerankers.base import BaseReranker

logger = logging.getLogger(__name__)


# ================================================================
# Direct Match Index
# ================================================================

class DirectMatcher:
    """
    Exact-match lookup: (name_lower, country_lower) → unique ROR ID.

    Indexes canonical names, aliases, labels, and acronyms from BOTH
    active and inactive records.  When an inactive record matches,
    follows the successor chain to return the active successor.

    This handles cases like INRA (inactive) → INRAE (active).

    From experiments: ~35% entity coverage at ~98% precision.
    """

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Unicode-safe normalization for matching.

        Transformations (applied to BOTH index and query sides):
          1. Casefold:     "LAmCoS" → "lamcos", "LAMCOS" → "lamcos"
          2. Strip accents: "Université" → "universite", "TÜRKİYE" → "turkiye"
          3. Remove dots:  "C.N.R.S." → "CNRS", "I.N.R.A." → "INRA"
          4. Hyphens→space: "INSA-Lyon" → "INSA Lyon"
          5. Collapse spaces: "INSA  Lyon" → "INSA Lyon"
        """
        import unicodedata
        import re
        if not text:
            return ""
        # Casefold (handles Turkish İ, German ß, etc.)
        t = text.strip().casefold()
        # NFKD decomposition + strip combining marks (accents)
        t = unicodedata.normalize("NFKD", t)
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        t = unicodedata.normalize("NFC", t)
        # Remove dots: "c.n.r.s." → "cnrs"
        t = t.replace(".", "")
        # Hyphens/dashes → space: "insa-lyon" → "insa lyon"
        t = re.sub(r"[-–—]", " ", t)
        # Collapse whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def __init__(
        self,
        active_records: List[RegistryRecord],
        all_records: Optional[List[RegistryRecord]] = None,
        verbose: bool = False,
    ):
        self._nc: Dict[Tuple[str, str], Set[str]] = {}
        self._name_to_ids: Dict[str, Set[str]] = {}
        self._id_to_record: Dict[str, RegistryRecord] = {}
        self._successor: Dict[str, str] = {}  # inactive_id → successor_id

        # Build record map from ALL records
        all_recs = all_records if all_records is not None else active_records
        for rec in all_recs:
            self._id_to_record[rec.id] = rec

        # Build successor map from inactive records
        n_successors = 0
        for rec in all_recs:
            if rec.status != "active" and rec.successor_id:
                self._successor[rec.id] = rec.successor_id
                n_successors += 1

        # Index ALL records (active + inactive) for name lookup
        for rec in all_recs:
            country = self._normalize(rec.country)
            cc = rec.country_code.strip().lower()

            for name in rec.all_names():
                nl = self._normalize(name)
                if not nl:
                    continue

                self._name_to_ids.setdefault(nl, set()).add(rec.id)

                if country:
                    self._nc.setdefault((nl, country), set()).add(rec.id)
                if cc:
                    self._nc.setdefault((nl, cc), set()).add(rec.id)

        if verbose:
            print(
                f"[DirectMatcher] {len(self._name_to_ids)} unique names, "
                f"{len(self._nc)} (name,country) pairs, "
                f"{len(self._id_to_record)} records, "
                f"{n_successors} successor mappings"
            )

    def _resolve_successor(self, rid: str, max_depth: int = 5) -> str:
        """Follow successor chain to find active record."""
        visited = set()
        current = rid
        for _ in range(max_depth):
            if current in visited:
                break
            visited.add(current)
            rec = self._id_to_record.get(current)
            if rec is None or rec.status == "active":
                return current
            successor = self._successor.get(current)
            if not successor:
                return current
            current = successor
        return current

    def match(
        self,
        entity_name: str,
        country: Optional[str] = None,
    ) -> Optional[Candidate]:
        """
        Try exact match.  If the matched record is inactive,
        follows the successor chain and returns the active successor.
        """
        nl = self._normalize(entity_name)
        if not nl:
            return None

        if country:
            cl = self._normalize(country)
            ids = self._nc.get((nl, cl), set())
            if len(ids) == 1:
                rid = next(iter(ids))
                rec = self._id_to_record[rid]

                # Follow successor if inactive
                if rec.status != "active" and rid in self._successor:
                    rid = self._resolve_successor(rid)
                    rec = self._id_to_record.get(rid, rec)

                return Candidate(
                    id=rid,
                    name=rec.name,
                    score=1.0,
                    source=rec.source,
                    matched_text=entity_name,
                    retriever="direct_match",
                    metadata={
                        "country": rec.country,
                        "country_code": rec.country_code,
                        "city": rec.city,
                        "types": rec.types,
                    },
                )
        return None



# ================================================================
# Entity Linker — Cascade Orchestrator
# ================================================================

class EntityLinker:
    """
    Entity linking component for the AffilGood pipeline.

    Implements a cascade: direct match → retrieval+reranking → LLM judge.

    Parameters
    ----------
    retriever : str or BaseRetriever
        "dense" (default) or a custom BaseRetriever instance.
    reranker : str or BaseReranker or None
        "cross_encoder" (default), "none"/None (retrieval only),
        or a custom BaseReranker instance.
    reranker_model : str or None
        Model name override for cross-encoder reranker.
    llm_judge : str or bool or None
        LLM model for low-confidence fallback.
        True → default model, str → specific model, None/False → disabled.
    llm_judge_model : str or None
        Alias for llm_judge (for clarity in config).
    llm_threshold : float
        Score below which LLM judge is invoked (default: 0.5).
    reranker_threshold : float
        Minimum reranker score to accept a match (default: 0.038).
        From experiments: jina_comet_large optimal threshold = 0.03784.
        If the reranker's best candidate scores below this, the match
        is rejected (or routed to LLM judge if enabled).
    data_source : str
        "ror" (default).
    threshold : float
        Minimum score to accept a final match (default: 0.3).
        From experiments: jina_comet_large threshold = 0.038.
    top_k : int
        Candidates to retrieve (default: 10).
    data_dir : str or Path or None
        Directory for registry data and indices.
    encoder_model : str or None
        Sentence-transformer model for dense encoding.
    index_type : str
        FAISS index type: "hnsw" (default).
    rebuild_index : bool
        Force rebuild of indices.
    device : str or None
        Device for models.
    verbose : bool
        Verbose logging.
    """

    def __init__(
        self,
        *,
        retriever: Union[str, BaseRetriever] = "dense",
        reranker: Union[str, BaseReranker, None] = "cross_encoder",
        reranker_model: Optional[str] = None,
        llm_judge: Union[str, bool, None] = None,
        llm_judge_model: Optional[str] = None,
        llm_threshold: float = 0.5,
        reranker_threshold: float = 0.038,
        data_source: str = "ror",
        threshold: float = 0.3,
        top_k: int = 10,
        data_dir: Optional[str] = None,
        encoder_model: Optional[str] = None,
        index_type: str = "hnsw",
        rebuild_index: bool = False,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.threshold = threshold
        self.top_k = top_k
        self.verbose = verbose
        self.data_source = data_source
        self.reranker_threshold = reranker_threshold
        self.llm_threshold = llm_threshold

        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self._data_dir = Path(data_dir)

        # --- 1. Load registry + build direct matcher ---
        registry = RegistryManager(data_dir=self._data_dir, verbose=verbose)
        records = registry.get_records(data_source, active_only=True)

        # Also load inactive records for successor resolution in direct match
        all_records = registry.get_records(data_source, active_only=False)

        if verbose:
            n_inactive = len(all_records) - len(records)
            print(f"[EntityLinker] Loaded {len(records)} active + "
                  f"{n_inactive} inactive {data_source} records")

        self._direct_matcher = DirectMatcher(
            active_records=records,
            all_records=all_records,
            verbose=verbose,
        )
        self._records = records
        self._record_map = {r.id: r for r in all_records}  # include inactive for lookups

        # --- 2. Initialize retriever ---
        if isinstance(retriever, BaseRetriever):
            self._retriever = retriever
        elif retriever == "dense":
            self._retriever = self._init_dense_retriever(
                records=records,
                encoder_model=encoder_model,
                index_type=index_type,
                rebuild_index=rebuild_index,
                device=device,
            )
        else:
            raise ValueError(f"Unknown retriever: '{retriever}'")

        # --- 3. Initialize reranker ---
        self._reranker: Optional[BaseReranker] = None
        if isinstance(reranker, BaseReranker):
            self._reranker = reranker
        elif reranker == "cross_encoder":
            from .rerankers.cross_encoder import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                model_name=reranker_model,
                device=device,
                verbose=verbose,
            )
        elif reranker in (None, "none", "None", False):
            self._reranker = None
        else:
            raise ValueError(f"Unknown reranker: '{reranker}'")

        # --- 4. Initialize LLM judge (lazy — loaded on first use) ---
        self._llm_judge: Optional[BaseReranker] = None
        self._llm_model_name: Optional[str] = None

        llm_spec = llm_judge_model or llm_judge
        if llm_spec is True:
            self._llm_model_name = None  # use default
        elif isinstance(llm_spec, str):
            self._llm_model_name = llm_spec
        # else: disabled

        self._llm_device = device

    # ------------------------------------------------------------------
    # Retriever initialization
    # ------------------------------------------------------------------

    def _init_dense_retriever(
        self,
        records: List[RegistryRecord],
        encoder_model: Optional[str],
        index_type: str,
        rebuild_index: bool,
        device: Optional[str],
    ) -> DenseRetriever:
        """Initialize dense retriever: build/load index → create retriever."""
        index_builder = IndexBuilder(
            data_dir=self._data_dir / self.data_source,
            encoder_model=encoder_model,
            device=device,
            verbose=self.verbose,
        )

        dense_index = index_builder.build_dense_index(
            records, index_type=index_type, rebuild=rebuild_index,
        )

        if not dense_index.record_map:
            # Use full record map (active + inactive) since FAISS may
            # contain inactive records that need successor resolution
            dense_index.record_map = self._record_map

        return DenseRetriever(
            dense_index=dense_index,
            index_builder=index_builder,
            threshold=self.threshold * 0.5,  # retriever threshold is looser
            verbose=self.verbose,
        )

    def _ensure_llm_judge(self):
        """Lazy-load LLM judge on first use."""
        if self._llm_judge is not None:
            return
        if self._llm_model_name is None and not hasattr(self, '_llm_model_name'):
            return

        from .rerankers.llm import LLMListwiseReranker
        self._llm_judge = LLMListwiseReranker(
            model_name=self._llm_model_name,
            device=self._llm_device,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    # Core linking: three-stage cascade
    # ------------------------------------------------------------------

    def link_one(
        self,
        entity_name: str,
        context: Optional[Dict[str, str]] = None,
        raw_affiliation: Optional[str] = None,
    ) -> Optional[Candidate]:
        """
        Link a single entity name to a registry record.

        Three-stage cascade:
          1. Direct match (name + country → unique record)
          2. Dense retrieval + reranking
          3. LLM judge for low-confidence results (if enabled)

        Parameters
        ----------
        entity_name : str
            Organization name from NER.
        context : dict, optional
            {"country": "Spain", "city": "Barcelona"}
        raw_affiliation : str, optional
            Full affiliation string (used as reranker query).

        Returns
        -------
        Candidate or None
        """
        if not entity_name or not entity_name.strip():
            return None

        ctx = context or {}

        # ── Stage 1: Direct Match ──
        country = ctx.get("country", "")
        dm_result = self._direct_matcher.match(entity_name, country)
        if dm_result is not None:
            return dm_result

        # ── Stage 2: Retrieval + Reranking ──
        candidates = self._retriever.retrieve(
            entity_name, top_k=self.top_k, context=ctx,
        )

        if not candidates:
            return None


        # Rerank
        if self._reranker is not None:
            # Use entity name as query, NOT the full affiliation.
            # The full affiliation contains ALL entities (e.g. both
            # "Università di Roma Tre" AND "INFN sezione di Roma Tre"),
            # which confuses the reranker into matching the wrong one.
            reranker_query = entity_name
            candidates = self._reranker.rerank(
                reranker_query, candidates, context=ctx,
            )

            # Quality gate: if reranker's best score is below its own
            # threshold, none of the candidates are good matches.
            # Reject hard — even the LLM can't salvage garbage candidates.
            # From experiments: jina_comet_large threshold = 0.038
            if candidates and candidates[0].score < self.reranker_threshold:
                if self.verbose:
                    logger.debug(
                        "'%s' → reranker top score %.4f < reranker_threshold %.4f → no match",
                        entity_name, candidates[0].score, self.reranker_threshold,
                    )
                return None



        best = candidates[0] if candidates else None

        if best is None:
            return None

        # ── Stage 3: LLM Judge (if enabled and score is low) ──
        if (
            self._llm_model_name is not None
            and best.score < self.llm_threshold
            and len(candidates) >= 2
        ):
            self._ensure_llm_judge()
            if self._llm_judge is not None:
                # Enrich candidates with alias info for the LLM
                enriched = self._enrich_candidates(candidates[:10])
                llm_ctx = dict(ctx)
                llm_ctx["entity_name"] = entity_name
                llm_query = raw_affiliation or entity_name

                llm_reranked = self._llm_judge.rerank(
                    llm_query, enriched, context=llm_ctx,
                )
                if llm_reranked:
                    best = llm_reranked[0]

        # Apply final threshold
        if best.score >= self.threshold:
            # Resolve successor if the matched record is inactive
            best = self._resolve_if_inactive(best)
            return best

        if self.verbose:
            logger.debug(
                "'%s' → '%s' score=%.3f < threshold=%.2f",
                entity_name, best.name, best.score, self.threshold,
            )
        return None

    def _resolve_if_inactive(self, candidate: Candidate) -> Candidate:
        """If candidate points to an inactive record, follow successor chain."""
        rec = self._record_map.get(candidate.id)
        if rec is None or rec.status == "active":
            return candidate

        # Use DirectMatcher's successor resolution
        resolved_id = self._direct_matcher._resolve_successor(candidate.id)
        if resolved_id != candidate.id:
            resolved_rec = self._record_map.get(resolved_id)
            if resolved_rec:
                return replace(
                    candidate,
                    id=resolved_id,
                    name=resolved_rec.name,
                    metadata={
                        "country": resolved_rec.country,
                        "country_code": resolved_rec.country_code,
                        "city": resolved_rec.city,
                        "types": resolved_rec.types,
                        "resolved_from": candidate.id,  # track original
                    },
                )
        return candidate

    def _enrich_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Add alias/acronym info to candidate metadata for the LLM."""
        enriched = []
        for c in candidates:
            rec = self._record_map.get(c.id)
            if rec:
                meta = dict(c.metadata)
                meta["aliases"] = rec.aliases[:5] + rec.acronyms
                enriched.append(replace(c, metadata=meta))
            else:
                enriched.append(c)
        return enriched

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def link(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Link entities in pipeline items to registry IDs.

        For each item, processes ORG (and optionally SUB) entities from NER.
        Populates item["entity_linking"].
        """
        results = []
        for item in items:
            out = dict(item)
            try:
                out["entity_linking"] = self._link_item(out)
            except Exception as e:
                if self.verbose:
                    text = out.get("raw_text", "")[:60]
                    logger.warning("Entity linking failed for '%s': %s", text, e)
                out["entity_linking"] = {"institutions": [], "subunits": []}
            results.append(out)
        return results

    def _link_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Link all entities in a single pipeline item."""
        ner_list = item.get("ner", [])
        if not ner_list or not isinstance(ner_list, list):
            return {"institutions": [], "subunits": []}

        ner_entities = ner_list[0] if isinstance(ner_list[0], dict) else {}

        context = self._extract_context(ner_entities)
        raw_aff = item.get("raw_text", "") or item.get("original", "")

        # Link ORG entities
        institutions = []
        for org_name in ner_entities.get("ORG", []):
            if not org_name:
                continue
            match = self.link_one(org_name, context=context, raw_affiliation=raw_aff)
            if match is not None:
                rec = self._record_map.get(match.id)
                institutions.append({
                    "query": org_name,
                    "id": {
                        "ror_id": match.id,
                        "ror_name": match.name,
                        "ror_country": rec.country if rec else "",
                        "ror_country_code": rec.country_code if rec else "",
                        "ror_city": rec.city if rec else "",
                        "ror_types": rec.types if rec else [],
                    },
                    "score": round(match.score, 4),
                    "source": match.source,
                    "retriever": match.retriever,
                })
            else:
                institutions.append({
                    "query": org_name,
                    "id": None,
                    "score": 0.0,
                    "source": None,
                    "retriever": None,
                })

        # SUB entities → subunits, NOT linked (pass through as-is)
        subunits = []
        for sub_name in ner_entities.get("SUB", []):
            if not sub_name:
                continue
            subunits.append({
                "query": sub_name,
                "id": None,
                "score": 0.0,
                "source": "ner",
                "retriever": None,
            })

        # SUBORG entities → subunits, LINKED (these are sub-organizations
        # that may have their own registry entry, e.g. "Harvard Medical School")
        for suborg_name in ner_entities.get("SUBORG", []):
            if not suborg_name:
                continue
            match = self.link_one(suborg_name, context=context, raw_affiliation=raw_aff)
            if match is not None:
                rec = self._record_map.get(match.id)
                subunits.append({
                    "query": suborg_name,
                    "id": {
                        "ror_id": match.id,
                        "ror_name": match.name,
                        "ror_country": rec.country if rec else "",
                        "ror_country_code": rec.country_code if rec else "",
                        "ror_city": rec.city if rec else "",
                        "ror_types": rec.types if rec else [],
                    },
                    "score": round(match.score, 4),
                    "source": match.source,
                    "retriever": match.retriever,
                })
            else:
                subunits.append({
                    "query": suborg_name, "id": None,
                    "score": 0.0, "source": None, "retriever": None,
                })

        return {"institutions": institutions, "subunits": subunits}

    @staticmethod
    def _extract_context(ner_entities: Dict[str, Any]) -> Dict[str, str]:
        """Extract NER context for disambiguation."""
        def _first(lst):
            if isinstance(lst, list) and lst:
                return lst[0]
            return ""
        return {
            "country": _first(ner_entities.get("COUNTRY")),
            "city": _first(ner_entities.get("CITY")),
            "region": _first(ner_entities.get("REGION")),
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        info = {
            "data_source": self.data_source,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "retriever": type(self._retriever).__name__,
            "reranker": type(self._reranker).__name__ if self._reranker else None,
            "llm_judge": self._llm_model_name or False,
            "cascade_stages": [
                "direct_match",
                "retrieval" + ("+reranking" if self._reranker else ""),
            ] + (["llm_judge"] if self._llm_model_name else []),
        }

        if isinstance(self._retriever, DenseRetriever):
            idx = self._retriever.dense_index
            info["index"] = {
                "num_vectors": idx.num_vectors,
                "num_records": idx.num_records,
                "type": idx.meta.get("index_type", "?"),
                "encoder": idx.meta.get("encoder", "?"),
            }

        return info

    def free(self):
        """Release GPU resources."""
        if self._reranker is not None:
            self._reranker.free()
        if self._llm_judge is not None:
            self._llm_judge.free()