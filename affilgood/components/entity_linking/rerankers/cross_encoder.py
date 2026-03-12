"""
Cross-encoder reranker for entity linking.

Scores (affiliation, candidate_description) pairs jointly.
Default model: cometadata/jina-reranker-v2-multilingual-affiliations-v5
(fine-tuned Jina v2 on affiliation→ROR pairs — best in experiments).
"""

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

from .base import BaseReranker
from ..retrievers.base import Candidate

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cometadata/jina-reranker-v2-multilingual-affiliations-v5"


# ------------------------------------------------------------------
# Compatibility patch for Jina reranker + transformers ≥4.45
# ------------------------------------------------------------------

def _patch_xlm_roberta_position_ids():
    """
    Fix compatibility between cometadata/jina-reranker-v2-* and
    jinaai/jina-reranker-v3 custom code and transformers ≥4.45.

    These models' custom code imports create_position_ids_from_input_ids
    which was removed in newer transformers versions.
    """
    try:
        from transformers.models.xlm_roberta import modeling_xlm_roberta
        import torch

        # Fix 1: missing create_position_ids_from_input_ids
        if not hasattr(modeling_xlm_roberta, "create_position_ids_from_input_ids"):
            def create_position_ids_from_input_ids(
                input_ids, padding_idx, past_key_values_length=0,
            ):
                mask = input_ids.ne(padding_idx).int()
                incremental_indices = (
                    torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
                ) * mask
                return incremental_indices.long() + padding_idx

            modeling_xlm_roberta.create_position_ids_from_input_ids = (
                create_position_ids_from_input_ids
            )

        # Fix 2 & 3: patch _finalize_model_loading
        import transformers.modeling_utils as mu
        if getattr(mu, "_patched_finalize_done", False):
            return
        mu._patched_finalize_done = True

        orig = mu.PreTrainedModel._finalize_model_loading
        _orig_finalize = orig.__func__ if hasattr(orig, "__func__") else orig

        def _patched_finalize(*args, **kwargs):
            model = args[1] if len(args) > 1 else args[0]
            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}
            _orig_mark = model.mark_tied_weights_as_initialized

            def _safe_mark(*a, **kw):
                try:
                    return _orig_mark(*a, **kw)
                except AttributeError:
                    pass

            model.mark_tied_weights_as_initialized = _safe_mark
            return orig(*args[1:], **kwargs)

        mu.PreTrainedModel._finalize_model_loading = classmethod(_patched_finalize)

    except Exception as e:
        logger.debug("XLM-RoBERTa patch skipped: %s", e)


class CrossEncoderReranker(BaseReranker):
    """
    Pointwise cross-encoder reranker.

    Scores each (query, candidate_text) pair independently using a
    cross-encoder model, then re-sorts candidates by score.

    Parameters
    ----------
    model_name : str
        HuggingFace cross-encoder model.
    device : str or None
        "cuda", "cpu", or None (auto).
    batch_size : int
        Batch size for scoring.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        self.model_name = model_name or DEFAULT_CROSS_ENCODER
        self.batch_size = batch_size
        self.verbose = verbose
        self._model = None
        self._device = device

    def _ensure_model(self):
        """Lazy-load the cross-encoder."""
        if self._model is not None:
            return

        # Apply compatibility patch for Jina reranker models
        _patch_xlm_roberta_position_ids()

        from sentence_transformers import CrossEncoder

        device = self._device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self._model = CrossEncoder(
            self.model_name, device=device, trust_remote_code=True,
        )
        self._device = device

        if self.verbose:
            logger.info("CrossEncoder loaded: %s on %s", self.model_name, device)

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Candidate]:
        if not candidates or not query:
            return candidates

        self._ensure_model()

        # Build (query, doc) pairs — doc = all name variants + city + country
        pairs = []
        for c in candidates:
            doc_parts = [c.name]
            aliases = c.metadata.get("aliases", [])
            if aliases:
                doc_parts.extend(aliases[:3])  # cap alias count
            if c.metadata.get("city"):
                doc_parts.append(c.metadata["city"])
            if c.metadata.get("country"):
                doc_parts.append(c.metadata["country"])
            doc_text = " | ".join(doc_parts)
            pairs.append((query, doc_text))

        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
        )

        reranked = []
        for cand, score in zip(candidates, scores):
            reranked.append(replace(cand, score=float(score), retriever="cross_encoder"))

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked

    def free(self):
        if self._model is not None:
            try:
                self._model.model.cpu()
            except Exception:
                pass
            del self._model
            self._model = None

            try:
                import torch, gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass