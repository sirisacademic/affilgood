"""
LLM listwise reranker for entity linking.

Presents ALL candidates to a small instruction-following LLM and lets
it choose the best match — enabling cross-candidate comparison that
pointwise cross-encoders cannot do.

Addresses three blind spots of pointwise rerankers (from error analysis):
  1. Same-name disambiguation  (two "University of Georgia" with different cities)
  2. Parent/child confusion    ("LSU System" vs "LSU")
  3. Acronym resolution        ("UCL" → "University College London" vs "University and College Union")

Scoring via first-token logit probabilities over candidate letters (A–J).
One forward pass per entity — no text generation needed.

Recommended models:
  - "Qwen/Qwen2.5-3B-Instruct"   (~6 GB, good balance)
  - "Qwen/Qwen2.5-1.5B-Instruct" (~3 GB, faster)
  - "Qwen/Qwen2.5-0.5B-Instruct" (~1 GB, minimal)
"""

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseReranker
from ..retrievers.base import Candidate

logger = logging.getLogger(__name__)

LETTERS = "ABCDEFGHIJKLMNOPQRST"

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert at linking research institution mentions in "
    "scientific publication affiliations to entries in the ROR "
    "(Research Organization Registry) database.\n"
    "You will be given an affiliation string, an entity mention to "
    "link, and a list of candidate ROR entries with their names, "
    "alternative names, city, and country.\n"
    "Answer with ONLY a single letter corresponding to the best "
    "matching entry, or N if none of the candidates match."
)


class LLMListwiseReranker(BaseReranker):
    """
    Listwise LLM reranker for entity linking.

    Uses first-token logit scoring (FIRST-style) — one forward pass
    per entity, no autoregressive generation.

    Parameters
    ----------
    model_name : str
        HuggingFace instruction-following model.
    device : str or None
        "cuda", "cpu", or None (auto).
    max_tokens : int
        Maximum input length.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_tokens: int = 2048,
        verbose: bool = False,
    ):
        self.model_name = model_name or DEFAULT_LLM_MODEL
        self.max_tokens = max_tokens
        self.verbose = verbose
        self._model = None
        self._tokenizer = None
        self._device = device
        self._letter_token_ids = {}

    def _ensure_model(self):
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = self._device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left",
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self._model.to(device)
        self._model.eval()
        self._device = str(next(self._model.parameters()).device)

        # Pre-compute token IDs for candidate letters + "N" (none)
        for ch in LETTERS + "N":
            ids = self._tokenizer.encode(ch, add_special_tokens=False)
            self._letter_token_ids[ch] = ids[-1]

        if self.verbose:
            logger.info("LLM reranker loaded: %s on %s", self.model_name, self._device)

    def _build_prompt(
        self,
        raw_affiliation: str,
        entity_name: str,
        candidates: List[Candidate],
    ) -> str:
        """Build the user prompt showing all candidates."""
        n = len(candidates)
        lines = []
        for i, c in enumerate(candidates):
            parts = [c.name]
            aliases = c.metadata.get("aliases", [])
            if aliases:
                parts.append(f"a.k.a. {', '.join(aliases[:3])}")
            if c.metadata.get("city"):
                parts.append(c.metadata["city"])
            if c.metadata.get("country"):
                parts.append(c.metadata["country"])
            lines.append(f"{LETTERS[i]}) {' — '.join(parts)}")

        letter_range = f"A–{LETTERS[n-1]}" if n > 1 else "A"

        return (
            f'Affiliation string:\n"{raw_affiliation}"\n\n'
            f'Which entry best matches the entity "{entity_name}"?\n\n'
            + "\n".join(lines) + "\n\n"
            f"Answer: one letter ({letter_range}) or N if none match."
        )

    def _score_candidates(
        self,
        prompt_text: str,
        n_candidates: int,
    ) -> Tuple[List[float], float]:
        """Score via first-token logits over candidate letters."""
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_tokens,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1, :]

        letter_logits = []
        for i in range(n_candidates):
            tid = self._letter_token_ids[LETTERS[i]]
            letter_logits.append(logits[tid].item())
        none_logit = logits[self._letter_token_ids["N"]].item()

        all_logits = torch.tensor(letter_logits + [none_logit])
        probs = torch.softmax(all_logits, dim=0).tolist()

        return probs[:n_candidates], probs[n_candidates]

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Candidate]:
        if not candidates or not query:
            return candidates

        self._ensure_model()

        # Limit to 20 (letter limit)
        candidates = candidates[:min(len(candidates), len(LETTERS))]

        # Use entity name from context if available, else first word of query
        entity_name = (context or {}).get("entity_name", query)

        prompt = self._build_prompt(query, entity_name, candidates)
        cand_probs, none_prob = self._score_candidates(prompt, len(candidates))

        if self.verbose:
            logger.debug(
                "LLM judge: none_prob=%.3f, best_cand_prob=%.3f",
                none_prob, max(cand_probs) if cand_probs else 0,
            )

        # If LLM thinks "none match" more than any candidate, return empty
        best_cand_prob = max(cand_probs) if cand_probs else 0
        if none_prob > best_cand_prob:
            if self.verbose:
                logger.debug("LLM judge: N (%.3f) > best candidate (%.3f) → no match", none_prob, best_cand_prob)
            return []

        reranked = []
        for cand, prob in zip(candidates, cand_probs):
            reranked.append(replace(cand, score=prob, retriever="llm_listwise"))

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked

    def free(self):
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass