"""
Lightweight span identification component.

Responsibilities:
- Identify meaningful spans in affiliation strings
- Optionally use a ML model
- Always return a stable structure

This component is defensive and batch-oriented.
"""

from typing import List, Dict, Any, Optional

DEFAULT_SPAN_MODEL = "nicolauduran45/affilgood-span-multilingual-v2"


class SpanIdentifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        verbose: bool = False,
        min_score: float = 0.0,
        fix_words: bool = True,
        merge_spans: bool = True,
    ):
        self.model_path = model_path or DEFAULT_SPAN_MODEL
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        self.min_score = min_score
        self.fix_words_enabled = fix_words
        self.merge_spans_enabled = merge_spans

        self._pipeline = None
        self._available = False

        self._load_model()

    # -------------------------------------------------
    # Model loading (safe)
    # -------------------------------------------------

    def _load_model(self):
        try:
            from transformers import pipeline

            if self.verbose:
                print(f"[Span] Loading model: {self.model_path}")

            self._pipeline = pipeline(
                "token-classification",
                model=self.model_path,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
            )
            self._available = True

        except Exception as e:
            if self.verbose:
                print(f"[Span] Model unavailable, using noop span identifier: {e}")
            self._pipeline = None
            self._available = False

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def identify_spans(
        self,
        items: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify spans for each input item.

        Always returns:
        - span_entities: list[str]
        """
        batch_size = batch_size or self.batch_size

        results = []
        texts = []

        for item in items:
            out = dict(item)
            out["span_entities"] = []
            results.append(out)
            texts.append(item.get("raw_text", ""))

        # No model → fallback to full text
        if not self._available:
            for out in results:
                text = out.get("raw_text", "")
                out["span_entities"] = [text] if text else []
            return results

        # -------------------------------------------------
        # Batched inference
        # -------------------------------------------------
        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset

            dataset = Dataset.from_dict({"text": texts})

            outputs = list(
                self._pipeline(
                    KeyDataset(dataset, "text"),
                    batch_size=batch_size,
                )
            )

        except Exception as e:
            if self.verbose:
                print(f"[Span] Inference failed, using fallback: {e}")
            for out in results:
                text = out.get("raw_text", "")
                out["span_entities"] = [text] if text else []
            return results

        # -------------------------------------------------
        # Post-process
        # -------------------------------------------------
        for out, raw_text, entities in zip(results, texts, outputs):
            spans = entities

            if self.fix_words_enabled:
                spans = self._fix_words(raw_text, spans)

            if self.merge_spans_enabled:
                spans = self._clean_and_merge_spans(
                    spans,
                    min_score=self.min_score,
                )

            span_entities = [
                ent.get("word", "")
                for ent in spans
                if ent.get("word")
            ]

            # Defensive fallback
            if not span_entities and raw_text:
                span_entities = [raw_text]

            out["span_entities"] = span_entities

        return results

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _fix_words(self, raw_text: str, entities: List[Dict[str, Any]]):
        for entity in entities:
            try:
                start, end = entity.get("start"), entity.get("end")
                if start is None or end is None:
                    continue
                entity["word"] = raw_text[start:end]
            except Exception:
                continue
        return entities

    def _clean_and_merge_spans(self, entities, min_score=0.0):
        entities = [e for e in entities if e.get("score", 0) >= min_score]

        merged = []
        i = 0

        while i < len(entities):
            current = entities[i]

            if i + 1 < len(entities):
                nxt = entities[i + 1]
                try:
                    if (
                        current.get("end") == nxt.get("start")
                        and nxt.get("word")
                        and nxt["word"][0].islower()
                    ):
                        merged.append({
                            "entity_group": current.get("entity_group"),
                            "score": min(current.get("score", 0), nxt.get("score", 0)),
                            "word": current.get("word", "") + nxt.get("word", ""),
                            "start": current.get("start"),
                            "end": nxt.get("end"),
                        })
                        i += 2
                        continue
                except Exception:
                    pass

            merged.append(current)
            i += 1

        return merged