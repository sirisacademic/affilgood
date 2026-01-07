"""
Lightweight Named Entity Recognition (NER) component.

Responsibilities:
- Run a NER model (if available)
- Apply optional post-processing
- Return a stable structure

This component is defensive and batch-oriented.
"""

from typing import List, Dict, Any, Optional

DEFAULT_NER_MODEL = "nicolauduran45/affilgood-ner-multilingual-v2"


class NER:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        verbose: bool = False,
        fix_words: bool = True,
        merge_entities: bool = True,
        min_score: float = 0.0,
    ):
        self.model_path = model_path or DEFAULT_NER_MODEL
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        self.fix_words_enabled = fix_words
        self.merge_entities_enabled = merge_entities
        self.min_score = min_score

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
                print(f"[NER] Loading model: {self.model_path}")

            self._pipeline = pipeline(
                "token-classification",
                model=self.model_path,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
            )
            self._available = True

        except Exception as e:
            if self.verbose:
                print(f"[NER] Model unavailable, using noop NER: {e}")
            self._pipeline = None
            self._available = False

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def recognize_entities(
        self,
        items: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run NER on span_entities.

        Always returns:
        - ner: list[dict] (per span)
        - ner_raw: list[list] (per span)
        """
        batch_size = batch_size or self.batch_size

        # Flatten spans
        flat_spans = []
        span_map = []  # (item_idx, span_idx)

        for item_idx, item in enumerate(items):
            spans = item.get("span_entities", [])
            for span_idx, span in enumerate(spans):
                flat_spans.append(span)
                span_map.append((item_idx, span_idx))

        # Prepare empty outputs
        results = []
        for item in items:
            out = dict(item)
            out["ner"] = [{} for _ in item.get("span_entities", [])]
            out["ner_raw"] = [[] for _ in item.get("span_entities", [])]
            results.append(out)

        # No spans or no model → noop
        if not flat_spans or not self._available:
            return results

        # -------------------------------------------------
        # Batched inference (KeyDataset)
        # -------------------------------------------------
        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset

            dataset = Dataset.from_dict({"text": flat_spans})

            outputs = list(
                self._pipeline(
                    KeyDataset(dataset, "text"),
                    batch_size=batch_size,
                )
            )

        except Exception as e:
            if self.verbose:
                print(f"[NER] Inference failed: {e}")
            return results

        # -------------------------------------------------
        # Post-process and map back
        # -------------------------------------------------
        for output, (item_idx, span_idx), span_text in zip(
            outputs, span_map, flat_spans
        ):
            raw = output

            if self.fix_words_enabled:
                raw = self._fix_words(span_text, raw)

            if self.merge_entities_enabled:
                raw = self._clean_and_merge_entities(
                    raw, min_score=self.min_score
                )

            structured = self._group_entities(raw)

            results[item_idx]["ner"][span_idx] = structured
            results[item_idx]["ner_raw"][span_idx] = raw

        return results

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _group_entities(self, raw_entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for ent in raw_entities:
            label = ent.get("entity_group")
            text = ent.get("word")

            if not label or not text:
                continue

            grouped.setdefault(label, []).append(text)

        return grouped

    def _fix_words(self, raw_text: str, entities: List[Dict[str, Any]]):
        for entity in entities:
            try:
                start, end = entity.get("start"), entity.get("end")
                if start is None or end is None:
                    continue

                entity_text = raw_text[start:end]

                last_open = entity_text.rfind("(")
                last_close = entity_text.rfind(")")

                if last_open > -1 and (last_close == -1 or last_open > last_close):
                    next_close = raw_text.find(")", end)
                    if next_close > -1:
                        between = raw_text[end:next_close]
                        if not any(d in between for d in [" ", ",", ";", ":", ".", "\n", "\t"]):
                            entity["end"] = next_close + 1
                            entity_text = raw_text[start : next_close + 1]

                entity["word"] = entity_text
            except Exception:
                continue

        return entities

    def _clean_and_merge_entities(self, entities, min_score=0.0):
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
                        and (nxt["word"][0].islower() or nxt["word"][0].isdigit())
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
