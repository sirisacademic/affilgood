"""
Internal pipeline orchestration for AffilGood.

This module wires together the different components:
- language preprocessing
- span identification
- NER
- normalization
- entity linking

This is NOT part of the public API.
"""

from typing import List, Dict, Any, Optional
import time


class AffiliationPipeline:
    """
    Orchestrates the full affiliation processing pipeline.

    This pipeline is defensive:
    - Components are optional
    - Missing dependencies do NOT crash installation
    - Stages degrade gracefully
    """

    def __init__(
        self,
        *,
        device: Optional[str],
        batch_size: int,
        enable_language_preprocessing: bool,
        enable_normalization: bool,
        entity_linking_sources,
        span_config: Dict[str, Any],
        ner_config: Dict[str, Any],
        linking_config: Dict[str, Any],
        normalization_config: Dict[str, Any],
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.batch_size = batch_size

        # -------------------------------------------------
        # Device handling
        # -------------------------------------------------
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.device = device

        # -------------------------------------------------
        # Language preprocessing (optional, safe)
        # -------------------------------------------------
        self.language_preprocessor = None
        if enable_language_preprocessing:
            try:
                from affilgood.preprocessing.llm_translator import LLMTranslator

                self.language_preprocessor = LLMTranslator(
                    verbose=verbose,
                    **(span_config or {}),
                )
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Language preprocessing disabled: {e}")

        # -------------------------------------------------
        # Span identification (safe)
        # -------------------------------------------------
        self.span_identifier = None
        try:
            from affilgood.span_identification.span_identifier import SpanIdentifier

            self.span_identifier = SpanIdentifier(
                device=device,
                batch_size=batch_size,
                **span_config,
            )
        except Exception as e:
            if self.verbose:
                print(f"[Pipeline] Span identifier unavailable: {e}")

        # -------------------------------------------------
        # Named Entity Recognition (safe)
        # -------------------------------------------------
        self.ner = None
        try:
            from affilgood.components.ner import NER

            self.ner = NER(
                device=device,
                batch_size=batch_size,
                verbose=True,
                **ner_config,
            )
        except Exception as e:
            if self.verbose:
                print(f"[Pipeline] NER unavailable: {e}")

        # -------------------------------------------------
        # Normalization (optional, safe)
        # -------------------------------------------------
        self.normalizer = None
        if enable_normalization:
            try:
                from affilgood.metadata_normalization.normalizer import GeoNormalizer

                self.normalizer = GeoNormalizer(**normalization_config)
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Normalization disabled: {e}")

        # -------------------------------------------------
        # Entity linking (safe)
        # -------------------------------------------------
        self.entity_linker = None
        try:
            from affilgood.entity_linking.entity_linker import EntityLinker

            self.entity_linker = EntityLinker(
                data_sources=entity_linking_sources,
                device=device,
                batch_size=batch_size,
                verbose=verbose,
                **linking_config,
            )
        except Exception as e:
            if self.verbose:
                print(f"[Pipeline] Entity linking unavailable: {e}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute the full pipeline on a list of texts.
        Always returns a list of dicts with a stable structure.
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size
        t0 = time.time()

        # -------------------------------------------------
        # Initialize base results (null-safe)
        # -------------------------------------------------
        results = [
            {
                "raw_text": text,
                "span_entities": [text],
                "ner": [{}],
                "entity_linking": {},
                "osm": [],
                "language_info": {},
            }
            for text in texts
        ]

        # -------------------------------------------------
        # 1. Language preprocessing
        # -------------------------------------------------
        processed_texts = texts
        preprocessing_info = [{} for _ in texts]

        if self.language_preprocessor:
            if self.verbose:
                print(f"[Pipeline] Language preprocessing ({len(texts)} texts)")
            preprocessing_info = self.language_preprocessor.process_batch(
                texts,
                batch_size=batch_size,
            )
            processed_texts = [
                item.get("processed_text", t)
                for item, t in zip(preprocessing_info, texts)
            ]

        # -------------------------------------------------
        # 2. Span identification
        # -------------------------------------------------
        if self.span_identifier:
            if self.verbose:
                print("[Pipeline] Span identification")

            spans = self.span_identifier.identify_spans(
                processed_texts,
                batch_size=batch_size,
            )

            for r, s in zip(results, spans):
                detected_spans = s.get("span_entities")

                # Defensive: fallback to full text
                if isinstance(detected_spans, list) and detected_spans:
                    r["span_entities"] = detected_spans
                else:
                    r["span_entities"] = [r["raw_text"]]

        # -------------------------------------------------
        # 3. Named Entity Recognition
        # -------------------------------------------------
        if self.ner:
            if self.verbose:
                print("[Pipeline] Named Entity Recognition")
            entities = self.ner.recognize_entities(
                results,
                batch_size=batch_size,
            )
            results = entities
        print(results)
        # -------------------------------------------------
        # 4. Normalization
        # -------------------------------------------------
        if self.normalizer:
            if self.verbose:
                print("[Pipeline] Normalization")
            results = self.normalizer.normalize(results)

        # -------------------------------------------------
        # 5. Entity linking
        # -------------------------------------------------
        if self.entity_linker:
            if self.verbose:
                print("[Pipeline] Entity linking")
            linked_results = self.entity_linker.process_in_chunks(results)

            # merge entity_linking back into original items
            for base, linked in zip(results, linked_results):
                base["entity_linking"] = linked.get("entity_linking", {})

        # -------------------------------------------------
        # 6. Attach language metadata
        # -------------------------------------------------
        for i, result in enumerate(results):
            result["language_info"] = preprocessing_info[i]

        if self.verbose:
            elapsed = time.time() - t0
            print(f"[Pipeline] Completed in {elapsed:.2f}s")

        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """
        Return information about the pipeline configuration.
        """
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "language_preprocessing": self.language_preprocessor is not None,
            "normalization": self.normalizer is not None,
            "entity_linking": self.entity_linker is not None,
        }
