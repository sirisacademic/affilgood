"""
Public API for AffilGood.

This module defines the stable, user-facing interface.
Internal components (NER, linking, normalization, etc.)
must NOT be imported directly by users.
"""

from typing import List, Union, Dict, Any, Optional


class AffilGood:
    """
    AffilGood public API.

    Example:
        ag = AffilGood()
        result = ag.process("Universitat Autònoma de Barcelona, Spain")
    """

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        output_format: str = "normalized",  # "normalized" | "full"
        enable_language_preprocessing: bool = False,
        enable_normalization: bool = True,
        entity_linking_sources: Union[str, List[str]] = "ror",
        verbose: bool = False,
        # Advanced / internal configs (kept grouped)
        span_config: Optional[Dict[str, Any]] = None,
        ner_config: Optional[Dict[str, Any]] = None,
        linking_config: Optional[Dict[str, Any]] = None,
        normalization_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AffilGood.

        Parameters
        ----------
        device:
            "cpu", "cuda", or None (auto-detect)
        batch_size:
            Default batch size for processing
        output_format:
            "normalized" (default) or "full"
        enable_language_preprocessing:
            Enable LLM-based translation / language handling
        enable_normalization:
            Enable geo / country normalization
        entity_linking_sources:
            Data sources for entity linking (e.g. "ror", ["ror", "wikidata"])
        verbose:
            Verbose logging

        The *_config arguments are advanced and mostly for power users.
        """

        self.verbose = verbose
        self.output_format = output_format
        self.batch_size = batch_size

        if output_format not in {"normalized", "full"}:
            raise ValueError(
                "output_format must be 'normalized' or 'full'"
            )

        # Lazy imports to keep API lightweight
        from affilgood.pipeline import AffiliationPipeline
        from affilgood.output import normalize_output

        self._normalize_output = normalize_output

        # Build pipeline configuration
        self._pipeline = AffiliationPipeline(
            device=device,
            batch_size=batch_size,
            enable_language_preprocessing=enable_language_preprocessing,
            enable_normalization=enable_normalization,
            entity_linking_sources=entity_linking_sources,
            span_config=span_config or {},
            ner_config=ner_config or {},
            linking_config=linking_config or {},
            normalization_config=normalization_config or {},
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def process(self, text: str, *, return_debug: bool = False):
        results = self.process_batch([text], return_debug=return_debug)

        if not results:
            return {
                "input": text,
                "institutions": [],
                "subunits": [],
                "location": None,
                "language": None,
                "confidence": None,
                "debug": {},
            }

        return results[0]

    def process_batch(
        self,
        texts: List[str],
        *,
        return_debug: bool = False,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of affiliation strings.
        """
        if not texts:
            return []

        raw_results = self._pipeline.run(
            texts,
            batch_size=batch_size or self.batch_size,
        )

        if self.output_format == "full" or return_debug:
            return raw_results

        # Default: normalized output
        return [
            self._normalize_output(result)
            for result in raw_results
        ]

    # ------------------------------------------------------------------
    # Introspection helpers (safe)
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """
        Return basic information about the pipeline configuration.
        """
        return self._pipeline.info()
