"""
Public API for AffilGood.

This module defines the stable, user-facing interface.
Internal components (NER, linking, normalization, etc.)
must NOT be imported directly by users.

Return semantics
----------------
process(str) → dict:
    {"raw_text": str, "outputs": [{"input": ..., "institutions": [...], ...}, ...]}

process([str]) → list[dict]:
    [{"raw_text": str, "outputs": [...]}, ...]

Input type determines output type (like HuggingFace pipelines).
"""

from typing import List, Union, Dict, Any, Optional
from collections.abc import Sequence


class AffilGood:
    """
    AffilGood public API.

    Example::

        ag = AffilGood()

        # Single string → single dict
        result = ag.process("UAB, Spain; MIT, USA")

        # List of strings → list of dicts
        results = ag.process(["UAB, Spain", "MIT, USA"])

        # With NUTS regions
        ag = AffilGood(add_nuts=True)
        result = ag.process("Universitat Autònoma de Barcelona, Spain")

        # With language detection
        ag = AffilGood(
            enable_language_preprocessing=True,
            language_config={"method": "combined_langdetect"},
        )
        result = ag.process("Université de Paris, France")
    """

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        output_format: str = "normalized",
        enable_language_detect: bool = False,
        enable_normalization: bool = True,
        enable_entity_linking: bool = True,
        entity_linking_sources: Union[str, List[str]] = "ror",
        add_nuts: bool = False,
        verbose: bool = False,
        span_config: Optional[Dict[str, Any]] = None,
        ner_config: Optional[Dict[str, Any]] = None,
        linking_config: Optional[Dict[str, Any]] = None,
        normalization_config: Optional[Dict[str, Any]] = None,
        language_config: Optional[Dict[str, Any]] = None,
        translate_config: Optional[Dict[str, Any]] = None,
        orgtype_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AffilGood.

        Parameters
        ----------
        device :
            "cpu", "cuda", or None (auto-detect)
        batch_size :
            Default batch size for processing
        output_format :
            "normalized" (default) or "full"
        enable_language_detect :
            Enable language detection (and future translation).
            When True, each span gets language_info before NER.
        enable_normalization :
            Enable geocoding / country normalization (OSM Nominatim)
        enable_entity_linking :
            Enable entity linking (e.g. ROR)
        entity_linking_sources :
            Data sources for entity linking
        add_nuts :
            Enable NUTS region enrichment for European locations.
        verbose :
            Verbose logging

        language_config : dict, optional
            Configuration for language detection. Keys:
            - method: "heuristic" (default), "langdetect", "e5",
              "fasttext", "combined_langdetect", "combined_e5"
            - default_language: fallback language code (default: "en")

        normalization_config : dict, optional
            Configuration for geocoder. Keys:
            - cache_dir, cache_expiration_days, user_agent,
              data_dir, nuts_shapefile.

        translate_config : dict, optional
            Configuration for non-Latin script translation.
            When provided, enables automatic translation of affiliations
            in Chinese, Japanese, Arabic, Russian, Persian, etc.
            Keys:
            - model_name: HuggingFace model (default: "Qwen/Qwen2.5-0.5B-Instruct")
            - device: "cpu" (default) or "cuda"
            - extra_languages: additional ISO language codes to translate
            Requires enable_language_detect=True.

        orgtype_config : dict, optional
            Configuration for organization type classification.
            When provided, classifies each institution into a two-level
            taxonomy (lvl1: e.g. "Higher Education", lvl2: e.g. "HEI.institution")
            using context from DuckDuckGo search + RoBERTa classifiers.
            Keys:
            - lvl1_model: HuggingFace model (default: "SIRIS-Lab/acty2de-roberta_lvl1_ctx")
            - lvl2_model: HuggingFace model (default: "SIRIS-Lab/acty2de-roberta_lvl2_ctx")
            - device: "cpu" (default) or "cuda"
            - cache_dir: directory for SQLite cache
            - search_sleep: seconds between DDG queries (default: 1.0)
        """
        self.verbose = verbose
        self.output_format = output_format
        self.batch_size = batch_size

        if output_format not in {"normalized", "full"}:
            raise ValueError(
                "output_format must be 'normalized' or 'full'"
            )

        from affilgood.pipeline import AffiliationPipeline
        from affilgood.output import normalize_output

        self._normalize_output = normalize_output

        self._pipeline = AffiliationPipeline(
            device=device,
            batch_size=batch_size,
            enable_language_detect=enable_language_detect,
            enable_normalization=enable_normalization,
            enable_entity_linking=enable_entity_linking,
            entity_linking_sources=entity_linking_sources,
            add_nuts=add_nuts,
            span_config=span_config or {},
            ner_config=ner_config or {},
            linking_config=linking_config or {},
            normalization_config=normalization_config or {},
            language_config=language_config or {},
            translate_config=translate_config,
            orgtype_config=orgtype_config,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def process(
        self,
        inputs: Union[str, List[str]],
        *,
        return_debug: bool = False,
        batch_size: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process one or more affiliation strings.

        Parameters
        ----------
        inputs : str or List[str]
            A single affiliation string, or a list of them.
            Input type determines output type.
        return_debug : bool
            If True, return full internal pipeline state.
        batch_size : int, optional
            Override default batch size.

        Returns
        -------
        If inputs is str → dict:
            {"raw_text": str, "outputs": [{"input": ..., ...}, ...]}

        If inputs is list[str] → list[dict]:
            [{"raw_text": str, "outputs": [...]}, ...]
        """
        single = isinstance(inputs, str)

        if single:
            texts = [inputs]
        elif isinstance(inputs, Sequence) and all(isinstance(t, str) for t in inputs):
            texts = list(inputs)
        else:
            raise TypeError(
                f"inputs must be a string or list of strings, got {type(inputs).__name__}"
            )

        if not texts:
            return [] if not single else {"raw_text": "", "outputs": []}

        grouped_raw = self._pipeline.run(
            texts,
            batch_size=batch_size or self.batch_size,
        )

        results = []
        for source_text, group in zip(texts, grouped_raw):
            if self.output_format == "full" or return_debug:
                outputs = group
            else:
                outputs = [
                    self._normalize_output(item) for item in group
                ]

            results.append({
                "raw_text": source_text,
                "outputs": outputs,
            })

        if single:
            return results[0]

        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return self._pipeline.info()