"""
Internal pipeline orchestration for AffilGood.

Architecture: fan-out / fan-in
------------------------------
    Input texts (N)
        │
    Span identification ──→ Expanded items (M ≥ N), one per affiliation span
        │
    Flatten for batch processing
        │
    Language detection (optional)
        │
    [future: translation of non-English affiliations]
        │
    NER (batch on all spans)
        │
    Entity Linking (optional — match ORGs to ROR, etc.)
        │
    Geocoding (optional — OSM Nominatim + NUTS)
        │
    Regroup by source input ──→ List[List[item]]
"""

from typing import List, Dict, Any, Optional
import time


class AffiliationPipeline:
    """
    Orchestrates the affiliation processing pipeline.
    """

    def __init__(
        self,
        *,
        device: Optional[str],
        batch_size: int,
        enable_language_detect: bool,
        enable_normalization: bool,
        enable_entity_linking: bool,
        entity_linking_sources,
        add_nuts: bool = False,
        span_config: Dict[str, Any],
        ner_config: Dict[str, Any],
        linking_config: Dict[str, Any],
        normalization_config: Dict[str, Any],
        language_config: Optional[Dict[str, Any]] = None,
        translate_config: Optional[Dict[str, Any]] = None,
        orgtype_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.batch_size = batch_size

        # Device handling
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.device = device

        # -------------------------------------------------
        # Components
        # -------------------------------------------------
        from affilgood.components.span import SpanIdentifier
        from affilgood.components.ner import NER

        self.span_identifier = SpanIdentifier(
            verbose=verbose,
            **(span_config or {}),
        )

        self.ner = NER(
            device=device,
            batch_size=batch_size,
            verbose=verbose,
            **(ner_config or {}),
        )

        # Language detection (optional)
        self.enable_language_detect = enable_language_detect
        self.language_detector = None

        if enable_language_detect:
            try:
                from affilgood.components.language import LanguageDetector
                self.language_detector = LanguageDetector(
                    device=device,
                    verbose=verbose,
                    **(language_config or {}),
                )
                if verbose:
                    print(f"[Pipeline] Language detection: method={self.language_detector.method}")
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] Failed to initialize LanguageDetector: {e}")

        # Translation (optional — for non-Latin script affiliations)
        self.translator = None

        if translate_config is not None:
            try:
                from affilgood.components.translate import AffiliationTranslator
                self.translator = AffiliationTranslator(
                    verbose=verbose,
                    **translate_config,
                )
                if verbose:
                    print(f"[Pipeline] Translation: model={self.translator.model_name}")
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] Failed to initialize Translator: {e}")

        # Entity linking (optional)
        self.enable_entity_linking = enable_entity_linking
        self.entity_linking_sources = entity_linking_sources
        self.entity_linker = None

        if enable_entity_linking:
            try:
                from affilgood.components.entity_linking import EntityLinker
                el_config = dict(linking_config or {})
                el_config.setdefault("device", device)
                el_config.setdefault("verbose", verbose)
                self.entity_linker = EntityLinker(**el_config)
                if verbose:
                    info = self.entity_linker.info()
                    print(f"[Pipeline] Entity linking: {info.get('retriever', '?')} retriever")
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] Failed to initialize EntityLinker: {e}")
                self.entity_linker = None

        # Geocoder (optional)
        self.enable_normalization = enable_normalization
        self.geocoder = None

        if enable_normalization:
            try:
                from affilgood.components.geocode import Geocoder
                self.geocoder = Geocoder(
                    verbose=verbose,
                    add_nuts=add_nuts,
                    **(normalization_config or {}),
                )
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] Failed to initialize Geocoder: {e}")

        # Organization type classification (optional)
        self.org_type_classifier = None

        if orgtype_config is not None:
            try:
                from affilgood.components.organization_type import OrganizationTypeClassifier
                ot_config = dict(orgtype_config)
                ot_config.setdefault("device", device or "cpu")
                ot_config.setdefault("verbose", verbose)
                self.org_type_classifier = OrganizationTypeClassifier(**ot_config)
                if verbose:
                    print(f"[Pipeline] Org type classification: enabled")
            except Exception as e:
                if verbose:
                    print(f"[Pipeline] Failed to initialize OrgTypeClassifier: {e}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        texts: List[str],
        *,
        batch_size: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute the pipeline on a list of texts.

        Returns List[List[Dict]] — grouped by input, one inner item per span.
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size
        t0 = time.time()

        # 0. Span identification (fan-out)
        if self.verbose:
            print("[Pipeline] Span identification (fan-out)")

        grouped_items = self._expand_spans(texts)

        if self.verbose:
            n_spans = sum(len(group) for group in grouped_items)
            print(f"[Pipeline]   {len(texts)} inputs → {n_spans} affiliation spans")

        flat_items, group_sizes = self._flatten(grouped_items)

        # 1. Language detection (optional)
        if self.language_detector is not None:
            if self.verbose:
                print("[Pipeline] Language detection")

            flat_items = self.language_detector.detect(flat_items)

            if self.verbose:
                langs = [item.get("language_info", {}).get("language", "?") for item in flat_items]
                from collections import Counter
                dist = dict(Counter(langs).most_common(5))
                print(f"[Pipeline]   Languages detected: {dist}")

        # 2. Translation of non-Latin script affiliations (optional)
        if self.translator is not None:
            if self.verbose:
                print("[Pipeline] Translation (non-Latin scripts)")

            flat_items = self.translator.translate(flat_items)

        # 3. NER
        if self.verbose:
            print("[Pipeline] Named Entity Recognition")

        flat_items = self.ner.recognize_entities(
            flat_items, batch_size=batch_size,
        )

        # 4. Entity Linking (optional)
        if self.entity_linker is not None:
            if self.verbose:
                print("[Pipeline] Entity Linking")
            flat_items = self.entity_linker.link(flat_items)

            if self.verbose:
                # Count successful links
                linked = sum(
                    1
                    for item in flat_items
                    for inst in item.get("entity_linking", {}).get("institutions", [])
                    if inst.get("id") is not None
                )
                total = sum(
                    len(item.get("entity_linking", {}).get("institutions", []))
                    for item in flat_items
                )
                print(f"[Pipeline]   Linked {linked}/{total} institutions")

        # 5. Geocoding (optional)
        if self.geocoder is not None:
            if self.verbose:
                print("[Pipeline] Geocoding")
            flat_items = self.geocoder.normalize(flat_items)

            # 5b. Feedback: fill missing locations from entity linking
            #     When NER missed CITY/COUNTRY but EL found a match,
            #     use ROR city+country to geocode.
            if self.entity_linker is not None:
                n_filled = 0
                for item in flat_items:
                    osm = item.get("osm", [])
                    if osm:
                        continue  # already geocoded

                    # Get city+country from best matched institution
                    el = item.get("entity_linking", {})
                    institutions = el.get("institutions", [])
                    ror_city = None
                    ror_country = None
                    for inst in institutions:
                        id_field = inst.get("id")
                        if isinstance(id_field, dict):
                            ror_city = ror_city or id_field.get("ror_city")
                            ror_country = ror_country or id_field.get("ror_country")

                    # Also check subunits
                    if not ror_city:
                        for sub in el.get("subunits", []):
                            id_field = sub.get("id")
                            if isinstance(id_field, dict):
                                ror_city = ror_city or id_field.get("ror_city")
                                ror_country = ror_country or id_field.get("ror_country")

                    if ror_city or ror_country:
                        # Build synthetic NER for geocoder
                        synthetic_ner = {}
                        if ror_city:
                            synthetic_ner["CITY"] = [ror_city]
                        if ror_country:
                            synthetic_ner["COUNTRY"] = [ror_country]

                        # Temporarily inject synthetic NER
                        original_ner = item.get("ner", [])
                        item["ner"] = [synthetic_ner]

                        try:
                            geocoded = self.geocoder.normalize([item])
                            osm_result = geocoded[0].get("osm", [])
                            if osm_result:
                                # Tag sources as "ror-osm"
                                entry = osm_result[0]
                                entry["_source_type"] = "ror-osm"
                                item["osm"] = osm_result
                                n_filled += 1

                                if self.verbose:
                                    print(f"[Pipeline]   ROR→geocode: "
                                          f"{ror_city}, {ror_country}")
                        except Exception:
                            pass

                        # Restore original NER
                        item["ner"] = original_ner

                if self.verbose and n_filled:
                    print(f"[Pipeline]   Filled {n_filled} locations from ROR data")

        # 6. Organization type classification (optional)
        if self.org_type_classifier is not None:
            if self.verbose:
                print("[Pipeline] Organization type classification")
            flat_items = self.org_type_classifier.classify(flat_items)

        # Regroup
        grouped = self._regroup(flat_items, group_sizes)

        if self.verbose:
            elapsed = time.time() - t0
            print(f"[Pipeline] Completed in {elapsed:.2f}s")

        return grouped

    # ------------------------------------------------------------------
    # Span expansion
    # ------------------------------------------------------------------

    def _expand_spans(
        self, texts: List[str]
    ) -> List[List[Dict[str, Any]]]:
        result: List[List[Dict[str, Any]]] = []

        for source_index, text in enumerate(texts):
            temp_item = {
                "raw_text": text,
                "span_entities": [],
                "ner": [], "ner_raw": [],
                "entity_linking": {},
                "osm": [], "language_info": {},
            }

            try:
                processed = self.span_identifier.identify_spans([temp_item])
                spans = processed[0].get("span_entities", [])
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Span failed for input {source_index}: {e}")
                spans = []

            if not spans:
                spans = [text]

            group: List[Dict[str, Any]] = []
            for span_index, span_text in enumerate(spans):
                group.append(_make_item(
                    raw_text=span_text,
                    source_text=text,
                    source_index=source_index,
                    span_index=span_index,
                ))
            result.append(group)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(grouped):
        flat, sizes = [], []
        for group in grouped:
            sizes.append(len(group))
            flat.extend(group)
        return flat, sizes

    @staticmethod
    def _regroup(flat, sizes):
        result, offset = [], 0
        for size in sizes:
            result.append(flat[offset : offset + size])
            offset += size
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        info = {
            "device": self.device,
            "batch_size": self.batch_size,
            "components": {
                "span": True,
                "language_detection": self.language_detector is not None,
                "ner": True,
                "entity_linking": self.entity_linker is not None,
                "normalization": self.geocoder is not None,
                "nuts": self.geocoder is not None and self.geocoder._nuts is not None,
            },
        }
        if self.language_detector is not None:
            info["language_method"] = self.language_detector.method
        if self.entity_linker is not None:
            info["entity_linking"] = self.entity_linker.info()
        if self.geocoder is not None:
            info["geocoder_cache"] = self.geocoder.cache_stats()
        return info


def _make_item(*, raw_text, source_text, source_index, span_index):
    return {
        "raw_text": raw_text,
        "source_text": source_text,
        "source_index": source_index,
        "span_index": span_index,
        "span_entities": [raw_text],
        "ner": [], "ner_raw": [],
        "entity_linking": {},
        "osm": [], "language_info": {},
    }