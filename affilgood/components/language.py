"""
Language detection component for AffilGood.

Detects the language of affiliation strings so downstream components
can decide whether translation is needed before NER/linking.

Pipeline position: after span identification, before NER.

Detection strategy (cascading):
1. Heuristic detection (fast, no model dependencies, handles
   non-Latin scripts and affiliations with strong character/keyword
   signals via heuristic_detector.py)
2. Model-based fallback when heuristic returns 'und' (undetermined):
   - "langdetect"  — lightweight, no GPU, good for Latin scripts
   - "e5"          — transformer, most accurate, needs GPU ideally
   - "fasttext"    — fast, broad coverage

Combined modes (heuristic + model, with conflict resolution):
   - "combined_langdetect" — heuristic first, langdetect tiebreaker
                             with special handling for ca, nl, East Asian
   - "combined_e5"         — heuristic first, E5 fallback for 'und'

Default mode: "heuristic" (no extra dependencies).

File layout
-----------
affilgood/components/
├── heuristic_detector.py   ← low-level scoring (keep as-is)
├── language.py             ← THIS FILE (clean class wrapping all logic)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported detection methods
_METHODS = {
    "heuristic",
    "langdetect",
    "e5",
    "fasttext",
    "combined_langdetect",
    "combined_e5",
}


class LanguageDetector:
    """
    Language detection component for the AffilGood pipeline.

    Parameters
    ----------
    method : str
        Detection method. One of:
        - "heuristic" (default): fast, no extra dependencies
        - "langdetect": heuristic + langdetect fallback for 'und'
        - "e5": heuristic + E5 transformer fallback for 'und'
        - "fasttext": heuristic + fasttext fallback for 'und'
        - "combined_langdetect": heuristic + langdetect with smart
          conflict resolution (Catalan, Dutch, East Asian handling)
        - "combined_e5": heuristic first, E5 only when 'und'
    default_language : str
        Language code when detection fails entirely (default: "en").
    device : str or None
        Device for model-based backends ("cpu", "cuda", None=auto).
    verbose : bool
        Enable verbose logging.
    """

    def __init__(
        self,
        *,
        method: str = "heuristic",
        default_language: str = "en",
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        if method not in _METHODS:
            raise ValueError(
                f"Unknown language detection method '{method}'. "
                f"Choose from: {sorted(_METHODS)}"
            )

        self.method = method
        self.default_language = default_language
        self.device = device
        self.verbose = verbose

        # Model state (instance-level, no globals)
        self._model = None
        self._tokenizer = None
        self._model_probs = None  # langdetect's detect_langs

        # Resolve the model backend needed
        self._backend = self._resolve_backend(method)

        # Load model if needed
        if self._backend is not None:
            self._load_backend(self._backend)

    @staticmethod
    def _resolve_backend(method: str) -> Optional[str]:
        """Map method name to the underlying model backend to load."""
        mapping = {
            "heuristic": None,
            "langdetect": "langdetect",
            "e5": "e5",
            "fasttext": "fasttext",
            "combined_langdetect": "langdetect",
            "combined_e5": "e5",
        }
        return mapping.get(method)

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_backend(self, backend: str):
        """Load the model backend (called once at init)."""
        try:
            if backend == "langdetect":
                from langdetect import detect, detect_langs, DetectorFactory
                DetectorFactory.seed = 2  # deterministic results
                self._model = detect
                self._model_probs = detect_langs
                if self.verbose:
                    print("[Language] langdetect backend loaded")

            elif backend == "e5":
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSequenceClassification,
                )
                model_name = "Mike0307/multilingual-e5-language-detection"
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=45,
                )
                if self.verbose:
                    print("[Language] E5 language detection model loaded")

            elif backend == "fasttext":
                import fasttext
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="facebook/fasttext-language-identification",
                    filename="model.bin",
                )
                self._model = fasttext.load_model(model_path)
                if self.verbose:
                    print(f"[Language] FastText model loaded from {model_path}")

        except Exception as e:
            logger.warning("Failed to load %s backend: %s", backend, e)
            if self.verbose:
                print(
                    f"[Language] Failed to load {backend}: {e}. "
                    f"Falling back to heuristic only."
                )
            self._model = None
            self._tokenizer = None
            self._model_probs = None

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def detect(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect language for each pipeline item.

        Populates item["language_info"] with::

            {
                "language": "es",
                "method": "heuristic",
                "confidence": None,
            }

        Parameters
        ----------
        items : list of dict
            Flat list of pipeline items (one per affiliation span).

        Returns
        -------
        list of dict
            Same items with "language_info" populated.
        """
        results = []

        for item in items:
            out = dict(item)
            text = out.get("raw_text", "")

            try:
                lang, method_used, confidence = self._detect_one(text)
            except Exception as e:
                if self.verbose:
                    print(f"[Language] Detection failed for '{text[:60]}': {e}")
                lang = self.default_language
                method_used = "default"
                confidence = None

            out["language_info"] = {
                "language": lang,
                "method": method_used,
                "confidence": confidence,
            }

            results.append(out)

        return results

    # ------------------------------------------------------------------
    # Single-text detection (dispatches by method)
    # ------------------------------------------------------------------

    def _detect_one(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """
        Detect language for a single text.

        Returns (language_code, method_used, confidence_or_None).
        """
        if not text or not text.strip():
            return self.default_language, "default", None

        # --- Method dispatch ---

        if self.method == "heuristic":
            return self._detect_heuristic_only(text)

        if self.method == "combined_langdetect":
            return self._detect_combined_langdetect(text)

        if self.method == "combined_e5":
            return self._detect_combined_e5(text)

        # Simple fallback methods: heuristic first, model if 'und'
        return self._detect_heuristic_then_model(text)

    # ------------------------------------------------------------------
    # Strategy: heuristic only
    # ------------------------------------------------------------------

    def _detect_heuristic_only(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """Pure heuristic detection."""
        from .heuristic_detector import get_language_heuristics

        lang = get_language_heuristics(text, default_lang=self.default_language)
        return lang, "heuristic", None

    # ------------------------------------------------------------------
    # Strategy: heuristic → model fallback (for 'und')
    # ------------------------------------------------------------------

    def _detect_heuristic_then_model(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """
        Two-step: heuristic first, model fallback when heuristic
        returns 'und' (undetermined).
        """
        from .heuristic_detector import get_language_heuristics

        heur_lang = get_language_heuristics(text, default_lang="und")

        if heur_lang != "und":
            return heur_lang, "heuristic", None

        # Model fallback
        if self._model is None:
            return self.default_language, "default", None

        if self._backend == "langdetect":
            return self._call_langdetect(text)
        elif self._backend == "e5":
            return self._call_e5(text)
        elif self._backend == "fasttext":
            return self._call_fasttext(text)

        return self.default_language, "default", None

    # ------------------------------------------------------------------
    # Strategy: combined heuristic + langdetect (with conflict resolution)
    # ------------------------------------------------------------------

    def _detect_combined_langdetect(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """
        Smart combination of heuristic + langdetect probabilities.

        Special handling for:
        - Catalan: langdetect is better at detecting 'ca'
        - Dutch: trust langdetect when heuristic didn't say 'en'
        - East Asian: prioritize Japanese (kana detection), then Chinese
        - Spanish/English: trust heuristic (strong keyword signals)

        Ported from legacy get_language_combined_heur_langdetect().
        """
        from .heuristic_detector import get_language_heuristics

        heur_lang = get_language_heuristics(
            text, default_lang=self.default_language,
        )

        # Get langdetect probabilities
        lang_probs = self._call_langdetect_probs(text)
        if not lang_probs:
            return heur_lang, "heuristic", None

        ld_top = max(lang_probs, key=lang_probs.get)
        ld_confidence = round(lang_probs[ld_top], 4)

        # --- Conflict resolution rules ---

        # Catalan: langdetect is reliable for 'ca'
        if ld_top == "ca":
            return "ca", "combined_langdetect", ld_confidence

        # Dutch: trust langdetect when heuristic didn't resolve to English
        if ld_top == "nl" and heur_lang != "en":
            return "nl", "combined_langdetect", ld_confidence

        # Spanish / English: heuristic has strong academic keyword signals
        if heur_lang in ("en", "es"):
            return heur_lang, "heuristic", None

        # East Asian: special handling
        if ld_top in ("ja", "zh", "ko"):
            if ld_top == "ja" or heur_lang == "ja":
                return "ja", "combined_langdetect", ld_confidence

            if ld_top == "zh":
                return "zh", "combined_langdetect", ld_confidence

            # Korean top, check second-best
            sorted_probs = sorted(
                lang_probs.items(), key=lambda x: x[1], reverse=True,
            )
            if len(sorted_probs) > 1 and sorted_probs[1][0] == "zh":
                return "zh", "combined_langdetect", round(sorted_probs[1][1], 4)

            return "ja", "combined_langdetect", ld_confidence

        # Agreement: both say the same thing
        if ld_top == heur_lang:
            return ld_top, "combined_langdetect", ld_confidence

        # Non-East-Asian langdetect result when heuristic was inconclusive
        if ld_top not in ("ko", "ja", "zh"):
            return ld_top, "combined_langdetect", ld_confidence

        return self.default_language, "default", None

    # ------------------------------------------------------------------
    # Strategy: combined heuristic + E5 (fallback for 'und')
    # ------------------------------------------------------------------

    def _detect_combined_e5(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """
        Two-step: heuristic first, E5 model only when heuristic
        returns 'und'.

        Ported from legacy get_language_combined_heur_e5().
        """
        from .heuristic_detector import get_language_heuristics

        heur_lang = get_language_heuristics(text, default_lang="und")

        if heur_lang != "und":
            return heur_lang, "heuristic", None

        # E5 fallback
        if self._model is not None:
            return self._call_e5(text)

        return self.default_language, "default", None

    # ------------------------------------------------------------------
    # Model-specific callers
    # ------------------------------------------------------------------

    def _call_langdetect(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """Call langdetect, return top prediction."""
        try:
            if self._model_probs is not None:
                probs = self._model_probs(text)
                if probs:
                    top = probs[0]
                    lang = top.lang.split("-")[0]
                    return lang, "langdetect", round(top.prob, 4)

            lang = self._model(text).split("-")[0]
            return lang, "langdetect", None
        except Exception:
            return self.default_language, "default", None

    def _call_langdetect_probs(self, text: str) -> Dict[str, float]:
        """Get langdetect probability distribution as dict."""
        if self._model_probs is None:
            return {}
        try:
            probs = self._model_probs(text)
            return {
                lang.lang.split("-")[0]: lang.prob
                for lang in probs
            }
        except Exception:
            return {}

    def _call_e5(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """Call E5 transformer model."""
        # E5 language list (fixed order matching model output)
        languages = [
            "ar", "eu", "br", "ca", "zh", "zh", "zh", "cv", "cs", "dv",
            "nl", "en", "eo", "et", "fr", "fy", "ka", "de", "el", "cnh",
            "id", "ia", "it", "ja", "kab", "rw", "ky", "lv", "mt", "mn",
            "fa", "pl", "pt", "ro", "rm", "ru", "sah", "sl", "es", "sv",
            "ta", "tt", "tr", "uk", "cy",
        ]

        try:
            import torch

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model.to(device)
            self._model.eval()

            tokenized = self._tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask,
                )

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            top_prob, top_idx = torch.topk(probs, 1)

            lang = languages[top_idx[0][0].item()]
            confidence = round(top_prob[0][0].item(), 4)

            return lang, "e5", confidence

        except Exception as e:
            logger.warning("E5 detection failed: %s", e)
            return self.default_language, "default", None

    def _call_fasttext(
        self, text: str,
    ) -> Tuple[str, str, Optional[float]]:
        """Call FastText model."""
        try:
            predictions = self._model.predict(text)
            label = predictions[0][0]
            confidence = float(predictions[1][0])

            # Extract 3-letter code → 2-letter
            code_3 = label.replace("__label__", "").split("_")[0]
            lang = self._lang_code_3_to_2(code_3)

            return lang, "fasttext", round(confidence, 4)

        except Exception as e:
            logger.warning("FastText detection failed: %s", e)
            return self.default_language, "default", None

    @staticmethod
    def _lang_code_3_to_2(code_3: str) -> str:
        """Convert ISO 639-3 → ISO 639-1. Returns 'und' if unknown."""
        try:
            import pycountry
            language = pycountry.languages.get(alpha_3=code_3)
            return language.alpha_2 if hasattr(language, "alpha_2") else "und"
        except (AttributeError, LookupError):
            return "und"

    # ------------------------------------------------------------------
    # Standalone API
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect language for a single text string (standalone usage).

        Returns
        -------
        dict
            {"language": "es", "method": "heuristic", "confidence": None}
        """
        lang, method_used, confidence = self._detect_one(text)
        return {
            "language": lang,
            "method": method_used,
            "confidence": confidence,
        }