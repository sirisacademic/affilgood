"""
Translation component for non-Latin script affiliations.

Translates affiliations in non-Latin scripts (Chinese, Japanese, Arabic,
Russian, Persian, Korean, Thai, Greek, etc.) to English before NER and
entity linking.

Uses a small instruction-following LLM (default: Qwen2.5-0.5B-Instruct,
~1GB) for zero-shot translation of research affiliations.

Pipeline position: after language detection, before NER.

    Span → Language → **Translation** → NER → Entity Linking → Geocoding
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Languages that use non-Latin scripts and benefit from translation
NON_LATIN_LANGUAGES = {
    "zh",  # Chinese
    "ja",  # Japanese
    "ko",  # Korean
    "ar",  # Arabic
    "ru",  # Russian
    "fa",  # Persian/Farsi
    "uk",  # Ukrainian
    "bg",  # Bulgarian
    "sr",  # Serbian (Cyrillic)
    "mk",  # Macedonian
    "ka",  # Georgian
    "hy",  # Armenian
    "he",  # Hebrew
    "th",  # Thai
    "hi",  # Hindi
    "bn",  # Bengali
    "ta",  # Tamil
    "te",  # Telugu
    "ml",  # Malayalam
    "kn",  # Kannada
    "my",  # Burmese
    "km",  # Khmer
    "lo",  # Lao
    "si",  # Sinhala
    "am",  # Amharic
    "el",  # Greek
}

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class AffiliationTranslator:
    """
    Translates non-Latin script affiliations to English.

    Only activates for languages in NON_LATIN_LANGUAGES.
    Preserves the original text in item["original_text"].

    Parameters
    ----------
    model_name : str
        HuggingFace instruction model for translation.
    device : str or None
        "cpu" (default), "cuda", or None (auto).
    extra_languages : set or None
        Additional language codes to translate (beyond the built-in list).
    verbose : bool
        Verbose logging.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = "cpu",
        extra_languages: Optional[set] = None,
        verbose: bool = False,
    ):
        self.model_name = model_name or DEFAULT_MODEL
        self.device = device or "cpu"
        self.verbose = verbose
        self._model = None
        self._tokenizer = None

        self.target_languages = set(NON_LATIN_LANGUAGES)
        if extra_languages:
            self.target_languages |= set(extra_languages)

    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto",
            device_map=self.device if self.device == "auto" else None,
        )
        if self.device not in ("auto",):
            self._model.to(self.device)
        self._model.eval()

        if self.verbose:
            logger.info("Translator loaded: %s on %s", self.model_name, self.device)

    def translate_one(self, text: str) -> str:
        """Translate a single affiliation string to English."""
        self._ensure_model()

        messages = [
            {
                "role": "system",
                "content": (
                    "Translate the following research affiliation to English. "
                    "Keep institution names, city names, and country names. "
                    "Output ONLY the translation, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ]

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        import torch
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
            )

        translated = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        return translated

    def translate(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Pipeline interface: translate non-Latin items in place.

        For items with a detected non-Latin language:
          - Saves original text to item["original_text"]
          - Replaces item["raw_text"] with English translation
          - Sets item["translated"] = True

        Items with Latin-script languages pass through unchanged.
        """
        results = []
        n_translated = 0

        for item in items:
            out = dict(item)
            lang = out.get("language_info", {}).get("language", "en")

            if lang in self.target_languages:
                raw = out.get("raw_text", "")
                if raw.strip():
                    translated = self.translate_one(raw)
                    out["original_text"] = raw
                    out["raw_text"] = translated
                    out["translated"] = True
                    n_translated += 1

                    if self.verbose:
                        logger.info("[Translate] %s → %s", raw[:60], translated[:60])

            results.append(out)

        if self.verbose and n_translated > 0:
            print(f"[Translate] Translated {n_translated}/{len(items)} items")

        return results

    def free(self):
        """Release model memory."""
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