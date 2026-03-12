"""
Organization type classification for AffilGood.

Classifies institutions into a two-level taxonomy using context-augmented
RoBERTa classifiers (SIRIS-Lab/acty2de).

Pipeline position: after Entity Linking, before/alongside Geocoding.

Taxonomy:
  Level 1: Company, Healthcare, Higher Education, Individual, Other,
           Primary and Secondary Education, Public Administration,
           Research Organization, Unknown

  Level 2: EDU.primary, EDU.secondary, EDU.vocational, HEA.hospital,
           HEA.network, HEA.non_hospital_care, HEA.other, HEI.institution,
           HEI.other, Individual, OTH.general, OTH.learned_society,
           OTH.museum, OTH.network, OTH.oth, PRC.large, PRC.sme,
           PUB.intl, PUB.local, PUB.national, PUB.regional,
           REC.facility, REC.institute, REC.lab, REC.other, Unknown

Context is fetched via DuckDuckGo search (top 10 results).
All queries and classifications are cached to avoid redundant lookups.
"""

import logging
import sqlite3
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Level 1 → Level 2 prefix mapping for validation
LVL1_TO_LVL2_PREFIX = {
    "Company": "PRC.",
    "Healthcare": "HEA.",
    "Higher Education": "HEI.",
    "Individual": "Individual",
    "Other": "OTH.",
    "Primary and Secondary Education": "EDU.",
    "Public Administration": "PUB.",
    "Research Organization": "REC.",
    "Unknown": "Unknown",
}

DEFAULT_LVL1_MODEL = "SIRIS-Lab/acty2de-roberta_lvl1_ctx"
DEFAULT_LVL2_MODEL = "SIRIS-Lab/acty2de-roberta_lvl2_ctx"


class OrgTypeCache:
    """SQLite cache for DDG queries and classification results."""

    def __init__(self, cache_path: Path):
        self._path = cache_path
        self._conn = sqlite3.connect(str(cache_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


class OrganizationTypeClassifier:
    """
    Classifies organizations into a two-level taxonomy.

    Uses DuckDuckGo search for context augmentation, then
    RoBERTa classifiers for lvl1 and lvl2 prediction.

    Parameters
    ----------
    lvl1_model : str
        HuggingFace model for level 1 classification.
    lvl2_model : str
        HuggingFace model for level 2 classification.
    device : str
        "cpu" or "cuda".
    cache_dir : str or Path or None
        Directory for SQLite cache. None = component data dir.
    search_sleep : float
        Sleep between DDG queries (rate limiting).
    verbose : bool
        Verbose logging.
    """

    def __init__(
        self,
        lvl1_model: Optional[str] = None,
        lvl2_model: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        search_sleep: float = 1.0,
        verbose: bool = False,
    ):
        self.lvl1_model_name = lvl1_model or DEFAULT_LVL1_MODEL
        self.lvl2_model_name = lvl2_model or DEFAULT_LVL2_MODEL
        self.device = device
        self.search_sleep = search_sleep
        self.verbose = verbose

        # Lazy-loaded classifiers
        self._lvl1 = None
        self._lvl2 = None

        # Cache
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "data"
        cache_path = Path(cache_dir) / "orgtype_cache.sqlite"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = OrgTypeCache(cache_path)

        if verbose:
            print(f"[OrgType] Cache: {cache_path}")

    def _ensure_models(self):
        """Lazy-load classifiers."""
        if self._lvl1 is not None:
            return

        from transformers import pipeline as hf_pipeline

        self._lvl1 = hf_pipeline(
            "text-classification",
            model=self.lvl1_model_name,
            truncation=True,
            max_length=512,
            device=self.device,
        )
        self._lvl2 = hf_pipeline(
            "text-classification",
            model=self.lvl2_model_name,
            truncation=True,
            max_length=512,
            device=self.device,
        )

        if self.verbose:
            print(f"[OrgType] Loaded classifiers on {self.device}")

    # Country code → DDG region mapping (most common)
    _COUNTRY_TO_REGION = {
        "united kingdom": "uk-en", "united states": "us-en",
        "france": "fr-fr", "germany": "de-de", "spain": "es-es",
        "italy": "it-it", "netherlands": "nl-nl", "belgium": "be-nl",
        "portugal": "pt-pt", "poland": "pl-pl", "sweden": "se-sv",
        "norway": "no-no", "denmark": "dk-da", "finland": "fi-fi",
        "austria": "at-de", "switzerland": "ch-de", "ireland": "ie-en",
        "czech republic": "cz-cs", "czechia": "cz-cs",
        "australia": "au-en", "canada": "ca-en", "new zealand": "nz-en",
        "japan": "jp-jp", "south korea": "kr-kr", "china": "cn-zh",
        "india": "in-en", "brazil": "br-pt", "mexico": "mx-es",
        "argentina": "ar-es", "turkey": "tr-tr", "türkiye": "tr-tr",
        "russia": "ru-ru", "israel": "il-he", "south africa": "za-en",
        "singapore": "sg-en", "malaysia": "my-en", "thailand": "th-th",
        "indonesia": "id-en", "philippines": "ph-en",
        "greece": "gr-el", "romania": "ro-ro", "hungary": "hu-hu",
        "croatia": "hr-hr", "slovenia": "si-sl", "slovakia": "sk-sk",
        "bulgaria": "bg-bg", "serbia": "rs-sr", "ukraine": "ua-uk",
        "estonia": "ee-et", "latvia": "lv-lv", "lithuania": "lt-lt",
    }

    # ------------------------------------------------------------------
    # Context fetching (DDG)
    # ------------------------------------------------------------------

    def _get_context(self, query: str, country: str = "") -> Optional[str]:
        """Fetch context via DuckDuckGo, with region calibration and caching."""
        cache_key = f"ddgs_v3::{query.lower().strip()}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            if cached == "__NONE__":
                return None
            return cached

        try:
            from ddgs import DDGS

            # Determine DDG region from country
            region = "wt-wt"  # worldwide default
            if country:
                region = self._COUNTRY_TO_REGION.get(
                    country.lower().strip(), "wt-wt"
                )

            # Quoted query for exact org name match
            search_query = f'{query}'

            ddgs = DDGS()

            # Try bing first, fallback to brave
            results = None
            for backend in ("bing", "brave"):
                try:
                    results = ddgs.text(
                        search_query,
                        region=region,
                        max_results=10,
                        backend=backend,
                    )
                    if results:
                        break
                except Exception:
                    continue

            if not results:
                self._cache.put(cache_key, "__NONE__")
                return None

            context_parts = [f"{r['title']}: {r['body']}" for r in results]
            context = f"{query}\n" + "\n".join(context_parts)

            self._cache.put(cache_key, context)
            time.sleep(self.search_sleep)
            return context

        except Exception as e:
            if self.verbose:
                logger.warning("DDG search failed for '%s': %s", query, e)
            self._cache.put(cache_key, "__NONE__")
            time.sleep(self.search_sleep)
            return None

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_one(
        self,
        org_name: str,
        country: str = "",
    ) -> Dict[str, Optional[str]]:
        """
        Classify a single organization.

        Returns {"lvl1": "Higher Education", "lvl2": "HEI.institution"}
        or {"lvl1": None, "lvl2": None} if classification fails.
        """
        # Check classification cache
        cache_key = f"type::{org_name.lower().strip()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return json.loads(cached)

        self._ensure_models()

        # Build search query
        query = org_name
        if country:
            query = f"{org_name}, {country}"

        context = self._get_context(query, country=country)
        if not context:
            result = {"lvl1": None, "lvl2": None}
            self._cache.put(cache_key, json.dumps(result))
            return result

        # Classify
        pred_lvl1 = self._lvl1(context)
        pred_lvl2 = self._lvl2(context)

        lvl1 = pred_lvl1[0]["label"] if pred_lvl1 else None
        lvl2 = pred_lvl2[0]["label"] if pred_lvl2 else None

        # Validate: lvl2 must match lvl1 prefix
        if lvl1 and lvl2:
            expected_prefix = LVL1_TO_LVL2_PREFIX.get(lvl1, "")
            if expected_prefix and not lvl2.startswith(expected_prefix):
                lvl2 = None  # mismatch → drop lvl2

        result = {"lvl1": lvl1, "lvl2": lvl2}
        self._cache.put(cache_key, json.dumps(result))

        if self.verbose:
            logger.debug("OrgType: '%s' → %s / %s", org_name, lvl1, lvl2)

        return result

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def classify(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Pipeline interface: classify institution types for all items.

        Adds "type" field to each institution in entity_linking.
        """
        results = []
        n_classified = 0

        for item in items:
            out = dict(item)
            el = out.get("entity_linking", {})

            # Classify institutions only (not subunits)
            for inst in el.get("institutions", []):
                org_type = self._classify_institution(inst)
                inst["type"] = org_type
                if org_type.get("lvl1"):
                    n_classified += 1

            results.append(out)

        if self.verbose and n_classified:
            total = sum(
                len(item.get("entity_linking", {}).get("institutions", []))
                for item in items
            )
            print(f"[OrgType] Classified {n_classified}/{total} institutions")

        return results

    def _classify_institution(self, inst: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Classify a single institution dict."""
        # Get name and country
        id_field = inst.get("id")
        if isinstance(id_field, dict):
            name = id_field.get("ror_name") or inst.get("query", "")
            country = id_field.get("ror_country", "")
        else:
            name = inst.get("query") or inst.get("name", "")
            country = ""

        if not name:
            return {"lvl1": None, "lvl2": None}

        return self.classify_one(name, country)

    def free(self):
        """Release model memory."""
        self._lvl1 = None
        self._lvl2 = None
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass