"""
Geocoding component for AffilGood.

Uses OSM Nominatim to enrich pipeline items with structured
location data (coordinates, city, region, country).

Optionally enriches with NUTS region codes (EU only) when
coordinates fall inside NUTS polygons.

Data files
----------
This component ships with package data in affilgood/components/data/:
- country_data.tsv
- nuts/NUTS_RG_01M_2021_4326.shp (+ .dbf, .shx, .prj)

These are resolved at runtime via _get_data_dir().
"""

import csv
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data directory resolution
# ------------------------------------------------------------------

def _get_data_dir() -> Path:
    return Path(__file__).parent / "data"


# ------------------------------------------------------------------
# Country data helpers
# ------------------------------------------------------------------

class CountryInfo(NamedTuple):
    """Structured country metadata from country_data.tsv."""
    name_short: str
    iso3: str
    continent: str
    un_region: str


def _load_country_data(
    data_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, CountryInfo]]:
    """
    Load country mappings from country_data.tsv.

    Builds two lookups:
    - name_to_short: maps any variant (name_official, name_short, ISO2, ISO3)
                     → name_short  (e.g. "Kingdom of Spain" → "Spain")
    - short_to_info: maps name_short → CountryInfo with iso3, continent, un_region

    Returns (name_to_short, short_to_info)
    """
    name_to_short: Dict[str, str] = {}
    short_to_info: Dict[str, CountryInfo] = {}

    tsv_path = data_dir / "country_data.tsv"
    if tsv_path.exists():
        try:
            with open(tsv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    short = row.get("name_short", "").strip()
                    official = row.get("name_official", "").strip()
                    iso3 = row.get("ISO3", "").strip()
                    iso2 = row.get("ISO2", "").strip()
                    continent = row.get("continent", "").strip()
                    un_region = row.get("UNregion", "").strip()

                    if not short:
                        continue

                    # Map all variants → name_short (first entry wins)
                    for key in (short, official, iso3, iso2):
                        if key:
                            name_to_short.setdefault(key, short)

                    # Full metadata keyed by name_short
                    if short not in short_to_info:
                        short_to_info[short] = CountryInfo(
                            name_short=short,
                            iso3=iso3,
                            continent=continent,
                            un_region=un_region,
                        )
        except Exception as e:
            logger.warning("Failed to load country_data.tsv: %s", e)

    return name_to_short, short_to_info


def _normalize_country_name(
    country: str,
    name_to_short: Dict[str, str],
    coco_module: Any = None,
) -> Optional[str]:
    """
    Normalize a country string to name_short.

    Examples:
        "Kingdom of Spain" → "Spain"
        "Italia"           → "Italy"  (via country_converter)
        "US"               → "United States"
    """
    if not country or not isinstance(country, str):
        return None

    country_clean = country.strip()

    if country_clean in name_to_short:
        return name_to_short[country_clean]

    country_title = country_clean.title()
    if country_title in name_to_short:
        return name_to_short[country_title]

    if coco_module is not None:
        try:
            result = coco_module.convert(names=country_clean, to="name_short")
            if isinstance(result, str) and result != "not found":
                return result
        except Exception:
            pass

    return country_clean


# ------------------------------------------------------------------
# Cache (SQLite)
# ------------------------------------------------------------------

class _GeoCache:
    """SQLite-backed cache. Only stores successful (non-None) results."""

    def __init__(self, db_path: str, expiration_days: int = 30):
        self.db_path = db_path
        self.expiration_seconds = expiration_days * 86400
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS geocache (
                    query TEXT PRIMARY KEY,
                    result TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created
                ON geocache(created_at)
            """)
            self._conn.commit()
        except Exception as e:
            logger.warning("Failed to initialize geocache: %s", e)
            self._conn = None

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        if self._conn is None:
            return None
        try:
            cursor = self._conn.execute(
                "SELECT result, created_at FROM geocache WHERE query = ?",
                (query,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            result_json, created_at = row
            if time.time() - created_at > self.expiration_seconds:
                self._conn.execute("DELETE FROM geocache WHERE query = ?", (query,))
                self._conn.commit()
                return None
            return json.loads(result_json)
        except Exception:
            return None

    def put(self, query: str, result: Dict[str, Any]):
        if self._conn is None or not result:
            return
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO geocache (query, result, created_at)
                   VALUES (?, ?, ?)""",
                (query, json.dumps(result), time.time()),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning("Failed to cache geocoding result: %s", e)

    def stats(self) -> Dict[str, Any]:
        if self._conn is None:
            return {"enabled": False}
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM geocache")
            count = cursor.fetchone()[0]
            return {"enabled": True, "db_path": self.db_path, "entries": count}
        except Exception:
            return {"enabled": True, "db_path": self.db_path, "entries": "unknown"}

    def clear(self):
        if self._conn is None:
            return
        try:
            self._conn.execute("DELETE FROM geocache")
            self._conn.commit()
        except Exception as e:
            logger.warning("Failed to clear geocache: %s", e)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ------------------------------------------------------------------
# Nominatim client
# ------------------------------------------------------------------

class _NominatimClient:
    """Thin wrapper around geopy Nominatim with retry and rate limiting."""

    MAX_RETRIES = 3
    MIN_DELAY = 1.0

    def __init__(self, user_agent: str = "affilgood"):
        from geopy.geocoders import Nominatim
        self._geocoder = Nominatim(user_agent=user_agent, timeout=10)
        self._last_request_time = 0.0

    def geocode(
        self, query: str, featuretype: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                self._rate_limit()
                result = self._geocoder.geocode(
                    query, addressdetails=True, language="en",
                    featuretype=featuretype,
                )
                if result is None and featuretype:
                    self._rate_limit()
                    result = self._geocoder.geocode(
                        query, addressdetails=True, language="en",
                    )
                if result is None:
                    return None
                return self._parse_result(result)
            except Exception as e:
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    logger.warning("Geocoding failed for '%s': %s", query, e)
                    return None

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_DELAY:
            time.sleep(self.MIN_DELAY - elapsed)
        self._last_request_time = time.time()

    @staticmethod
    def _parse_result(result) -> Optional[Dict[str, Any]]:
        raw = getattr(result, "raw", None)
        if raw is None:
            return None

        address = raw.get("address", {})
        addresstype = raw.get("addresstype", "")

        if addresstype and addresstype not in address:
            return None

        city = None
        for key in ("city", "town", "village", "municipality"):
            city = address.get(key)
            if city:
                break

        parsed = {
            "CITY": city,
            "STATE_DISTRICT": address.get("state_district"),
            "COUNTY": address.get("county"),
            "PROVINCE": address.get("province"),
            "STATE": address.get("state"),
            "REGION": address.get("region"),
            "COUNTRY": address.get("country"),
            "COORDS": (raw.get("lat"), raw.get("lon")),
            "OSM_ID": raw.get("osm_id"),
        }

        has_data = any(
            v is not None for k, v in parsed.items()
            if k not in ("COORDS", "OSM_ID")
        )
        if not has_data and parsed["COORDS"] == (None, None):
            return None

        return {k: v for k, v in parsed.items() if v is not None}


# ------------------------------------------------------------------
# NUTS lookup
# ------------------------------------------------------------------

class _NUTSLookup:
    """
    Spatial lookup of NUTS regions from coordinates.

    Expected shapefile: <data_dir>/nuts/NUTS_RG_01M_2021_4326.shp

    Download from Eurostat:
        https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/nuts
    """

    def __init__(self, shapefile_path: str, verbose: bool = False):
        import geopandas as gpd

        self._gdf = gpd.read_file(shapefile_path)
        self._gdf = self._gdf[
            ["NUTS_ID", "NUTS_NAME", "LEVL_CODE", "geometry"]
        ].to_crs("EPSG:4326")

        self._by_level: Dict[int, Any] = {}
        for level in (0, 1, 2, 3):
            subset = self._gdf[self._gdf["LEVL_CODE"] == level].copy()
            if not subset.empty:
                self._by_level[level] = subset

        if verbose:
            total = len(self._gdf)
            levels = sorted(self._by_level.keys())
            print(f"[NUTS] Loaded {total} polygons, levels: {levels}")

    def lookup(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        from shapely.geometry import Point

        point = Point(lon, lat)
        result: Dict[str, Any] = {}

        for level, gdf_level in self._by_level.items():
            mask = gdf_level.geometry.contains(point)
            matches = gdf_level[mask]
            if not matches.empty:
                row = matches.iloc[0]
                result[f"nuts{level}_id"] = row["NUTS_ID"]
                result[f"nuts{level}_name"] = row["NUTS_NAME"]

        return result if result else None


# ------------------------------------------------------------------
# Search query construction
# ------------------------------------------------------------------

def _clean_string(s: str) -> str:
    s = s.title()
    parts = []
    for part in s.split(","):
        cleaned = re.sub(
            r'[!()\-\[\]{};:\'"\\<>./?@#$%^&*_~·]+', "", part
        ).strip()
        if cleaned:
            parts.append(cleaned)
    return ", ".join(parts)


def _build_search_query(ner_entities: Dict[str, Any]) -> Optional[str]:
    """Build "featuretype:query" from NER entities, or None."""
    city = _first(ner_entities.get("CITY"))
    region = _first(ner_entities.get("REGION"))
    country = _first(ner_entities.get("COUNTRY"))

    if city:
        featuretype = "city"
        parts = [city]
        if region:
            parts.append(region)
        if country:
            parts.append(country)
    elif region:
        featuretype = "state"
        parts = [region]
        if country:
            parts.append(country)
    elif country:
        featuretype = "country"
        parts = [country]
    else:
        return None

    query_str = _clean_string(", ".join(parts))
    return f"{featuretype}:{query_str}"


def _first(lst: Any) -> Optional[str]:
    if isinstance(lst, list) and lst:
        return lst[0]
    return None


# ------------------------------------------------------------------
# Public component
# ------------------------------------------------------------------

class Geocoder:
    """
    Geocoding component for the AffilGood pipeline.

    Parameters
    ----------
    verbose : bool
        Enable verbose logging.
    cache_enabled : bool
        Enable SQLite-backed geocoding cache.
    cache_dir : str or None
        Directory for cache database. Defaults to package data dir.
    cache_expiration_days : int
        Cache entry expiration in days.
    user_agent : str
        User agent string for Nominatim API.
    data_dir : str or None
        Directory containing country_data.tsv.
        Defaults to affilgood/components/data/.
    add_nuts : bool
        Enable NUTS region enrichment (requires shapefile).
    nuts_shapefile : str or None
        Explicit path to NUTS shapefile. If None, auto-detected
        in data_dir/nuts/.
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        cache_expiration_days: int = 30,
        user_agent: str = "affilgood",
        data_dir: Optional[str] = None,
        add_nuts: bool = False,
        nuts_shapefile: Optional[str] = None,
    ):
        self.verbose = verbose

        pkg_data_dir = _get_data_dir()
        resolved_data_dir = Path(data_dir) if data_dir else pkg_data_dir
        resolved_cache_dir = Path(cache_dir) if cache_dir else pkg_data_dir

        # Load country mappings
        self._name_to_short, self._short_to_info = _load_country_data(resolved_data_dir)

        if verbose:
            n = len(self._name_to_short)
            m = len(self._short_to_info)
            print(f"[Geocoder] Loaded {n} country mappings, {m} countries from {resolved_data_dir}")

        # country_converter (optional fallback)
        self._coco = None
        try:
            import country_converter as coco
            self._coco = coco
        except ImportError:
            if verbose:
                print("[Geocoder] country_converter not available, using mappings only")

        # Cache
        self._cache: Optional[_GeoCache] = None
        if cache_enabled:
            db_path = str(resolved_cache_dir / "geocache.sqlite")
            self._cache = _GeoCache(db_path, expiration_days=cache_expiration_days)
            if verbose:
                stats = self._cache.stats()
                print(f"[Geocoder] Cache: {stats.get('entries', 0)} entries at {db_path}")

        # Nominatim client
        self._client = _NominatimClient(user_agent=user_agent)

        # NUTS lookup (optional)
        self._nuts: Optional[_NUTSLookup] = None
        self.add_nuts = add_nuts

        if add_nuts:
            nuts_path = nuts_shapefile or self._find_nuts_shapefile(resolved_data_dir)
            if nuts_path and Path(nuts_path).exists():
                try:
                    self._nuts = _NUTSLookup(str(nuts_path), verbose=verbose)
                except Exception as e:
                    if verbose:
                        print(f"[Geocoder] Failed to load NUTS shapefile: {e}")
                    self._nuts = None
            else:
                if verbose:
                    expected = nuts_path or str(resolved_data_dir / "nuts" / "NUTS_RG_01M_2021_4326.shp")
                    nuts_dir = resolved_data_dir / "nuts"
                    print(
                        f"[Geocoder] NUTS shapefile not found at: {expected}\n"
                        f"  Download from: https://ec.europa.eu/eurostat/web/gisco/"
                        f"geodata/statistical-units/nuts\n"
                        f"  Place .shp/.dbf/.shx/.prj in: {nuts_dir}"
                    )

    @staticmethod
    def _find_nuts_shapefile(data_dir: Path) -> Optional[str]:
        candidates = [
            data_dir / "nuts" / "NUTS_RG_01M_2021_4326.shp",
            data_dir / "nuts" / "NUTS_RG_01M_2021_4326_shp.shp",
            data_dir / "NUTS_RG_01M_2021_4326.shp",
            data_dir / "NUTS_RG_01M_2021_4326_shp" / "NUTS_RG_01M_2021_4326.shp",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return str(candidates[0])

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def normalize(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            out = dict(item)
            try:
                osm_data = self._geocode_item(out)
                out["osm"] = [osm_data] if osm_data else []
            except Exception as e:
                if self.verbose:
                    print(f"[Geocoder] Failed for '{out.get('raw_text', '')[:60]}': {e}")
                out["osm"] = []
            results.append(out)
        return results

    # ------------------------------------------------------------------
    # Core geocoding logic
    # ------------------------------------------------------------------

    def _geocode_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ner_list = item.get("ner", [])
        if not ner_list or not isinstance(ner_list, list):
            return None

        ner_entities = ner_list[0] if isinstance(ner_list[0], dict) else {}
        if not ner_entities:
            return None

        query = _build_search_query(ner_entities)
        if not query:
            return None

        if self.verbose:
            print(f"[Geocoder] Query: {query}")

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(query)
            if cached is not None:
                if self.verbose:
                    print("[Geocoder]   Cache hit")
                return self._enrich_result(cached, ner_entities)

        # Call Nominatim
        featuretype, _, location_query = query.partition(":")
        osm_result = self._client.geocode(
            location_query,
            featuretype=featuretype if featuretype != "settlement" else None,
        )

        if osm_result is None:
            if self.verbose:
                print("[Geocoder]   No result from Nominatim")
            return None

        # Cache raw OSM result (before enrichment)
        if self._cache is not None:
            self._cache.put(query, osm_result)

        return self._enrich_result(osm_result, ner_entities)

    def _enrich_result(
        self,
        osm_result: Dict[str, Any],
        ner_entities: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich an OSM result with:
        - COUNTRY → name_short (e.g. "Kingdom of Spain" → "Spain")
        - COUNTRY_CODE (ISO3)
        - CONTINENT
        - UN_REGION
        - NUTS codes (if enabled)
        """
        result = dict(osm_result)

        # Normalize country
        country_raw = result.get("COUNTRY")
        if country_raw:
            country_short = _normalize_country_name(
                country_raw, self._name_to_short, self._coco
            )
            if country_short:
                result["COUNTRY"] = country_short

                # Look up full metadata
                info = self._short_to_info.get(country_short)
                if info:
                    if info.iso3:
                        result["COUNTRY_CODE"] = info.iso3
                    if info.continent:
                        result["CONTINENT"] = info.continent
                    if info.un_region:
                        result["UN_REGION"] = info.un_region

        # NUTS enrichment
        if self._nuts is not None:
            coords = result.get("COORDS")
            if coords and len(coords) == 2:
                try:
                    lat = float(coords[0])
                    lon = float(coords[1])
                    nuts_data = self._nuts.lookup(lat, lon)
                    if nuts_data:
                        result["NUTS"] = nuts_data
                except (TypeError, ValueError):
                    pass

        return result

    # ------------------------------------------------------------------
    # Standalone helpers
    # ------------------------------------------------------------------

    def normalize_country(self, country: str) -> Optional[Dict[str, Any]]:
        """
        Normalize a single country name.

        Returns {"name": "Spain", "code": "ESP",
                 "continent": "Europe", "un_region": "Southern Europe"}
        """
        name = _normalize_country_name(
            country, self._name_to_short, self._coco
        )
        if not name:
            return None

        info = self._short_to_info.get(name)
        if info:
            return {
                "name": name,
                "code": info.iso3 or None,
                "continent": info.continent or None,
                "un_region": info.un_region or None,
            }

        return {"name": name, "code": None, "continent": None, "un_region": None}

    def cache_stats(self) -> Dict[str, Any]:
        if self._cache is None:
            return {"enabled": False}
        return self._cache.stats()

    def clear_cache(self):
        if self._cache is not None:
            self._cache.clear()
            if self.verbose:
                print("[Geocoder] Cache cleared")

    def close(self):
        if self._cache is not None:
            self._cache.close()