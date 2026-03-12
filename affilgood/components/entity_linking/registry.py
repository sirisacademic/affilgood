"""
Registry data management for entity linking.

Downloads, loads, and normalizes registry data (ROR, etc.)
into a common RegistryRecord schema for indexing and retrieval.

ROR dump source: Zenodo (https://zenodo.org/communities/ror-data)
"""

import json
import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Zenodo API for latest ROR dump
_ZENODO_ROR_API = "https://zenodo.org/api/records?communities=ror-data&sort=mostrecent&size=1"


# ------------------------------------------------------------------
# Common record schema
# ------------------------------------------------------------------

@dataclass
class RegistryRecord:
    """
    Normalized record from any registry.

    All registries (ROR, Wikidata, SICRIS, etc.) normalize to this
    schema so retrievers/rerankers are registry-agnostic.
    """
    id: str                             # "https://ror.org/052gg0110"
    name: str                           # canonical display name
    aliases: List[str] = field(default_factory=list)    # alternative names
    acronyms: List[str] = field(default_factory=list)   # "MIT", "CSIC"
    labels: List[str] = field(default_factory=list)     # translated names
    country: str = ""                   # "Spain"
    country_code: str = ""              # "ES"
    city: str = ""                      # "Barcelona"
    types: List[str] = field(default_factory=list)      # ["Education"]
    status: str = "active"              # "active" or "withdrawn"
    source: str = ""                    # "ror", "wikidata", etc.
    url: str = ""                       # institutional website
    parent_id: Optional[str] = None     # for hierarchical orgs
    successor_id: Optional[str] = None  # for inactive → active resolution

    def all_names(self) -> List[str]:
        """All searchable name variants (for indexing)."""
        names = [self.name]
        names.extend(self.aliases)
        names.extend(self.labels)
        names.extend(a for a in self.acronyms if len(a) >= 2)
        # Deduplicate preserving order
        seen = set()
        unique = []
        for n in names:
            n_lower = n.strip().lower()
            if n_lower and n_lower not in seen:
                seen.add(n_lower)
                unique.append(n.strip())
        return unique

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": self.aliases,
            "acronyms": self.acronyms,
            "labels": self.labels,
            "country": self.country,
            "country_code": self.country_code,
            "city": self.city,
            "types": self.types,
            "status": self.status,
            "source": self.source,
            "url": self.url,
            "parent_id": self.parent_id,
            "successor_id": self.successor_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistryRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------
# Registry Manager
# ------------------------------------------------------------------

class RegistryManager:
    """
    Manages registry data (download, load, normalize).

    Parameters
    ----------
    data_dir : str or Path or None
        Directory for registry data. Defaults to
        <package>/entity_linking/data/
    verbose : bool
        Verbose logging.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.verbose = verbose

        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_records(
        self,
        registry: str = "ror",
        *,
        active_only: bool = True,
    ) -> List[RegistryRecord]:
        """
        Get normalized records for a registry.

        Downloads data on first call if not present.
        Caches normalized records to disk as JSONL.

        Parameters
        ----------
        registry : str
            "ror" (default). Others can be added.
        active_only : bool
            Filter to active records only (default: True).

        Returns
        -------
        list of RegistryRecord
        """
        if registry == "ror":
            return self._load_ror(active_only=active_only)
        else:
            raise ValueError(f"Unknown registry: '{registry}'. Supported: 'ror'")

    # ------------------------------------------------------------------
    # ROR
    # ------------------------------------------------------------------

    def _load_ror(self, *, active_only: bool = True) -> List[RegistryRecord]:
        """Load ROR records. Download if not present."""
        ror_dir = self.data_dir / "ror"
        ror_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached normalized records
        normalized_path = ror_dir / "ror_records.jsonl"
        if normalized_path.exists():
            if self.verbose:
                print(f"[Registry] Loading cached ROR records from {normalized_path}")
            return self._load_jsonl(normalized_path, active_only=active_only)

        # Check for raw dump
        dump_path = self._find_ror_dump(ror_dir)
        if dump_path is None:
            if self.verbose:
                print("[Registry] ROR dump not found, downloading...")
            dump_path = self.download_ror()

        if dump_path is None:
            raise FileNotFoundError(
                f"ROR dump not found in {ror_dir}. "
                f"Download manually from https://zenodo.org/communities/ror-data "
                f"and place the JSON file in {ror_dir}/"
            )

        # Normalize and cache
        if self.verbose:
            print(f"[Registry] Normalizing ROR dump: {dump_path}")

        records = self._normalize_ror_dump(dump_path)
        self._save_jsonl(records, normalized_path)

        if self.verbose:
            print(f"[Registry] Saved {len(records)} records to {normalized_path}")

        if active_only:
            records = [r for r in records if r.status == "active"]
            if self.verbose:
                print(f"[Registry] {len(records)} active records")

        return records

    def download_ror(self, force: bool = False) -> Optional[Path]:
        """
        Download latest ROR dump from Zenodo.

        Returns path to the extracted JSON file, or None on failure.
        """
        ror_dir = self.data_dir / "ror"
        ror_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if not force:
            existing = self._find_ror_dump(ror_dir)
            if existing:
                if self.verbose:
                    print(f"[Registry] ROR dump already present: {existing}")
                return existing

        try:
            import urllib.request

            if self.verbose:
                print("[Registry] Fetching latest ROR release from Zenodo...")

            # Get latest release metadata
            req = urllib.request.Request(
                _ZENODO_ROR_API,
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                releases = json.loads(resp.read())

            if not releases:
                logger.warning("No ROR releases found on Zenodo")
                return None

            # Find the zip file in the latest release
            latest = releases[0] if isinstance(releases, list) else releases.get("hits", {}).get("hits", [None])[0]
            if latest is None:
                return None

            files = latest.get("files", [])
            zip_url = None
            for f in files:
                fname = f.get("key", "") or f.get("filename", "")
                if fname.endswith(".zip"):
                    zip_url = f.get("links", {}).get("self") or f.get("url")
                    break

            if zip_url is None:
                logger.warning("No zip file found in latest ROR release")
                return None

            # Download
            zip_path = ror_dir / "ror_dump.zip"
            if self.verbose:
                print(f"[Registry] Downloading: {zip_url}")

            urllib.request.urlretrieve(zip_url, str(zip_path))

            # Extract
            if self.verbose:
                print(f"[Registry] Extracting {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zf:
                json_files = [n for n in zf.namelist() if n.endswith(".json")]
                if not json_files:
                    logger.warning("No JSON file found in ROR zip")
                    return None
                # Extract the largest JSON (the dump itself)
                json_files.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
                zf.extract(json_files[0], str(ror_dir))
                extracted = ror_dir / json_files[0]

            # Clean up zip
            zip_path.unlink(missing_ok=True)

            if self.verbose:
                print(f"[Registry] ROR dump extracted: {extracted}")

            return extracted

        except Exception as e:
            logger.warning("Failed to download ROR dump: %s", e)
            if self.verbose:
                print(f"[Registry] Download failed: {e}")
            return None

    def _find_ror_dump(self, ror_dir: Path) -> Optional[Path]:
        """Find an existing ROR dump JSON in the directory."""
        # Look for common naming patterns
        candidates = list(ror_dir.glob("v*.json"))  # v1.63-2025-04-03-ror-data.json
        candidates += list(ror_dir.glob("ror-data*.json"))
        candidates += list(ror_dir.glob("*ror*.json"))

        # Exclude our normalized file
        candidates = [
            p for p in candidates
            if p.name != "ror_records.jsonl" and p.suffix == ".json"
        ]

        if not candidates:
            return None

        # Return most recently modified
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # ROR normalization
    # ------------------------------------------------------------------

    def _normalize_ror_dump(self, dump_path: Path) -> List[RegistryRecord]:
        """
        Parse ROR dump JSON and normalize to RegistryRecord list.

        Handles both ROR schema v1 and v2 formats.
        """
        with open(dump_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records = []
        for entry in raw:
            try:
                record = self._normalize_ror_entry(entry)
                if record is not None:
                    records.append(record)
            except Exception as e:
                rid = entry.get("id", "?")
                logger.debug("Failed to normalize ROR entry %s: %s", rid, e)

        return records

    @staticmethod
    def _normalize_ror_entry(entry: Dict[str, Any]) -> Optional[RegistryRecord]:
        """
        Normalize a single ROR entry (supports v1 and v2 schema).
        """
        ror_id = entry.get("id", "")
        if not ror_id:
            return None

        # --- Schema v2 (names array) ---
        if "names" in entry:
            return RegistryManager._normalize_ror_v2(entry)

        # --- Schema v1 (name + aliases + labels) ---
        return RegistryManager._normalize_ror_v1(entry)

    @staticmethod
    def _normalize_ror_v1(entry: Dict[str, Any]) -> RegistryRecord:
        """Normalize ROR schema v1."""
        # Country
        country_obj = entry.get("country", {})
        country_name = country_obj.get("country_name", "")
        country_code = country_obj.get("country_code", "")

        # City (from addresses array)
        city = ""
        addresses = entry.get("addresses", [])
        if addresses:
            city = addresses[0].get("city", "")

        # Labels (translated names)
        labels = [
            lab.get("label", "")
            for lab in entry.get("labels", [])
            if lab.get("label")
        ]

        # Links
        links = entry.get("links", [])
        url = links[0] if links else ""

        # Relationships → parent, successor
        parent_id = None
        successor_id = None
        for rel in entry.get("relationships", []):
            rel_type = rel.get("type", "").lower()
            if rel_type == "parent":
                parent_id = rel.get("id")
            elif rel_type == "successor":
                successor_id = rel.get("id")

        return RegistryRecord(
            id=entry["id"],
            name=entry.get("name", ""),
            aliases=entry.get("aliases", []),
            acronyms=entry.get("acronyms", []),
            labels=labels,
            country=country_name,
            country_code=country_code,
            city=city,
            types=entry.get("types", []),
            status=entry.get("status", "active"),
            source="ror",
            url=url,
            parent_id=parent_id,
            successor_id=successor_id,
        )

    @staticmethod
    def _normalize_ror_v2(entry: Dict[str, Any]) -> RegistryRecord:
        """Normalize ROR schema v2 (names array structure)."""
        name = ""
        aliases = []
        acronyms = []
        labels = []

        for name_entry in entry.get("names", []):
            value = name_entry.get("value", "")
            types = name_entry.get("types", [])

            if not value:
                continue

            if "ror_display" in types:
                name = value
            elif "acronym" in types:
                acronyms.append(value)
            elif "alias" in types:
                aliases.append(value)
            elif "label" in types:
                labels.append(value)

        # Fallback: if no ror_display, use first name
        if not name:
            all_names = entry.get("names", [])
            if all_names:
                name = all_names[0].get("value", "")

        # Location
        country_name = ""
        country_code = ""
        city = ""

        locations = entry.get("locations", [])
        if locations:
            loc = locations[0]
            geo = loc.get("geonames_details", {})
            city = geo.get("name", "")
            country_name = geo.get("country_name", "")
            country_code = geo.get("country_code", "")

        # Links
        url = ""
        for link in entry.get("links", []):
            if link.get("type") == "website":
                url = link.get("value", "")
                break

        # Relationships
        parent_id = None
        successor_id = None
        for rel in entry.get("relationships", []):
            rel_type = rel.get("type", "").lower()
            if rel_type == "parent":
                parent_id = rel.get("id")
            elif rel_type == "successor":
                successor_id = rel.get("id")

        # Types (v2 uses list of strings)
        types = entry.get("types", [])

        return RegistryRecord(
            id=entry["id"],
            name=name,
            aliases=aliases,
            acronyms=acronyms,
            labels=labels,
            country=country_name,
            country_code=country_code,
            city=city,
            types=types,
            status=entry.get("status", "active"),
            source="ror",
            url=url,
            parent_id=parent_id,
            successor_id=successor_id,
        )

    # ------------------------------------------------------------------
    # JSONL persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_jsonl(records: List[RegistryRecord], path: Path):
        """Save records to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def _load_jsonl(
        path: Path, *, active_only: bool = True,
    ) -> List[RegistryRecord]:
        """Load records from JSONL file."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rec = RegistryRecord.from_dict(d)
                if active_only and rec.status != "active":
                    continue
                records.append(rec)
        return records

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        ror_dir = self.data_dir / "ror"
        has_dump = self._find_ror_dump(ror_dir) is not None
        has_normalized = (ror_dir / "ror_records.jsonl").exists()
        return {
            "data_dir": str(self.data_dir),
            "ror_dump_present": has_dump,
            "ror_normalized": has_normalized,
        }