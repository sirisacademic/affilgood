import os
from pathlib import Path

VERSION = "v2.0.0"
HF_URL = f"https://huggingface.co/datasets/SIRIS-Lab/affilgood-data/resolve/main/affilgood-data-{VERSION}.zip"

# Paths inside the zip that map to the cache root
# e.g. affilgood/components/entity_linking/data/ror/dense/faiss.index
#   → ~/.cache/affilgood/v2.0.0/ror/dense/faiss.index
_STRIP_PREFIXES = [
    "affilgood/components/entity_linking/data/",
    "affilgood/components/data/",
]


def _progress_hook(block_num: int, block_size: int, total_size: int):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100 / total_size)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
    else:
        mb_done = (block_num * block_size) / 1e6
        print(f"\r  Downloaded {mb_done:.1f} MB...", end="", flush=True)


def get_data_dir(override: str | None = None) -> Path:
    """Resolve data directory: explicit > env var > user cache."""
    if override:
        return Path(override)
    if env := os.environ.get("AFFILGOOD_DATA_DIR"):
        return Path(env)
    return Path.home() / ".cache" / "affilgood" / VERSION


def ensure_data(data_dir: Path | None = None, force: bool = False) -> Path:
    """
    Check data files exist; download if missing.
    Returns resolved data_dir.
    """
    d = get_data_dir(data_dir)
    sentinel = d / "ror" / "dense" / "faiss.index"

    # Fast path — no locking needed if data already present
    if sentinel.exists() and not force:
        return d

    # Slow path — acquire lock before downloading
    # Prevents parallel workers from downloading simultaneously
    from filelock import FileLock

    d.mkdir(parents=True, exist_ok=True)
    lock_path = d / ".download.lock"

    with FileLock(str(lock_path), timeout=600):  # 10-min timeout
        # Re-check inside lock: another worker may have finished while we waited
        if sentinel.exists() and not force:
            return d
        _download_and_extract(d)

    return d


def _download_and_extract(target: Path):
    """Download zip from HuggingFace and extract to target."""
    import zipfile
    import tempfile
    from urllib.request import urlretrieve

    target.mkdir(parents=True, exist_ok=True)

    print(f"[AffilGood] Data files not found. Downloading {VERSION} (~330 MB compressed)...")
    print(f"[AffilGood] Target: {target}")
    print(f"[AffilGood] To skip, set AFFILGOOD_DATA_DIR to an existing data directory.")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        urlretrieve(HF_URL, tmp_path, reporthook=_progress_hook)
        print()
        with zipfile.ZipFile(tmp_path) as zf:
            _extract_mapped(zf, target)
        print(f"[AffilGood] ✓ Data ready at {target}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_mapped(zf, target: Path):
    """
    Extract zip remapping the two data roots to the cache root.

    Zip layout:
      affilgood/components/entity_linking/data/ror/dense/faiss.index
      affilgood/components/entity_linking/data/ror/ror_records.jsonl
      affilgood/components/data/nuts/NUTS_RG_01M_2021_4326.shp
      ...

    Result in target (~/.cache/affilgood/v2.0.0/):
      ror/dense/faiss.index
      ror/dense/faiss_ids.json
      ror/dense/faiss_texts.json
      ror/dense/faiss_meta.json
      ror/ror_records.jsonl
      nuts/NUTS_RG_01M_2021_4326.shp
      nuts/NUTS_RG_01M_2021_4326.dbf
      ...
    """
    for member in zf.namelist():
        # Try each known prefix — use the first that matches
        rel = None
        for prefix in _STRIP_PREFIXES:
            if member.startswith(prefix):
                rel = member[len(prefix):]
                break

        # Skip entries that don't match any known prefix, or are directory entries
        if not rel or rel.endswith("/"):
            continue

        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(member))


def cli_download():
    """Entry point for the affilgood-download console script."""
    import argparse

    parser = argparse.ArgumentParser(description="Download AffilGood data files")
    parser.add_argument("--data-dir", default=None,
                        help="Override target directory (default: ~/.cache/affilgood/v2.0.0/)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if data files already exist")
    args = parser.parse_args()

    target = Path(args.data_dir) if args.data_dir else None
    resolved = ensure_data(data_dir=target, force=args.force)
    print(f"\n✅ Data ready at: {resolved}")