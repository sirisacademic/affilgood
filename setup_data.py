#!/usr/bin/env python3
"""
Download and prepare AffilGood data files.

Run once after installation:
    python setup_data.py

Downloads data from GitHub Release:
  - ROR registry dump
  - Pre-built FAISS dense index
  - NUTS shapefiles
  - Country/continent lookups
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# ── Configuration ──

HF_REPO = "SIRIS-Lab/affilgood-data"
VERSION = "v2.0.0"
RELEASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/affilgood-data-{VERSION}.zip"
ZIP_FILENAME = f"affilgood-data-{VERSION}.zip"


def find_repo_root() -> Path:
    """Find the affilgood repository root."""
    script_dir = Path(__file__).parent.resolve()
    if (script_dir / "affilgood" / "components").exists():
        return script_dir
    if (script_dir.parent / "affilgood" / "components").exists():
        return script_dir.parent
    try:
        import affilgood
        pkg_dir = Path(affilgood.__file__).parent.parent
        if (pkg_dir / "affilgood" / "components").exists():
            return pkg_dir
    except ImportError:
        pass
    return script_dir


def download_with_progress(url: str, dest: Path):
    """Download a file with progress reporting."""
    print(f"  Downloading from:\n  {url}")
    print(f"  Saving to: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  [{pct:5.1f}%] {mb_down:.1f} / {mb_total:.1f} MB", end="", flush=True)

    urlretrieve(url, dest, reporthook=progress_hook)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download AffilGood data files from GitHub Release"
    )
    parser.add_argument(
        "--data-dir", type=str,
        help="Override target directory (default: auto-detect repo root)"
    )
    parser.add_argument(
        "--url", type=str, default=RELEASE_URL,
        help=f"Override download URL (default: {RELEASE_URL})"
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the zip file after extraction"
    )
    args = parser.parse_args()

    if args.data_dir:
        target_dir = Path(args.data_dir).resolve()
    else:
        target_dir = find_repo_root()

    print(f"AffilGood Data Setup ({VERSION})")
    print(f"=" * 50)
    print(f"Target: {target_dir}")
    print()

    # Check if data already exists
    index_path = target_dir / "affilgood/components/entity_linking/data/ror/dense/faiss.index"
    if index_path.exists():
        print(f"Data already exists at {target_dir}")
        response = input("Re-download and overwrite? [y/N] ").strip().lower()
        if response != "y":
            print("Skipped.")
            return

    # Download
    zip_path = target_dir / ZIP_FILENAME
    print(f"Step 1: Download ({ZIP_FILENAME})")
    download_with_progress(args.url, zip_path)

    # Extract
    print(f"\nStep 2: Extract")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"  {len(members)} files")
        zf.extractall(target_dir)
    print(f"  Extracted to {target_dir}")

    # Clean up zip
    if not args.keep_zip:
        zip_path.unlink()
        print(f"  Removed {ZIP_FILENAME}")

    # Verify
    print(f"\nStep 3: Verify")
    checks = [
        ("ROR records", "affilgood/components/entity_linking/data/ror/ror_records.jsonl"),
        ("FAISS index", "affilgood/components/entity_linking/data/ror/dense/faiss.index"),
        ("FAISS IDs", "affilgood/components/entity_linking/data/ror/dense/faiss_ids.json"),
        ("FAISS texts", "affilgood/components/entity_linking/data/ror/dense/faiss_texts.json"),
        ("FAISS meta", "affilgood/components/entity_linking/data/ror/dense/faiss_meta.json"),
    ]

    all_ok = True
    for name, rel_path in checks:
        p = target_dir / rel_path
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  ✓ {name} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name} — NOT FOUND")
            all_ok = False

    if all_ok:
        print(f"\n✅ All data files ready!")
    else:
        print(f"\n⚠️  Some files missing. Check the extraction path.")

    meta_path = target_dir / "affilgood/components/entity_linking/data/ror/dense/faiss_meta.json"
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        print(f"\n  Index: {meta.get('num_vectors', '?'):,} vectors, "
              f"dim={meta.get('dim', '?')}, "
              f"encoder={meta.get('encoder', '?')}")


if __name__ == "__main__":
    main()
