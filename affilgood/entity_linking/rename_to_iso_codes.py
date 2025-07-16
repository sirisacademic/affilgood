#!/usr/bin/env python3

import os
import shutil
import pycountry

# Define base paths
base_dir = "/home/pablo/affilgood/affilgood/entity_linking"
cache_dir = os.path.join(base_dir, "wikidata_cache", "complete_datasets")
hnsw_dir = os.path.join(base_dir, "hnsw_indices")
whoosh_dir = os.path.join(base_dir, "whoosh_indices")

# Country name to ISO code mapping
country_to_iso = {
    'Albania': 'AL',
    'Algeria': 'DZ', 
    'Bosnia and Herzegovina': 'BA',
    'Bulgaria': 'BG',
    'Croatia': 'HR',
    'Cyprus': 'CY',
    'Egypt': 'EG',
    'France': 'FR',
    'Greece': 'GR',
    'Israel': 'IL',
    'Italy': 'IT',
    'Jordan': 'JO',
    'Kosovo': 'XK',  # ISO 3166-1 alpha-2 code for Kosovo
    'Lebanon': 'LB',
    'Libya': 'LY',
    'Malta': 'MT',
    'Mauritania': 'MR',
    'MC': 'MC',  # Already ISO code (Monaco)
    'MD': 'MD',  # Already ISO code (Moldova)
    'Montenegro': 'ME',
    'Morocco': 'MA',
    'North Macedonia': 'MK',
    'Portugal': 'PT',
    'PS': 'PS',  # Already ISO code (Palestine)
    'Romania': 'RO',
    'RS': 'RS',  # Already ISO code (Serbia)
    'Slovenia': 'SI',
    'Spain': 'ES',
    'Syria': 'SY',
    'Tunisia': 'TN',
    'Turkey': 'TR',
    'Switzerland': 'CH',
}

def rename_cache_files():
    """Rename dataset cache files to use ISO codes."""
    print("=== Renaming dataset cache files ===")
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return
    
    files = os.listdir(cache_dir)
    
    for filename in files:
        if filename.startswith('dataset_') and filename.endswith(('.parquet', '.json')):
            # Extract country name from filename
            # Format: dataset_CountryName_extended.parquet/json
            parts = filename.replace('dataset_', '').split('_')
            if len(parts) >= 2:
                country_name = parts[0]
                
                # Handle special cases with spaces
                if country_name == 'North' and len(parts) >= 3:
                    country_name = 'North Macedonia'
                elif country_name == 'Bosnia' and len(parts) >= 4:
                    country_name = 'Bosnia and Herzegovina'
                
                if country_name in country_to_iso:
                    iso_code = country_to_iso[country_name]
                    
                    # Create new filename
                    extension = filename.split('.')[-1]
                    new_filename = f"dataset_{iso_code}_extended.{extension}"
                    
                    old_path = os.path.join(cache_dir, filename)
                    new_path = os.path.join(cache_dir, new_filename)
                    
                    if os.path.exists(old_path):
                        print(f"  {filename} -> {new_filename}")
                        shutil.move(old_path, new_path)
                    else:
                        print(f"  File not found: {filename}")
                else:
                    print(f"  Unknown country: {country_name} in {filename}")

def rename_index_directories(index_dir, index_type):
    """Rename index directories to use ISO codes."""
    print(f"\n=== Renaming {index_type} indices ===")
    
    if not os.path.exists(index_dir):
        print(f"Index directory not found: {index_dir}")
        return
    
    directories = os.listdir(index_dir)
    
    for dirname in directories:
        if dirname.startswith('wikidata_extended_'):
            # Extract country name from directory name
            country_name = dirname.replace('wikidata_extended_', '')
            
            if country_name in country_to_iso:
                iso_code = country_to_iso[country_name]
                
                # Create new directory name
                new_dirname = f"wikidata_extended_{iso_code}"
                
                old_path = os.path.join(index_dir, dirname)
                new_path = os.path.join(index_dir, new_dirname)
                
                if os.path.exists(old_path):
                    print(f"  {dirname} -> {new_dirname}")
                    shutil.move(old_path, new_path)
                else:
                    print(f"  Directory not found: {dirname}")
            else:
                print(f"  Unknown country: {country_name} in {dirname}")

def verify_renaming():
    """Verify the renaming was successful."""
    print("\n=== Verification ===")
    
    # Check cache files
    print("Dataset cache files:")
    if os.path.exists(cache_dir):
        files = sorted(os.listdir(cache_dir))
        for f in files:
            if f.startswith('dataset_') and f.endswith(('.parquet', '.json')):
                print(f"  {f}")
    
    # Check HNSW indices
    print("\nHNSW indices:")
    if os.path.exists(hnsw_dir):
        dirs = sorted(os.listdir(hnsw_dir))
        for d in dirs:
            if d.startswith('wikidata_extended_'):
                print(f"  {d}")
    
    # Check Whoosh indices
    print("\nWhoosh indices:")
    if os.path.exists(whoosh_dir):
        dirs = sorted(os.listdir(whoosh_dir))
        for d in dirs:
            if d.startswith('wikidata_extended_'):
                print(f"  {d}")

def main():
    print("Starting WikiData cache renaming to ISO codes...")
    
    # Create backup list before renaming
    print("\n=== Current files before renaming ===")
    verify_renaming()
    
    # Perform renaming
    rename_cache_files()
    rename_index_directories(hnsw_dir, "HNSW")
    rename_index_directories(whoosh_dir, "Whoosh")
    
    # Verify results
    verify_renaming()
    
    print("\n✅ Renaming complete!")
    print("\nISO code mapping used:")
    for country, iso in sorted(country_to_iso.items()):
        print(f"  {country} -> {iso}")

if __name__ == "__main__":
    main()
