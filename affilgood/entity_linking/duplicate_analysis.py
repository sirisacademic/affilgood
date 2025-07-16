import pandas as pd

# Load your data
df = pd.read_parquet('wikidata_cache/complete_datasets/dataset_France_extended.parquet')

print(f"Total rows: {len(df)}")
print(f"Unique IDs: {len(df[['id']].drop_duplicates())}")
print(f"Duplicates: {len(df) - len(df[['id']].drop_duplicates())}")

# Method 1: Find IDs that appear more than once
duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]['id'].unique()
print(f"\nNumber of IDs with duplicates: {len(duplicate_ids)}")

# Method 2: Show sample duplicates
print("\n=== SAMPLE DUPLICATES ===")
for i, duplicate_id in enumerate(duplicate_ids[:5]):  # Show first 5 duplicate IDs
    print(f"\n--- Duplicate {i+1}: ID = {duplicate_id} ---")
    duplicate_rows = df[df['id'] == duplicate_id]
    print(f"This ID appears {len(duplicate_rows)} times:")
    
    # Show relevant columns to understand the differences
    cols_to_show = ['id', 'name', 'location_type', 'city', 'country_name', 'aliases', 'external_ids']
    available_cols = [col for col in cols_to_show if col in df.columns]
    
    for idx, row in duplicate_rows.iterrows():
        print(f"  Row {idx}:")
        for col in available_cols:
            print(f"    {col}: {row[col]}")
        print()

# Method 3: Count duplicates by location_type (if that column exists)
if 'location_type' in df.columns:
    print("\n=== DUPLICATES BY LOCATION_TYPE ===")
    duplicate_df = df[df.duplicated(subset=['id'], keep=False)]
    location_counts = duplicate_df['location_type'].value_counts()
    print(location_counts)

# Method 4: Analyze what makes rows different
print("\n=== DUPLICATE ANALYSIS ===")
sample_duplicate_id = duplicate_ids[0]
sample_rows = df[df['id'] == sample_duplicate_id]

print(f"Analyzing duplicates for ID: {sample_duplicate_id}")
print(f"Number of rows: {len(sample_rows)}")

# Check which columns have different values
print("\nColumns with different values across duplicates:")
for col in df.columns:
    unique_values = sample_rows[col].dropna().unique()
    if len(unique_values) > 1:
        print(f"  {col}: {list(unique_values)}")

# Method 5: Show duplicate statistics
print("\n=== DUPLICATE STATISTICS ===")
duplicate_counts = df['id'].value_counts()
duplicate_stats = duplicate_counts[duplicate_counts > 1]

print(f"IDs with 2 duplicates: {len(duplicate_stats[duplicate_stats == 2])}")
print(f"IDs with 3 duplicates: {len(duplicate_stats[duplicate_stats == 3])}")
print(f"IDs with 4+ duplicates: {len(duplicate_stats[duplicate_stats >= 4])}")

print(f"\nMost duplicated IDs:")
print(duplicate_stats.head(10))

# Method 6: Check if duplicates are from different processing steps
if 'aliases' in df.columns and 'external_ids' in df.columns:
    print("\n=== CHECKING DUPLICATE SOURCES ===")
    
    # Look at a few duplicate cases
    for duplicate_id in duplicate_ids[:3]:
        duplicate_rows = df[df['id'] == duplicate_id]
        print(f"\nID {duplicate_id}:")
        print(f"  Names: {duplicate_rows['name'].unique()}")
        if 'city' in df.columns:
            print(f"  Cities: {duplicate_rows['city'].unique()}")
        if 'location_type' in df.columns:
            print(f"  Location types: {duplicate_rows['location_type'].unique()}")
