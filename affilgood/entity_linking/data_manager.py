import os
import sys
import boto3
import gzip
import json
import requests
import zipfile
from io import BytesIO
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime, timedelta
from .constants import *
from unidecode import unidecode
from .utils.text_utils import get_variants_list
from .utils.translation_mappings import translate_institution_name

### TODO: Split into specific data manager classes for the different linkers.

# Ensure S2AFF is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH)))
from s2aff.consts import PATHS

class DataManager:
    """Manages data updates, S3 syncing, and file resolution for entity linking."""

    def __init__(self):
        self.data_s2aff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), S2AFF_PATH, "data"))
        if not os.path.exists(self.data_s2aff_path):
            print(f"Creating directory: {self.data_s2aff_path}")
            os.makedirs(self.data_s2aff_path, exist_ok=True)

    def get_s3_client(self):
        """Returns an S3 client with unsigned configuration."""
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def list_s3_objects(self, bucket, prefix):
        """Lists objects in an S3 bucket with a given prefix."""
        s3 = self.get_s3_client()
        paginator = s3.get_paginator("list_objects_v2")
        response_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        return [obj for page in response_iterator if "Contents" in page for obj in page["Contents"]]

    def should_skip_s3_key(self, s3_key):
        """Determines if an S3 key should be skipped based on OMIT_S2AFF substrings."""
        if not s3_key:
            return False  # If the key is None, it cannot match OMIT_S2AFF
        return any(substring in s3_key for substring in OMIT_S2AFF)

    def download_file(self, url_or_bucket, key_or_path, local_path, is_s3=False, omit_s2aff=False):
        """
        Downloads a file from S3 or a URL to a local path, skipping if it matches OMIT_S2AFF.
        
        Args:
            url_or_bucket (str): The URL or S3 bucket name.
            key_or_path (str): The S3 key or URL path.
            local_path (str): Local path where the file should be saved.
            is_s3 (bool): Whether the source is S3 (True) or an HTTP URL (False).
        """
        # Check for omission
        if omit_s2aff and self.should_skip_s3_key(key_or_path):
            print(f"Skipping download of {key_or_path} due to OMIT_S2AFF rules.")
            return
        # Ensure the local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            if is_s3:
                # Validate key_or_path
                if not key_or_path:
                    print(f"Invalid key or path: {key_or_path}. Skipping download.")
                    return
                print(f"Downloading from S3: {key_or_path} to {local_path}")
                s3 = self.get_s3_client()
                s3.download_file(url_or_bucket, key_or_path, local_path)
            else:
                # Validate url_or_bucket
                if os.path.exists(local_path):
                    print(f"Using existing local file {local_path}")
                    return
                if not url_or_bucket:
                    print(f"Invalid url: {url_or_bucket}. Skipping download.")
                    return
                print(f"Downloading from URL: {url_or_bucket} to {local_path}")
                response = requests.get(url_or_bucket)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            print(f"Failed to download {key_or_path} to {local_path}: {e}")

    def resolve_s2aff_paths(self):
        """Converts missing file paths in PATHS to their corresponding URLs."""
        for key, path in PATHS.items():
            if not os.path.exists(path):
                print(f"File '{path}' does not exist. Resolving to URL.")
                PATHS[key] = f"https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/s2aff-release/{os.path.basename(path)}"

    def ensure_s2aff_files(self):
        """Ensures all required files in PATHS exist locally."""
        for key, path in PATHS.items():
            if path.startswith("http"):
                local_path = os.path.join(self.data_s2aff_path, os.path.basename(path))
                if not os.path.exists(local_path):
                    print(f"Ensuring file: {key}, URL: {path}")
                    self.download_file(path, os.path.basename(path), local_path, omit_s2aff=True)
            elif not os.path.exists(path):
                print(f"File {path} is missing but not resolvable to a URL.")
                raise FileNotFoundError(f"Required file {key} not found at {path}.")
        # Update OpenAlex work counts
        self.update_openalex_works_counts()

    def get_local_ror_dumps(self):
        """Returns a sorted list of local ROR dump files."""
        files = [os.path.join(self.data_s2aff_path, f) for f in os.listdir(self.data_s2aff_path) if "ror-data.json" in f]
        return sorted(files, reverse=True)

    def get_latest_ror(self, ror_dump_path=None):
        """Fetches the latest ROR dump or returns the most recent local one."""
        #print(f"ror_dump_path: {ror_dump_path}")
        if ror_dump_path is None and ROR_DUMP_PATH:
            ror_dump_path = ROR_DUMP_PATH
            if os.path.exists(ror_dump_path):
                print(f"Using existing ROR dump from {ror_dump_path}")
                return ror_dump_path
            elif os.path.exists(os.path.join(self.data_s2aff_path, ror_dump_path)):
                return os.path.join(self.data_s2aff_path, ror_dump_path)
            else:
                print(f"ROR dump not found in {ror_dump_path} nor {os.path.join(self.data_s2aff_path, ror_dump_path)}.")
        print(f"Fetching latest ROR dump from {ROR_DUMP_LINK}")
        try:
            response = requests.get(ROR_DUMP_LINK)
            response.raise_for_status()
            download_url = response.json()["hits"]["hits"][0]["files"][0]["links"]["self"]
            file_name = response.json()["hits"]["hits"][0]["files"][0]["key"]
            file_path = os.path.join(self.data_s2aff_path, file_name)
            self.download_file(download_url, file_path, file_path)
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(self.data_s2aff_path)
            return self.get_local_ror_dumps()[0]
        except Exception as e:
            print(f"Failed to fetch ROR dump: {e}")
            local_dumps = self.get_local_ror_dumps()
            if local_dumps:
                print(f"Using existing ROR dump: {local_dumps[0]}")
                return local_dumps[0]
            else:
                return None

    def should_update_openalex(self, days_threshold=UPDATE_OPENALEX_WORK_COUNTS_OLDER_THAN):
        """Checks if OpenAlex work counts need updating."""
        path = PATHS.get("openalex_works_counts", "")
        if not os.path.exists(path):
            print(f"File '{path}' not found. Updating...")
            return True
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        return age > timedelta(days=days_threshold)
   
    def update_openalex_works_counts(self):
        """Updates OpenAlex work counts."""
        if self.should_update_openalex():
            bucket = "openalex"
            prefix = "data/institutions/"
            s3 = self.get_s3_client()
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            works_count_dict = {}
            for page in pages:
                for obj in page["Contents"]:
                    if obj["Key"].startswith("part") and obj["Key"].endswith(".gz"):
                        print(obj)
                        obj = s3.get_object(Bucket=bucket, Key=obj["Key"])
                        with gzip.GzipFile(fileobj=BytesIO(obj["Body"].read())) as f:
                            for line in f:
                                line = json.loads(line)
                                ror_id = line["ror"]
                                works_count = line["works_count"]
                                works_count_dict[ror_id] = works_count
            # Convert works_count_dict to a dataframe and save it
            df = pd.DataFrame.from_dict(works_count_dict, orient="index", columns=["works_count"]).reset_index()
            df.columns = ["ror", "works_count"]
            df.to_csv(PATHS["openalex_works_counts"], index=False)
            print(f"Updated works counts saved to {PATHS['openalex_works_counts']}")

    def ensure_whoosh_index(self, whoosh_index_dir):
        """Ensures the Whoosh index exists, creating it if necessary."""
        if os.path.exists(whoosh_index_dir) and os.listdir(whoosh_index_dir):
            print(f"Using existing Whoosh index in {whoosh_index_dir}")
            return True
        print(f"Whoosh index not found in {whoosh_index_dir}, creating it...")
        return self.create_whoosh_index(whoosh_index_dir)

    def create_whoosh_index(self, whoosh_index_dir):
        """Creates a Whoosh index from the latest ROR data."""
        from whoosh.index import create_in
        from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED
        from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
        # Get the latest ROR data
        ror_file = self.get_latest_ror()
        if not ror_file:
            print("Failed to get ROR data for Whoosh index creation")
            return False
        # Create the index directory if it doesn't exist
        os.makedirs(whoosh_index_dir, exist_ok=True)
        # Create the schema
        schema = Schema(
            # Core ROR fields
            ror_id=ID(stored=True),
            ror_name=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            ror_name_normalized=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            # Names and aliases
            aliases_text=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            aliases_list=STORED,
            acronyms=KEYWORD(stored=True),
            # Location information
            city=TEXT(analyzer=StandardAnalyzer(), stored=True),
            region=TEXT(analyzer=StandardAnalyzer(), stored=True),
            country=TEXT(analyzer=StandardAnalyzer(), stored=True),
            country_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            # Combined fields for search
            name=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            location=TEXT(analyzer=StandardAnalyzer(), stored=True),
            # Parent organizations
            parent=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            # For boosting and filtering
            ror_name_length=STORED
        )
        # Create the index
        ix = create_in(whoosh_index_dir, schema)
        writer = ix.writer()
        # Process and add documents
        print(f"Reading ROR data from {ror_file}...")
        with open(ror_file, 'r', encoding='utf-8') as f:
            ror_data = json.load(f)
        print(f"Indexing {len(ror_data)} organizations...")
        for org in ror_data:
            doc = self._process_organization_for_whoosh(org)
            writer.add_document(**doc)
        # Commit changes
        print("Committing changes to Whoosh index...")
        writer.commit()
        print(f"Whoosh index created successfully in {whoosh_index_dir}")
        return True

    def _process_organization_for_whoosh(self, org):
        """Process a ROR organization entry for Whoosh indexing with uniform translation support."""
        original_name = org['name']
        doc = {
            'ror_id': org['id'],
            'ror_name': original_name,
            'ror_name_length': len(original_name.split())
        }
        # For searchable fields, include both original and normalized versions
        normalized_name = unidecode(original_name)
        if normalized_name != original_name:
            doc['ror_name_normalized'] = normalized_name
        # Generate translations for the institution name
        translated_names = translate_institution_name(original_name)
        # Also translate any aliases to catch more variations
        aliases = org.get('aliases', [])
        labels = [label['label'] for label in org.get('labels', [])]
        # Translate aliases if they exist
        translated_aliases = []
        for alias in aliases + labels:
            translated_aliases.extend(translate_institution_name(alias))
        # Use get_variants_list for aliases with translated names
        all_aliases = aliases + labels + translated_names + translated_aliases
        # Generate all variants including normalized ones
        all_alias_variants = get_variants_list(all_aliases)
        doc['aliases_text'] = ' ||| '.join(all_alias_variants)
        doc['aliases_list'] = all_alias_variants
        # Process acronyms
        acronyms = org.get('acronyms', [])
        all_acronyms_variants = get_variants_list(acronyms)
        doc['acronyms'] = ' ||| '.join(all_acronyms_variants)
        # Combined field with all names
        all_names = [original_name, normalized_name] + acronyms + all_aliases
        doc['name'] = ' ||| '.join(filter(None, all_names))
        # Process location information
        if 'addresses' in org and len(org['addresses']) > 0:
            address = org['addresses'][0]  # Use first address
            doc['city'] = address.get('city', '')
            # Try to get region from geonames
            if 'geonames_city' in address and 'geonames_admin1' in address['geonames_city']:
                doc['region'] = address['geonames_city']['geonames_admin1'].get('name', '')
            # Process country information
            if 'country' in org:
                doc['country'] = org['country'].get('country_code', '')
                doc['country_name'] = org['country'].get('country_name', '')
            # Combined location field
            location_parts = [
                doc.get('city', ''),
                doc.get('region', ''),
                doc.get('country_name', '')
            ]
            doc['location'] = ' '.join(filter(None, location_parts))
        # Process parent organizations
        parents = [r['label'] for r in org.get('relationships', []) 
                   if r.get('type') == 'Parent']
        doc['parent'] = ' '.join(parents)
        return doc


