from typing import List, Dict, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import os
import pandas as pd
from tqdm import tqdm
from typing import Optional, List
import pycountry
from pathlib import Path

# Get Languge Codes per Country
csv_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vR-4tZAXgYbEHDPUOz9USaOGFRuztTqeMidYMwaIqftWrGyzwkXeAgq1IOrhcd7EZ6bWrthbNFK0guQ/pub?gid=0&single=true&output=csv'

# Load into DataFrame
lang_codes_df = pd.read_csv(csv_url)

# Step 1: Create lookup of country_code → language codes
lang_lookup = lang_codes_df.set_index('country_exonym')['lang_codes'].str.split('|').to_dict()

organisation_types = {
    "business": "Q4830453",
    "enterprise": "Q6881511",
    "public company": "Q891723",
    "company": "Q783794",
    "non-profit": "Q163740",
    "public university": "Q875538",
    "university": "Q3918",
    "university hospital": "Q1059324",
    "hospital": "Q16917",
    "organization": "Q43229",
    "foundation": "Q157031",
    "association": "Q15911314",
    "research institute": "Q31855",
    "government agency": "Q327333",
    "government council": "Q58235902",
    "laboratory": "Q483242"
}

organisation_types_long = {'Q4830453': 'business',
 'Q163740': 'nonprofit organization',
 'Q43229': 'organization',
 'Q3918': 'university',
 'Q4287745': 'medical organization',
 'Q16917': 'hospital',
 'Q13226383': 'facility',
 'Q2385804': 'educational institution',
 'Q327333': 'government agency',
 'Q6881511': 'enterprise',
 'Q31855': 'research institute',
 'Q891723': 'public company',
 'Q23002054': 'private not-for-profit educational institution',
 'Q23002039': 'public educational institution of the United States',
 'Q902104': 'private university',
 'Q875538': 'public university',
 'Q483242': 'laboratory',
 'Q38723': 'higher education institution',
 'Q189004': 'college',
 'Q3152824': 'cultural institution',
 'Q708676': 'charitable organization',
 'Q4671277': 'academic institution',
 'Q1336920': 'community college',
 'Q3550864': 'French UMR',
 'Q33506': 'museum',
 'Q157031': 'foundation',
 'Q7315155': 'research center',
 'Q1664720': 'institute',
 'Q3914': 'school',
 'Q2085381': 'publisher',
 'Q955824': 'learned society',
 'Q494230': 'medical school',
 'Q207694': 'art museum',
 'Q23002042': 'private educational institution',
 'Q114853062': 'academic publisher',
 'Q1059324': 'university hospital',
 'Q5341295': 'educational organization',
 'Q7075': 'library',
 'Q41176': 'building',
 'Q4201890': 'institute of the Russian Academy of Sciences',
 'Q155271': 'think tank',
 'Q1365560': 'university of applied sciences',
 'Q10729872': 'medical association',
 'Q79913': 'non-governmental organization',
 'Q748019': 'scientific society',
 'Q7257872': 'public hospital',
 'Q108935461': 'research institution',
 'Q56351315': 'Japanese high school',
 'Q28564': 'public library',
 'Q1371037': 'institute of technology',
 'Q1143635': 'business school',
 'Q1663017': 'engineering school',
 'Q23002037': 'public educational institution',
 'Q166118': 'archive',
 'Q783794': 'company',
 'Q62078547': 'public research university',
 'Q398141': 'school district',
 'Q383092': 'art school',
 'Q829080': 'professional association',
 'Q484652': 'international organization',
 'Q4260475': 'medical facility',
 'Q1774587': 'hospital network',
 'Q2178147': 'trade association',
 'Q6954187': 'NHS foundation trust',
 'Q23002052': 'private for-profit educational institution',
 'Q1519799': 'ministry of health',
 'Q192350': 'ministry',
 'Q1110684': 'regulatory college',
 'Q26271642': 'library network',
 'Q15936437': 'research university',
 'Q1254933': 'astronomical observatory',
 'Q20857085': 'state agency of the United States',
 'Q21822439': 'further education college',
 'Q6954197': 'NHS trust',
 'Q2659904': 'government organization',
 'Q48204': 'voluntary association',
 'Q20857065': 'United States federal agency',
 'Q1377182': 'liberal arts college',
 'Q184644': 'conservatory',
 'Q5774403': 'historical society',
 'Q414147': 'academy of sciences',
 'Q615150': 'land-grant university',
 'Q17431399': 'national museum',
 'Q1970365': 'natural history museum',
 'Q1364302': 'ministry of agriculture',
 'Q15911314': 'association',
 'Q265662': 'national university',
 'Q2269756': 'ministry of education',
 'Q4284971': 'private hospital',
 'Q847027': 'grande école',
 'Q431603': 'advocacy group',
 'Q1785733': 'environmental organization',
 'Q16735822': 'history museum',
 'Q1497649': 'memory institution',
 'Q9826': 'high school',
 'Q588140': 'science museum',
 'Q270791': 'state-owned enterprise',
 'Q155076': 'juridical person',
 'Q209465': 'university campus',
 'Q17072837': 'medical college in India',
 'Q22806': 'national library',
 'Q98658352': 'higher education institution directly under Ministry of Education of the People’s Republic of China',
 'Q370258': 'junior college',
 'Q210999': 'psychiatric hospital',
 'Q644264': "children's hospital",
 'Q19644607': 'pharmaceutical company',
 'Q3551775': 'university in France',
 'Q786820': 'automobile manufacturer',
 'Q219577': 'holding company',
 'Q1802122': 'German state agency',
 'Q811979': 'architectural structure',
 'Q12317349': 'main library',
 'Q167346': 'botanical garden',
 'Q178790': 'labor union',
 'Q19869268': 'medical society',
 'Q20168706': 'Fraunhofer institute',
 'Q6979657': 'national public health institute',
 'Q1058914': 'software company',
 'Q6540832': 'liberal arts college in the United States',
 'Q658255': 'subsidiary',
 'Q917441': 'environment ministry',
 'Q17028020': 'vocational university',
 'Q194166': 'consortium',
 'Q245065': 'intergovernmental organization',
 'Q43501': 'zoo',
 'Q66344': 'central bank',
 'Q7188': 'government',
 'Q115154402': 'independent museum',
 'Q4409567': 'prefectural museum',
 'Q841248': 'college of technology in Japan',
 'Q64027599': 'gas station chain',
 'Q1328899': 'standards organization',
 'Q3343298': 'non-departmental public body',
 'Q161726': 'multinational corporation',
 'Q167037': 'corporation',
 'Q1966910': 'national academy',
 'Q159334': 'secondary school',
 'Q22687': 'bank',
 'Q16519632': 'scientific organization'}

class OrganisationIndex:
    def __init__(self):
        """
        :param countries: List of country Q-codes (e.g., ['Q30', 'Q145'] for USA, UK).
        :param org_types: List of organization type Q-codes (e.g., ['Q43229'] for company).
        """
        self.endpoint_url = "https://query.wikidata.org/sparql"
        self.headers = {"User-Agent": "OrganisationIndexBot/1.0 (your-email@example.com)"}

    def get_country_qids_from_wikidata(self):
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        query = """
        SELECT ?country ?countryLabel WHERE {
        ?country wdt:P31 wd:Q6256.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        
        countries = {res["countryLabel"]["value"]: res["country"]["value"].split("/")[-1] 
                    for res in results["results"]["bindings"]}
        return countries

    def fetch_results(self,type_qid, country_qid):

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f"""
        SELECT ?id ?name_en 
            (GROUP_CONCAT(DISTINCT CONCAT(LANG(?alias), ":", STR(?alias)); separator="|") AS ?aliases)
            (GROUP_CONCAT(DISTINCT CONCAT(LANG(?name_full), ":", STR(?name_full)); separator="|") AS ?all_names)
            (GROUP_CONCAT(DISTINCT ?acronym; separator="|") AS ?acronyms)
            ?city ?region ?country_name ?country_code
            (GROUP_CONCAT(DISTINCT ?parent_name; separator="|") AS ?relationships)
            (GROUP_CONCAT(DISTINCT ?website; separator="|") AS ?websites)
            ?coordinates ?formation_label
            ?hq_city ?hq_country_name
            ?ror ?grid ?lei ?crunchbase
        WHERE {{
        ?id wdt:P31 wd:{type_qid};  # instance of type
            wdt:P17 wd:{country_qid}.      # country

        OPTIONAL {{ ?id rdfs:label ?name_en. FILTER(LANG(?name_en) = "en") }}
        OPTIONAL {{ ?id rdfs:label ?name_full. }}
        OPTIONAL {{ ?id skos:altLabel ?alias. }}
        OPTIONAL {{ ?id wdt:P1813 ?acronym. }}
        OPTIONAL {{ ?id wdt:P6782 ?ror. }}
        OPTIONAL {{ ?id wdt:P2427 ?grid. }}
        OPTIONAL {{ ?id wdt:P1278 ?lei. }}
        OPTIONAL {{ ?id wdt:P2088 ?crunchbase. }}
        OPTIONAL {{ ?id wdt:P856 ?website. }}
        OPTIONAL {{ ?id wdt:P625 ?coordinates. }}
        OPTIONAL {{ 
            ?id wdt:P740 ?formation_location.
            ?formation_location rdfs:label ?formation_label.
            FILTER(LANG(?formation_label) = "en")
        }}

        OPTIONAL {{
            ?id wdt:P131 ?city_item.
            ?city_item wdt:P31/wdt:P279* wd:Q515.
            ?city_item rdfs:label ?city.
            FILTER(LANG(?city) = "en")

            OPTIONAL {{
            ?city_item wdt:P131 ?region_item.
            ?region_item rdfs:label ?region.
            FILTER(LANG(?region) = "en")
            }}

            OPTIONAL {{
            ?city_item wdt:P17 ?country_item.
            ?country_item rdfs:label ?country_name.
            ?country_item wdt:P298 ?country_code.
            FILTER(LANG(?country_name) = "en")
            }}
        }}

        OPTIONAL {{
            ?id wdt:P749 ?parent.
            ?parent rdfs:label ?parent_name.
            FILTER(LANG(?parent_name) = "en")
        }}

        OPTIONAL {{
            ?id wdt:P159 ?hq_item.
            ?hq_item rdfs:label ?hq_city.
            FILTER(LANG(?hq_city) = "en")
            OPTIONAL {{
            ?hq_item wdt:P17 ?hq_country.
            ?hq_country rdfs:label ?hq_country_name.
            FILTER(LANG(?hq_country_name) = "en")
            }}
        }}

        # Fallback if missing
        wd:{country_qid} rdfs:label ?fallback_country_label.
        FILTER(LANG(?fallback_country_label) = "en")
        wd:{country_qid} wdt:P298 ?fallback_country_code.

        BIND(COALESCE(?country_name, ?fallback_country_label) AS ?country_name)
        BIND(COALESCE(?country_code, ?fallback_country_code) AS ?country_code)

        }}
        GROUP BY ?id ?name_en ?city ?region ?country_name ?country_code
                ?coordinates ?formation_label ?hq_city ?hq_country_name
                ?ror ?grid ?website ?lei ?crunchbase
        """

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        results = sparql.query().convert()

        # Extract bindings
        rows = results['results']['bindings']

        # Convert to flat list of dicts
        flattened_rows = []
        for row in rows:
            flat_row = {}
            for key, value in row.items():
                flat_row[key] = value.get('value', None)
            flattened_rows.append(flat_row)

        return flattened_rows
        

    def group_external_ids(self, df, group_keys=['id', 'name_en']):
        # Detect which external ID columns actually exist
        possible_id_cols = ['crunchbase', 'ror', 'grid', 'lei']
        external_id_cols = [col for col in possible_id_cols if col in df.columns]

        # Group by organization and aggregate external IDs into sets (remove duplicates)
        grouped = df.groupby(group_keys)[external_id_cols].agg(
            lambda x: list(set(x.dropna()))
        ).reset_index()

        # Convert lists with single items to scalars (optional)
        def normalize(d):
            return {k: (v[0] if len(v) == 1 else v) for k, v in d.items()}

        # Build external_ids dictionary per row
        def build_ids(row):
            ids = {col: row[col] for col in external_id_cols if row[col]}
            return normalize(ids)

        grouped['external_ids'] = grouped.apply(build_ids, axis=1)

        return grouped[[*group_keys, 'external_ids']]


    def expand_organization_locations(self, row):
        base_data = {
            'id': row['id'],
            'name': row['name_en'],
            'aliases': row['aliases'],
            'all_names': row['all_names'],
            'acronyms': row.get('acronyms'),
            'relationships': row.get('relationships'),
            'websites': row.get('websites'),
            'external_ids': row.get('external_ids'),
        }

        entries = []

        city = row.get('city')
        region = row.get('region')
        country_name = row.get('country_name')
        country_code = row.get('country_code')
        hq_city = row.get('hq_city')
        hq_country_name = row.get('hq_country_name')
        coordinates = row.get('coordinates')
        formation_label = row.get('formation_label')

        # 1. Primary location
        if pd.notna(city) and pd.notna(country_name):
            entry = base_data.copy()
            entry.update({
                'location_type': 'primary',
                'city': city,
                'region': region,
                'country_name': country_name,
                'country_code': country_code,
            })
            entries.append(entry)

        # 1.b. Primary location with no city
        if pd.isna(city) and pd.notna(country_name) and country_name != hq_country_name:
            entry = base_data.copy()
            entry.update({
                'location_type': 'country_only',
                'city': None,
                'region': None,
                'country_name': country_name,
                'country_code': country_code,
                'coordinates': coordinates,
            })
            entries.append(entry)

        # 2. Headquarters (if different)
        if (
            pd.notna(hq_city) and pd.notna(hq_country_name) and
            (hq_city != city or hq_country_name != country_name)
        ):
            entry = base_data.copy()
            entry.update({
                'location_type': 'headquarters',
                'city': hq_city,
                'region': None,
                'country_name': hq_country_name,
                'country_code': None,
            })
            entries.append(entry)

        # 3. Fallback (Formation Label)
        if not entries and pd.notna(formation_label):
            city_fallback = formation_label if pd.isna(city) and pd.isna(hq_city) else None
            entry = base_data.copy()
            entry.update({
                'location_type': 'formation_label',
                'city': city_fallback,
                'region': None,
                'country_name': country_name,
                'country_code': country_code,
            })
            entries.append(entry)

        # 4. Fallback (Coordinates)
        if not entries and pd.notna(coordinates):
            entry = base_data.copy()
            entry.update({
                'location_type': 'coordinates',
                'city': None,
                'region': None,
                'country_name': country_name,
                'country_code': country_code,
                'coordinates': coordinates,
            })
            entries.append(entry)

        return entries


    # Step 2: Define helper to parse LANG:label format into a dict
    def parse_lang_values(self, value_str):
        if pd.isna(value_str):
            return {}
        parts = value_str.split('|')
        return {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts if ':' in p}

    # Step 3: Main processing
    def process_lang_name_variants(self, row):
        lang_codes = lang_lookup.get(row['country_name'], ['en'])
        
        # Parse all_names and aliases into dicts
        names_dict = self.parse_lang_values(row.get('all_names', ''))
        aliases_dict = self.parse_lang_values(row.get('aliases', ''))

        # Keep only official langs + English
        filtered_names = {k: v for k, v in names_dict.items() if k in lang_codes or k == 'en'}
        filtered_aliases = {k: v for k, v in aliases_dict.items() if k in lang_codes or k == 'en'}
        
        # Determine main name: prefer first match in lang_codes, fallback to English, else use current name_en
        name = None
        for lang in lang_codes:
            if lang in filtered_names:
                name = filtered_names[lang]
                break
        if not name:
            name = filtered_names.get('en', row['name'])

        return pd.Series({
            'name': name,
            'all_names': filtered_names,
            'aliases': filtered_aliases
        })
    
    def country_flag_emoji_from_name(self, name: str) -> str:
        """
        Returns the emoji flag for a given country name.
        """
        try:
            country = pycountry.countries.lookup(name)
            alpha_2 = country.alpha_2.upper()
            return ''.join(chr(127397 + ord(char)) for char in alpha_2)
        except LookupError:
            return ''
    
    def get_index(self, countries: Optional[List[str]] = None, org_types: Optional[List[str]] = None):
        # Load all countries/types if not provided
        name_to_qid= self.get_country_qids_from_wikidata()
        qid_to_country_name = {qid: name for name, qid in name_to_qid.items()}

        if isinstance(countries, str):
            if countries == "all":
                    countries = list(qid_to_country_name.keys())
            else:
                    countries = [countries]  # Wrap single country string into a list
        elif countries is None:
            countries = list(qid_to_country_name.keys())

        if isinstance(org_types, str):
            if org_types == "short":
                org_types = list(organisation_types.keys())
            elif org_types == "extended":
                org_types = list(organisation_types_extended.keys())
            else:
                org_types = [org_types]  # Wrap unknown single string into a list
        elif org_types is None:
            org_types = list(organisation_types.keys())

        temp_dir = Path("/tmp/org_index")
        temp_dir.mkdir(parents=True, exist_ok=True)

        all_rows = []

        
        for country in countries:
            country_name = qid_to_country_name.get(country, "Unknown")
            flag = self.country_flag_emoji_from_name(country_name)
            loop_desc = f"{flag} {country_name} ({country})"

            with tqdm(desc=loop_desc, total=1, leave=True) as pbar:
                for org_type in tqdm(org_types, desc="Fetching Org Types", leave=False):
                    try:
                        res = self.fetch_results(org_type, country)
                        df = pd.DataFrame(res)

                        if df.empty:
                            pbar.update(1)
                            continue

                        external_ids_df = self.group_external_ids(df)
                        df_merged = df.drop(columns=['crunchbase', 'ror', 'grid', 'lei'], errors='ignore') \
                                    .drop_duplicates(subset=['id', 'name_en']) \
                                    .merge(external_ids_df, on=['id', 'name_en'], how='left')

                        for _, row in df_merged.iterrows():
                            all_rows.extend(self.expand_organization_locations(row))

                        expanded_country_df = pd.DataFrame(all_rows)

                        expanded_country_df.to_parquet(temp_dir / f"{org_type}_{country}.parquet", index=False)

                    except Exception as e:
                        print(f"Error processing {org_type}, {country}: {e}")
                    finally:
                        pbar.update(1)

        if not all_rows:
            return pd.DataFrame()

        # Final processing
        expanded_df = pd.DataFrame(all_rows)
        expanded_df[['name', 'all_names', 'aliases']] = expanded_df.apply(self.process_lang_name_variants, axis=1)

        expanded_df = expanded_df[expanded_df.name.notnull()] \
                                .drop_duplicates(['id', 'name', 'city', 'country_name'], keep='first')

        expanded_df = expanded_df[expanded_df.country_name.isin(lang_lookup.keys())] \
                                .reset_index(drop=True)

        return expanded_df

def main():
    countries = ['Q17', 'Q33']  # USA, UK (as QIDs)
    org_types = ['Q33506', 'Q431603']  # Company type QIDs
    
    indexer = OrganisationIndex()  # Create an instance of OrganisationIndex

    organisations = indexer.get_index(countries=countries, org_types=org_types)
    
# Ensure the script is being executed directly
if __name__ == "__main__":
    main()