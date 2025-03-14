import os
from whoosh import index
from whoosh.qparser import MultifieldParser, QueryParser, OrGroup, AndGroup
from whoosh.query import Term, Or, And, FuzzyTerm
from whoosh.analysis import StemmingAnalyzer
from .base_linker import BaseLinker
from .utils.text_utils import (
    get_variants_list, get_variants_country,
    get_stopwords, get_legal_entities, load_abbreviations
)
from .utils.translation_mappings import translate_institution_name
from .llm_reranker import LLMReranker
from unidecode import unidecode

class WhooshLinker(BaseLinker):
    """Whoosh-based entity linker."""
    
    def __init__(
                self,
                data_manager=None,
                index_dir=None,
                rebuild_index=False,
                threshold_score=0.25,
                rerank=True,
                rerank_model_name=None,
                number_candidates_rerank=5,
                max_hits=10,
                debug=False
            ):
        super().__init__()
        self.data_manager = data_manager
        self.threshold_score = threshold_score
        self.max_hits = max_hits
        self.debug = debug
        self.index_dir = index_dir
        self.rebuild_index = rebuild_index
        # Number of candidate results to retrieve from the index for each grouped org/suborg.
        self.number_candidates_rerank = number_candidates_rerank
        self.rerank = rerank
        self.reranker = LLMReranker(model_name=rerank_model_name) if rerank else None
        # Initialization is deferred until we have a data_manager
        self.is_initialized = False
        # If data_manager is provided, initialize now
        if self.data_manager is not None:
            self.initialize()
    
    def initialize(self):
        """Initialize the Whoosh linker components."""
        if self.is_initialized:
            return
            
        # Prepare the Whoosh index
        self.index_dir = self._prepare_index(self.index_dir, self.rebuild_index)
        
        # Set up the Whoosh components
        self.analyzer = StemmingAnalyzer()
        self.stopwords = get_stopwords('u')
        self.legal_entities = get_legal_entities()
        self.abbreviations = load_abbreviations()
        
        # Set up the Whoosh index
        self.ix = index.open_dir(self.index_dir)
        self.name_parser = MultifieldParser(["ror_name", "ror_name_normalized"], schema=self.ix.schema, group=AndGroup)
        self.name_parser_flex = MultifieldParser(["ror_name", "ror_name_normalized"], schema=self.ix.schema, group=OrGroup)
        self.alias_parser = QueryParser("aliases_text", schema=self.ix.schema, group=AndGroup)
        self.alias_parser_flex = QueryParser("aliases_text", schema=self.ix.schema, group=OrGroup)
        
        # Consider types of matches for initial re-ranking.
        self.strong_keywords_set = {"name", "alias"}
        self.location_keywords_set = {"city", "region", "country"}
        
        self.is_initialized = True
 
    def _prepare_index(self, index_dir, rebuild_index=False):
        """
        Prepare the Whoosh index directory, creating or rebuilding it if necessary.
        
        Args:
            index_dir (str): Path to the Whoosh index directory
            rebuild_index (bool): Whether to rebuild the index even if it exists
            
        Returns:
            str: Path to the prepared index directory
        """
        # Default index directory if none provided
        if not index_dir:
            index_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'whoosh_index')
        
        # Create the directory if it doesn't exist
        if not os.path.exists(index_dir):
            os.makedirs(index_dir, exist_ok=True)
            print(f"Created Whoosh index directory at {index_dir}")
            rebuild_index = True  # Force index creation for new directory
        
        # Check if the index needs to be built or rebuilt
        index_exists = os.path.exists(index_dir) and os.listdir(index_dir)
        
        if rebuild_index or not index_exists:
            if not self.data_manager:
                raise ValueError("Cannot build Whoosh index: data_manager is not set")
                
            print(f"{'Rebuilding' if index_exists else 'Building new'} Whoosh index in {index_dir}")
            success = self.data_manager.create_whoosh_index(index_dir)
            
            if not success:
                raise RuntimeError(f"Failed to {'rebuild' if index_exists else 'build'} Whoosh index in {index_dir}")
        
        return index_dir
        
    def _clean_organization_name(self, name):
        """Normalize organization name by removing legal entities, stopwords, and applying transliteration."""
        if not name:
            return ""
        name = name.replace("'", " ")
        name = name.replace('"', " ")
        words = name.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords and word not in self.legal_entities]
        return " ".join(filtered_words)
    
    def _refine_names_alias_matches(self, results, search_text):
        """Refine search results by checking for overlaps in both names and aliases."""
        search_tokens = set(search_text.lower().split())
        search_text_lower = search_text.lower()
        
        refined_results = []
        
        for hit in results:
            name = hit.get("ror_name", "")
            name_lower = name.lower()
            aliases_list = hit.get("aliases_list", [])
            
            # Add the name itself to the potential matches list
            all_potential_matches = [name] + aliases_list
            
            best_match_score = 0
            best_match_text = None
            best_match_type = None
            
            # Check each potential match (name + aliases)
            for potential_match in all_potential_matches:
                if not potential_match:
                    continue
                    
                potential_match_lower = potential_match.lower()
                
                # 1. Exact match (highest priority)
                if potential_match_lower == search_text_lower:
                    match_score = 1.0
                    best_match_score = match_score
                    best_match_text = potential_match
                    best_match_type = "name" if potential_match == name else "alias"
                    break
                    
                # 2. One is substring of the other
                elif potential_match_lower in search_text_lower or search_text_lower in potential_match_lower:
                    # Calculate how much of the shorter string is contained in the longer one
                    shorter = min(len(potential_match_lower), len(search_text_lower))
                    longer = max(len(potential_match_lower), len(search_text_lower))
                    containment_score = shorter / longer * 0.9  # 90% of exact match
                    
                    if containment_score > best_match_score:
                        best_match_score = containment_score
                        best_match_text = potential_match
                        best_match_type = "name" if potential_match == name else "alias"
                
                # 3. Token overlap
                else:
                    match_tokens = set(potential_match_lower.split())
                    common_tokens = search_tokens.intersection(match_tokens)
                    
                    if common_tokens:
                        # Calculate Jaccard similarity: intersection / union
                        jaccard = len(common_tokens) / len(search_tokens.union(match_tokens))
                        overlap_score = jaccard * 0.8  # 80% of exact match
                        
                        if overlap_score > best_match_score:
                            best_match_score = overlap_score
                            best_match_text = potential_match
                            best_match_type = "name" if potential_match == name else "alias"
            
            # Apply score boost based on match quality
            if best_match_score > 0:
                # Scale the boost based on match quality (1.0 to 2.0)
                # Give a slight additional boost to name matches vs alias matches
                name_boost = 0.1 if best_match_type == "name" else 0
                boost_factor = 1.0 + best_match_score + name_boost
                hit.score *= boost_factor
                
                match_type = "EXACT" if best_match_score == 1.0 else f"PARTIAL ({best_match_score:.2f})"
                if self.debug:
                    print(f"=> {name} - {match_type} {best_match_type}: '{best_match_text}' - Score boosted by {boost_factor:.2f}x")
            else:
                if self.debug:
                    print(f"=> {name} - NO MATCH: {name} or {aliases_list}")
            
            refined_results.append(hit)
        
        # Sort by score descending
        return sorted(refined_results, key=lambda x: x.score, reverse=True)
       
    def _adjust_scores_based_on_match_type(self, top_entities):
        """Optimizes score adjustments by checking name and location keyword matches."""
        for candidate, info in top_entities.items():
            # Convert descriptions to a set for faster lookup
            description_set = {desc.lower() for desc in info["matches"]}

            # Check for strong name match
            if any(kw in description_set for kw in self.strong_keywords_set):
                continue  # No need to penalize if strong name match exists

            # Check for location match
            if any(kw in description_set for kw in self.location_keywords_set):
                info["score"] *= 0.5  # Apply penalty for location-only match

        return top_entities
    
    def get_single_prediction(self, organization):
        affiliation = []
        suborg_ner = organization.get('suborg', '')
        org_ner = organization['main']
        location_ner = organization['location']
        if suborg_ner:
            affiliation.append(suborg_ner)
        affiliation.append(org_ner)
        if location_ner:
            affiliation.append(location_ner)
        affiliation_string_ner = ', '.join(affiliation)
        if self.debug:
            print(f"Getting prediction for {affiliation_string_ner}")
        predicted_id, predicted_name, predicted_score = self._get_from_cache(affiliation_string_ner)
        if predicted_id is not None:
            return predicted_id, predicted_name, predicted_score
        candidate_matches = self.get_candidate_matches(organization)
        if len(candidate_matches) > 1:
            if self.rerank:
                candidate_orgs = []
                for candidate_match in candidate_matches[:self.number_candidates_rerank]:
                    ror_id = candidate_match.get('ror_id', '')
                    org_name = candidate_match.get('ror_name', '')
                    name_location = []
                    if ror_id and org_name:
                        name_location.append(org_name)
                        city = candidate_match.get('city', '')
                        country = candidate_match.get('country_name', candidate_match.get('country', ''))
                        if city:
                            name_location.append(city)
                        if country:
                            name_location.append(country)
                        org_name_location = ', '.join(name_location)
                        candidate_orgs.append(f'{org_name_location} ({ror_id})')
                # Re-rank organizations.
                if len(candidate_orgs) > 1:
                    reranked_id = self.reranker.rerank(affiliation=affiliation_string_ner, candidates=candidate_orgs)
                    matched_org = next((org for org in candidate_matches if type(reranked_id)==str and reranked_id in org.get("ror_id")), None)
                    if matched_org:
                        predicted_name = matched_org.get("ror_name", reranked_id)
                        predicted_id = reranked_id
                        predicted_score = getattr(matched_org, "score", 0)
                    else:
                        predicted_name = None
                        predicted_id = None
                        predicted_score = 0
                    return predicted_id, predicted_name, predicted_score
            else:
                first_result = candidate_matches[0]
                predicted_id = first_result.get('ror_id', '')
                predicted_name = first_result.get('ror_name', '')
                predicted_score =  getattr(first_result, 'score', 0)
        else:
            predicted_id = None
            predicted_name = None
            predicted_score = 0
        self._update_cache(affiliation_string_ner, predicted_id, predicted_name, predicted_score)
        return predicted_id, predicted_name, predicted_score

    def get_candidate_matches(self, organization):
        """Perform search using name, aliases, parent, and location."""

        org_name = organization["main"]
        suborg = organization.get("suborg", "")
        city = organization["city"]
        region = organization["region"]
        country = organization["country"]

        if not org_name:
            return []

        name_variants = get_variants_list([org_name])
        name_variants.extend(translate_institution_name(org_name))
        clean_name_variants = {self._clean_organization_name(name_variant) for name_variant in name_variants if name_variant.strip()}
        clean_name_variants = {variant for variant in clean_name_variants if variant.strip()}
        
        # Get acronym variants
        acronym_variants = self.abbreviations.get(org_name.lower(), [])
        clean_name_variants.union(set(acronym_variants))
        
        if suborg:
            suborg_variants = get_variants_list([suborg])
            suborg_variants.extend(translate_institution_name(suborg))
            clean_suborg_variants = {self._clean_organization_name(suborg_variant) for suborg_variant in suborg_variants}
            clean_suborg_variants = {variant for variant in clean_suborg_variants if variant.strip()}
            clean_name_variants = {f"{suborg}, {org}" for org in clean_name_variants for suborg in clean_suborg_variants}
            
        if suborg:
            suborg_variants = get_variants_list([suborg])
            suborg_variants.extend(translate_institution_name(suborg))
            clean_suborg_variants = {self._clean_organization_name(suborg_variant) for suborg_variant in suborg_variants if suborg_variant.strip()}
            clean_suborg_variants = {variant for variant in clean_suborg_variants if variant.strip()}
            combined_variants = {f"{suborg}, {org}" for org in clean_name_variants for suborg in clean_suborg_variants 
                                 if org.strip() and suborg.strip()}
            clean_name_variants = combined_variants if combined_variants else clean_name_variants
        
        # Get country variants.
        country_variants = set()
        if country:
            country_variants.union(set(get_variants_country(country)))
        
        with self.ix.searcher() as searcher:
            # Build our queries
            name_queries = []
            location_queries = []
            # Store descriptions for debugging
            query_descriptions = []  
            location_descriptions = ["Location"]
            
            # Name variant queries with proper boosting
            for name_variant in clean_name_variants:
                # Add exact phrase query for multi-word terms (highest priority)
                if len(name_variant.split()) > 1:
                    exact_query = self.name_parser.parse(f'"{name_variant}"')  # Quote for exact phrase
                    exact_query = exact_query.with_boost(3.0)  # Triple weight for exact match
                    name_queries.append(exact_query)
                    query_descriptions.append(f"Exact name match: '{name_variant}'. Query: {exact_query}")

                # Regular term query (medium priority)
                name_query = self.name_parser.parse(name_variant)
                name_query = name_query.with_boost(1.5)  # 50% more weight
                name_queries.append(name_query)
                query_descriptions.append(f"Name match: '{name_variant}'. Query: {name_query}")
                
                # Alias query (medium priority) - now use aliases_text field
                alias_query = self.alias_parser.parse(name_variant)
                alias_query = alias_query.with_boost(1.3)  # 30% more weight
                name_queries.append(alias_query)
                query_descriptions.append(f"Alias match: '{name_variant}'. Query: {alias_query}")

                # Regular term query (lower priority - no boost)
                name_flex_query = self.name_parser_flex.parse(name_variant)
                name_queries.append(name_flex_query)
                query_descriptions.append(f"Name match (flex): '{name_variant}'. Query: {name_flex_query}")
                
                # Alias query (lower priority - no boost) - now use aliases_text field
                alias_flex_query = self.alias_parser_flex.parse(name_variant)
                name_queries.append(alias_flex_query)
                query_descriptions.append(f"Alias match (flex): '{name_variant}'. Query: {alias_flex_query}")
                
                # Fuzzy query (lowest priority - no boost)
                # Only add fuzzy query if the term is at least 3 characters long
                if len(name_variant) >= 10:
                    fuzzy_query = FuzzyTerm("ror_name", name_variant, maxdist=2, boost=1.0)
                    name_queries.append(fuzzy_query)
                    query_descriptions.append(f"Fuzzy name match: '{name_variant}'. Query: {fuzzy_query}")

                """
                # Suborg/Parent relationship
                suborg_query = self.name_parser.parse(suborg)
                parent_query = Term("parent", org_name.lower())
                parent_or_query = Or([suborg_query, parent_query], boost=1.5)  # Add boost
                queries.append(parent_or_query)
                query_descriptions.append(f"Suborg/Parent relationship: '{suborg}' in '{org_name}'. Query: {parent_or_query}")
                """

            # Location data.
            if city:
                city_query = Term("city", city.lower(), boost=1.5)  # Higher boost for city
                location_queries.append(city_query)
                location_descriptions.append(f"\n   City: '{city}'. Query: {city_query}")
            
            if region:
                region_query = Term("region", region.lower(), boost=1)
                location_queries.append(region_query)
                location_descriptions.append(f"\n   Region: '{region}'. Query: {region_query}")
                
            if country:
                country_query = Term("country_name", country.lower(), boost=1)
                location_queries.append(country_query)
                location_descriptions.append(f"\n   Country: '{country}'. Query: {country_query}")
            
            # Add country queries from variants
            for country in country_variants:
                if country.lower() != location.get('country', '').lower():
                    country_variant_query = Term("country_name", country.lower(), boost=1)
                    location_queries.append(country_variant_query)
                    location_descriptions.append(f"\n   Country Variant: '{country}'. Query: {country_variant_query}")
                
            query_descriptions.append(f"Location: {', '.join(location_descriptions)}")
           
            # Combine name queries (OR).
            if name_queries:
                combined_name_query = Or(name_queries)
            else:
                combined_name_query = None

            # Combine location queries (OR).
            if location_queries:
                combined_location_query = Or(location_queries)
            else:
                combined_location_query = None
            
            # Combine queries.
            if combined_name_query and combined_location_query:
                final_query = And([combined_name_query, combined_location_query])
            elif combined_name_query:
                final_query = combined_name_query
            else:
                final_query = None

            if final_query:

                # After combining queries and running the final search:
                all_results = searcher.search(final_query, limit=self.max_hits)
                
                if all_results:
                    all_results = self._refine_names_alias_matches(all_results, org_name)

                if not all_results:
                    return []

                # Consider top-10 initially.
                top_entities = {
                    f"{hit.get('ror_name', '')} ({hit.get('ror_id', '')})": {
                        "score": getattr(hit, "score", 0),
                        "matches": hit.get("matches", {})
                    }
                    for hit in all_results[:10]
                }

                self._adjust_scores_based_on_match_type(top_entities)

                # Filter and format results.
                filtered_results = [hit for hit in all_results if getattr(hit, "score", 0) >= self.threshold_score]
                filtered_results.sort(key=lambda hit: hit.score, reverse=True)

                ror_seen = set()
                unique_results = []
                for hit in filtered_results:
                    ror_id = hit.get('ror_id', '')
                    if ror_id and ror_id not in ror_seen:
                        ror_seen.add(ror_id)
                        unique_results.append(hit)
                return unique_results

            else:
                if self.debug:
                    print("\nNo query could be generated from the parsed data.")
                return []
            
            return []
