import torch
import time
import re
from typing import List, Dict, Union, Any, Optional

DEFAULT_ENTITY_LINKERS = 'Dense'

class AffilGood:

    def __init__(self, 
                 span_separator='',  
                 span_model_path=None, 
                 ner_model_path=None, 
                 entity_linkers=None,
                 rerank=True,
                 reranker=None,
                 detailed_entity_linking_results=False,
                 initial_threshold_entity_linking=0.30,
                 final_threshold_entity_linking=0.65,
                 return_scores=False,
                 metadata_normalization=True,
                 use_country_cache=True,
                 use_osm_cache=True,
                 use_entity_linking_cache=True,
                 language_preprocessing=False,
                 llm_model_translate=None,
                 use_external_llm_translate=True,
                 verbose=False,
                 device=None,
                 batch_size=32,
                 data_sources="ror",  # ["ror", "wikidata", "plugin_source"]
                 use_wikidata_labels_with_ror=False, # Whether to entich ROR indices with previously downloaded WikiData labels for ROR records
                 wikidata_org_types='short',  # Organization types for WikiData
                 wikidata_countries=None,     # Countries for WikiData
                 data_source_configs=None):   # Dictionary of configuration for additional sources

        # Verbose?
        self.verbose = verbose
        
        # Rerank?
        self.rerank = rerank
        
        # Set reranker for combined results if provided
        self.reranker = reranker
        
        # Data sources to use for entity linking
        self.data_sources = data_sources if isinstance(data_sources, list) else [data_sources]
        
        # Data source configurations
        self.data_source_configs = data_source_configs or {}
        
        self.use_wikidata_labels_with_ror = use_wikidata_labels_with_ror
        
        # For backward compatibility, store WikiData parameters directly
        self.wikidata_org_types = wikidata_org_types
        self.wikidata_countries = wikidata_countries
        
        # Add WikiData configuration if not already present
        if "wikidata" in self.data_sources and "wikidata" not in self.data_source_configs:
            self.data_source_configs["wikidata"] = {
                "org_types": wikidata_org_types,
                "countries": wikidata_countries,
                "verbose": verbose
            }
        
        if self.verbose:
            print(f"Using data sources: {self.data_sources}")
            for source in self.data_sources:
                if source in self.data_source_configs:
                    print(f"  {source} configuration: {self.data_source_configs[source]}")
        
        # Initialize data source handlers if available
        try:
            from affilgood.entity_linking.plugins import DataSourceRegistry
            for source in self.data_sources:
                handler = DataSourceRegistry.get_handler(source)
                if handler and hasattr(handler, 'initialize'):
                    config = self.data_source_configs.get(source, {})
                    config['verbose'] = verbose  # Add verbose flag
                    handler.initialize(config)
        except ImportError:
            # Plugin system not available
            if self.verbose:
                print("Plugin system not available - using only built-in data sources")
        
        # Batch size
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize language preprocessor if enabled
        if language_preprocessing:
            from affilgood.preprocessing.llm_translator import LLMTranslator
            self.language_preprocessor = LLMTranslator(
                model_name=llm_model_translate,
                use_external_api=use_external_llm_translate,
                verbose=verbose
            )
            if self.verbose:
                print(f"Initialized LLM translator {self.language_preprocessor.model_name}")
        else:
            self.language_preprocessor = None
            if self.verbose:
                print(f'Language preprocessing is disabled')
        
        # Initialize span identifier           
        if span_separator and type(span_separator)==str and len(span_separator)==1:
            if self.verbose:
                print(f'Initializing simple span separator by character: {span_separator}')
            from affilgood.span_identification.simple_span_identifier import SimpleSpanIdentifier
            self.span_identifier = SimpleSpanIdentifier(separator=span_separator)
        else:
            if span_model_path == "noop":
                if self.verbose:
                    print(f'Span identification is disabled')
                from affilgood.span_identification.noop_span_identifier import NoopSpanIdentifier
                self.span_identifier = NoopSpanIdentifier()
            else:
                from affilgood.span_identification.span_identifier import SpanIdentifier
                if self.verbose:
                    print(f'Initializing span identifier')
                self.span_identifier = SpanIdentifier(model_path=span_model_path, device=device, batch_size=batch_size)
                if self.verbose:
                    print(f'Initialized span identifier: {self.span_identifier.model_path}')
        
        # Initialize NER model
        from affilgood.ner.ner import NER
        if self.verbose:
            print(f'Initializing NER')
        self.ner = NER(model_path=ner_model_path, device=device, batch_size=batch_size)
        if self.verbose:
            print(f'Initialized NER: {self.ner.model_path}')
        
        # Initialize entity linker with the provided linkers and data sources
        from affilgood.entity_linking.entity_linker import EntityLinker
        entity_linkers = entity_linkers if entity_linkers else DEFAULT_ENTITY_LINKERS
        # Handle the case where entity_linkers is a single string or object.
        if type(entity_linkers) != list:
            entity_linkers = [entity_linkers]
        if self.verbose:
            print(f'Initializing entity linkers: {entity_linkers} for data sources: {self.data_sources}')
        self.entity_linker = EntityLinker(
                                linkers=entity_linkers,
                                rerank=self.rerank,
                                reranker=self.reranker,
                                detailed_results=detailed_entity_linking_results,
                                linker_threshold=initial_threshold_entity_linking,
                                final_threshold=final_threshold_entity_linking,
                                verbose=self.verbose,
                                use_cache=use_entity_linking_cache,
                                return_scores=return_scores,
                                batch_size=batch_size,
                                data_sources=self.data_sources,
                                use_wikidata_labels_with_ror=self.use_wikidata_labels_with_ror,
                                wikidata_org_types=self.wikidata_org_types,
                                wikidata_countries=self.wikidata_countries,
                                data_source_configs=self.data_source_configs)
        
        # Initialize normalizer
        from affilgood.metadata_normalization.normalizer import GeoNormalizer
        normalizer = GeoNormalizer(
            use_country_cache=use_country_cache, 
            use_osm_cache=use_osm_cache
        ) if metadata_normalization else None
        if normalizer:
            if self.verbose:
                print(f'Initializing normalizer: {normalizer}')
            self.normalizer = normalizer
        else:
            print(f'Normalizer is disabled')

    def process(self, text_or_texts, batch_size=None):
        """
        Main public entry point. Accepts either a single string or a list of strings,
        and returns standardized results with entity_linking results from all configured data sources.
        """
        is_single_input = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single_input else text_or_texts
        results = self.batch_process(texts, batch_size=batch_size)
        return results[0] if is_single_input else results

    def batch_process(self, texts, batch_size=32):
        """
        Process a batch of affiliation strings efficiently using batch processing
        at each stage of the pipeline.
        
        Args:
            texts: List of affiliation strings to process
            batch_size: Size of batches for processing
            
        Returns:
            List of processed entities for each input text
        """
        if not texts:
            return []

        batch_size = batch_size if batch_size is not None else self.batch_size

        start_time = time.time() if self.verbose else None
        
        # 1. Language preprocessing if enabled
        if self.language_preprocessor:
            if self.verbose:
                print(f"Batch language preprocessing {len(texts)} texts...")
            preprocessing_results = self.language_preprocessor.process_batch(texts, batch_size=batch_size)
            processed_texts = [result["processed_text"] for result in preprocessing_results]
        else:
            preprocessing_results = [{}] * len(texts)  # Empty info if no preprocessor
            processed_texts = texts
        
        if self.verbose and start_time:
            elapsed = time.time() - start_time
            print(f"Language preprocessing completed in {elapsed:.2f}s")
            start_time = time.time()
        
        # 2. Span identification - process all texts in one batch
        if self.verbose:
            print(f"Identifying spans for {len(processed_texts)} texts...")
        
        spans = self.span_identifier.identify_spans(processed_texts, batch_size=batch_size)
        
        if self.verbose and start_time:
            elapsed = time.time() - start_time
            print(f"Span identification completed in {elapsed:.2f}s")
            print(f"Identified {sum(len(item['span_entities']) for item in spans)} spans")
            start_time = time.time()
        
        # 3. Named Entity Recognition - process all spans in one batch
        if self.verbose:
            print(f"Recognizing entities...")
        
        entities = self.ner.recognize_entities(spans, batch_size=batch_size)
        
        if self.verbose and start_time:
            elapsed = time.time() - start_time
            print(f"Entity recognition completed in {elapsed:.2f}s")
            start_time = time.time()
        
        # 4. Entity normalization
        if self.verbose:
            print(f"Normalizing entities...")
        
        normalized_data = self.normalizer.normalize(entities)
        
        if self.verbose and start_time:
            elapsed = time.time() - start_time
            print(f"Entity normalization completed in {elapsed:.2f}s")
            start_time = time.time()
        
        # 5. Entity linking
        if self.verbose:
            print(f"Linking entities...")
        
        results = self.entity_linker.process_in_chunks(normalized_data)
        
        if self.verbose and start_time:
            elapsed = time.time() - start_time
            print(f"Entity linking completed in {elapsed:.2f}s")
        
        # 6. Add language detection/translation metadata
        for i, result in enumerate(results):
            if i < len(preprocessing_results):
                result["language_info"] = preprocessing_results[i]
        
        return results

    def get_span(self, text):
        """Identifies spans within the input text."""
        return self.span_identifier.identify_spans(text)

    def get_ner(self, spans):
        """Performs named entity recognition on identified spans."""
        return self.ner.recognize_entities(spans)

    def get_entity_linking(self, entities):
        """Links recognized entities to identifiers."""
        results = self.entity_linker.process_in_chunks(entities)
        # Process the internal structure to ensure the linked_organizations field is properly populated
        for item in results:
            if 'ror' in item and not 'linked_orgs' in item:
                # Build the linked_orgs structure if not already present
                # This might happen if there was caching or other shortcuts in the pipeline
                item['linked_orgs'] = []
        return results
        
    def get_normalization(self, entities):
        """Normalizes the linked entity metadata."""
        return self.normalizer.normalize(entities)
    
    def get_language_stats(self):
        """Returns statistics from the language preprocessor if enabled."""
        if self.language_preprocessor:
            stats = self.language_preprocessor.get_stats()
            
            # Calculate additional metrics if processing has occurred
            if stats["processed"] > 0:
                stats["avg_processing_time"] = stats["total_processing_time"] / stats["processed"]
                
                if stats["translations_performed"] > 0:
                    stats["avg_translation_time"] = stats["total_translation_time"] / stats["translations_performed"]
                else:
                    stats["avg_translation_time"] = None
            
            return stats
        else:
            return {"error": "Language preprocessor is not enabled"}
