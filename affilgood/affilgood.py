import torch
from affilgood.ner.ner import NER
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.metadata_normalization.normalizer import GeoNormalizer

#DEFAULT_SPAN_MODEL = 'nicolauduran45/affilgood-span-v2'
#DEFAULT_NER_MODEL = 'nicolauduran45/affilgood-ner-multilingual-v2'

DEFAULT_SPAN_MODEL = 'SIRIS-Lab/affilgood-span-multilingual'
DEFAULT_NER_MODEL = 'SIRIS-Lab/affilgood-NER-multilingual'
DEFAULT_ENTITY_LINKERS = 'S2AFF'

class AffilGood:

    def __init__(self, 
                 span_separator='',  
                 span_model_path=None, 
                 ner_model_path=None, 
                 entity_linkers=None,
                 return_scores=False,
                 metadata_normalization=True, 
                 verbose=True,
                 device=None):
        
        # Verbose?
        self.verbose = verbose
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
                span_model_path = span_model_path if span_model_path else DEFAULT_SPAN_MODEL
                if self.verbose:
                    print(f'Initializing span identifier: {span_model_path}')
                self.span_identifier = SpanIdentifier(model_path=span_model_path, device=device)
        
        # Initialize NER model
        ner_model_path = ner_model_path if ner_model_path else DEFAULT_NER_MODEL
        if self.verbose:
            print(f'Initializing NER: {ner_model_path}')
        self.ner = NER(model_path=ner_model_path, device=device)
        
        # Initialize entity linker with the provided linkers
        entity_linkers = entity_linkers if entity_linkers else DEFAULT_ENTITY_LINKERS
        # Handle the case where entity_linkers is a single string or object.
        if type(entity_linkers) != list:
            entity_linkers = [entity_linkers]
        if self.verbose:
            print(f'Initializing entity linkers: {entity_linkers}')
        self.entity_linker = EntityLinker(linkers=entity_linkers, return_scores=return_scores)
        
        # Initialize normalizer
        normalizer = GeoNormalizer() if metadata_normalization else None
        if normalizer:
            if self.verbose:
                print(f'Initializing normalizer: {normalizer}')
            self.normalizer = normalizer
        else:
            print(f'Normalizer is disabled')

    # Rest of the methods remain the same
    def process(self, text):
        """Executes all steps: span identification, NER, entity linking, and normalization."""
        spans = self.get_span(text)
        entities = self.get_ner(spans)
        normalized_data = self.get_normalization(entities)
        linked_entities = self.get_entity_linking(normalized_data)
        return linked_entities

    def get_span(self, text):
        """Identifies spans within the input text."""
        return self.span_identifier.identify_spans(text)

    def get_ner(self, spans):
        """Performs named entity recognition on identified spans."""
        return self.ner.recognize_entities(spans)

    def get_entity_linking(self, entities):
        """Links recognized entities to identifiers."""
        return self.entity_linker.process_in_chunks(entities)

    def get_normalization(self, entities):
        """Normalizes the linked entity metadata."""
        return self.normalizer.normalize(entities)
        
        
