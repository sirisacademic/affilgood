# affilgood.py
from affilgood.span_identification.span_identifier import SpanIdentifier
from affilgood.ner.ner import NER
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.metadata_normalization.normalizer import GeoNormalizer
from affilgood.entity_linking.S2AFF.s2aff.ror import RORIndex
from affilgood.entity_linking.S2AFF.s2aff.model import PairwiseRORLightGBMReranker

class AffilGood:
    def __init__(self, span_model_path='nicolauduran45/affilgood-span-v2', ner_model_path='nicolauduran45/affilgood-ner-multilingual-v2', linker_model='default', metadata_normalization=True, device="cpu"):
        self.span_identifier = SpanIdentifier(model_path=span_model_path, device=device)
        self.ner = NER(model_path=ner_model_path, device=device)
        self.entity_linker = EntityLinker(method='S2AFF', device=device)
        self.normalizer = GeoNormalizer()
        self.ror_index = RORIndex('affilgood/entity_linking/v1.55-2024-10-31-ror-data.json')
        self.pairwise_model = PairwiseRORLightGBMReranker(self.ror_index, model_path = 'affilgood/entity_linking/lightgbm_model.booster',kenlm_model_path = 'affilgood/entity_linking/raw_affiliations_lowercased.binary' )

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
        return self.entity_linker.process_chunk_el(entities,self.ror_index, self.pairwise_model)

    def get_normalization(self, entities):
        """Normalizes the linked entity metadata."""
        return self.normalizer.normalize(entities)
    
    def normalize_country(self, country):
        """Normalizes the linked entity metadata."""
        return self.normalizer.normalize_country(country)