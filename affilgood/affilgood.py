# affilgood.py
from affilgood.span_identification.span_identifier import SpanIdentifier
from affilgood.ner.ner import NER
from affilgood.entity_linking.entity_linker import EntityLinker
from affilgood.metadata_normalization.normalizer import Normalizer

class AffilGood:
    def __init__(self, span_model_path='nicolauduran45/affilgood-span-v2', ner_model_path='nicolauduran45/affilgood-ner-multilingual-v2', linker_model='default', metadata_normalization=True, device="cpu"):
        self.span_identifier = SpanIdentifier(model_path=span_model_path, device=device)
        #self.ner = NER(model_path=ner_model_path, device=device)
        #self.entity_linker = EntityLinker(model=linker_model, device=device)
        #self.normalizer = Normalizer()

    def process(self, text):
        """Executes all steps: span identification, NER, entity linking, and normalization."""
        spans = self.get_span(text)
        entities = self.get_ner(spans)
        #linked_entities = self.get_entity_linking(entities)
        #normalized_data = self.get_normalization(linked_entities)
        return entities

    def get_span(self, text):
        """Identifies spans within the input text."""
        return self.span_identifier.identify_spans(text)

    def get_ner(self, spans):
        """Performs named entity recognition on identified spans."""
        return self.ner.recognize_entities(spans)

    def get_entity_linking(self, entities):
        """Links recognized entities to identifiers."""
        return self.entity_linker.link_entities(entities)

    def get_normalization(self, linked_entities):
        """Normalizes the linked entity metadata."""
        return self.normalizer.normalize_metadata(linked_entities)