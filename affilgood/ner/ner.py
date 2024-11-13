# ner.py
from transformers import pipeline

class NER:
    def __init__(self, model_path, device="cpu"):
        self.model = pipeline("ner", model=model_path, device=device)

    def recognize_entities(self, span_text):
        # Use model to perform NER on identified spans
        entities = self.model(span_text)
        return entities