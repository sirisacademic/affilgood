# normalizer.py

class Normalizer:
    def __init__(self, normalization_rules):
        # Load normalization rules directly as a parameter
        self.rules = normalization_rules

    def normalize_metadata(self, linked_entities):
        # Apply normalization rules to linked entities
        normalized_data = [self.apply_rules(entity) for entity in linked_entities]
        return normalized_data

    def apply_rules(self, entity):
        # Normalize an individual entity based on rules
        return entity  # Placeholder