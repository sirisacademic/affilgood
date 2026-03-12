from affilgood.components.ner import NER


def test_ner_importable():
    ner = NER(model_path=None)
    assert ner is not None

from affilgood.components import NER

class DummyPipeline:
    def __call__(self, dataset, batch_size=32):
        # One output per input span
        return [
            [
                {
                    "entity_group": "ORG",
                    "word": "Universitat Autònoma de Barcelona",
                    "start": 0,
                    "end": 33,
                    "score": 0.99,
                }
            ]
        ]


def test_ner_recognize_entities_with_text():
    ner = NER(model_path=None)
    ner._pipeline = DummyPipeline()
    ner._available = True

    items = [
        {
            "span_entities": [
                "Universitat Autònoma de Barcelona, Spain"
            ]
        }
    ]

    out = ner.recognize_entities(items)

    assert len(out) == 1
    assert "ner" in out[0]
    assert "ner_raw" in out[0]

    # One span → one ner entry
    assert len(out[0]["ner"]) == 1

    ner_result = out[0]["ner"][0]

    assert "ORG" in ner_result
    assert ner_result["ORG"] == ["Universitat Autònoma de Barcelona"]
