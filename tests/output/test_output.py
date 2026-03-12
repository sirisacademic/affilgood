from affilgood.output import normalize_output


def test_normalize_output_empty():
    out = normalize_output({})
    assert out["input"] == ""
    assert out["institutions"] == []
    assert out["subunits"] == []
    assert out["location"] is None
    assert out["language"] is None
    assert out["confidence"] is None


def test_normalize_output_from_ner():
    raw = {
        "raw_text": "MIT",
        "ner": [{"ORG": ["MIT"]}],
    }

    out = normalize_output(raw)

    assert len(out["institutions"]) == 1
    assert out["institutions"][0]["name"] == "MIT"
    assert out["institutions"][0]["source"] == "ner"


def test_normalize_output_multiple_spans():
    raw = {
        "raw_text": "Dept A, University B",
        "ner": [
            {"SUB": ["Dept A"]},
            {"ORG": ["University B"]},
        ],
    }

    out = normalize_output(raw)

    assert len(out["subunits"]) == 1
    assert len(out["institutions"]) == 1
    assert out["institutions"][0]["name"] == "University B"