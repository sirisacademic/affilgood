def test_span_fallback():
    from affilgood.components.span import SpanIdentifier

    span = SpanIdentifier(model_path="__invalid__", verbose=False)

    items = [{"raw_text": "Universitat Autònoma de Barcelona"}]
    out = span.identify_spans(items)

    assert out[0]["span_entities"] == ["Universitat Autònoma de Barcelona"]