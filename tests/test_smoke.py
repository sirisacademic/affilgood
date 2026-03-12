def test_import_affilgood():
    from affilgood import AffilGood
    assert AffilGood is not None

def test_basic_affiliation_returns_schema():
    from affilgood import AffilGood

    ag = AffilGood()
    result = ag.process("Universitat Autònoma de Barcelona, Spain")

    assert "institutions" in result
    assert isinstance(result["institutions"], list)

def test_pipeline_never_crashes():
    from affilgood import AffilGood
    
    ag = AffilGood()
    result = ag.process("")

    assert result["input"] == ""

def test_basic_affiliation_cpu():
    from affilgood import AffilGood

    ag = AffilGood(
        device="cpu",
        verbose=False  # avoid known dev-branch fragility
    )

    result = ag.process("Universitat de Barcelona, Spain")

    assert result is not None

def test_disable_components_still_returns_output():
    from affilgood import AffilGood
    
    ag = AffilGood(
        enable_entity_linking=False,
        enable_normalization=False
    )
    result = ag.process("MIT")

    assert result["institutions"]