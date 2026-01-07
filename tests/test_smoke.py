def test_import_affilgood():
    from affilgood import AffilGood
    assert AffilGood is not None


def test_basic_affiliation_cpu():
    from affilgood import AffilGood

    ag = AffilGood(
        device="cpu",
        verbose=False  # avoid known dev-branch fragility
    )

    result = ag.process("Universitat de Barcelona, Spain")

    assert result is not None
