from pvnet_app.models.registry import get_model_specs


def test_get_model_specs():
    """Test for getting all model specs"""

    # Test getting all model
    models = get_model_specs()
    assert len(models) == 6
    assert models[0].name == "pvnet_intra_allbells30"

    # Test getting only critical models
    models = get_model_specs(get_critical_only=True)
    assert len(models) == 5
    assert all(m.is_critical for m in models)
