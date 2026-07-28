from pvnet_app.models.registry import get_model_specs


def test_get_model_specs():
    """Test for getting all model specs"""

    # Test getting all model
    models = get_model_specs()
    assert len(models) == 1
    assert models[0].name == "pvnet_day_ahead"

    # Test getting only critical models
    models = get_model_specs(get_critical_only=True)
    assert len(models) == 1
    assert all(m.is_critical for m in models)
