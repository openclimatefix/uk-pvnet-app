from pvnet_app.model_configs.pydantic_models import get_all_models


def test_get_all_models():
    """Test for getting all models"""
    models = get_all_models()
    assert len(models) == 6
    assert models[0].name == "pvnet_v2"


def test_get_all_models_get_critical_only():
    """Test for getting all the critcal models"""
    models = get_all_models(get_critical_only=True)
    assert len(models) == 2
    assert all(m.is_critical for m in models)


def test_get_all_models_get_day_ahead_only():
    """Test for getting all the day ahead models"""
    models = get_all_models(get_day_ahead_only=True)
    assert len(models) == 1
    assert models[0].is_day_ahead

