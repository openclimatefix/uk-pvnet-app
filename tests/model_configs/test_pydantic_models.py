""" Test for getting all ml models"""
from pvnet_app.model_configs.pydantic_models import get_all_models


def test_get_all_models():
    """Test for getting all models"""
    models = get_all_models()
    assert len(models) == 1
    assert models[0].name == "pvnet_v2"


def test_get_all_models_get_ecmwf_only():
    """Test for getting all models with ecmwf_only"""
    models = get_all_models(get_ecmwf_only=True)
    assert len(models) == 1
    assert models[0].ecmwf_only


def test_get_all_models_get_day_ahead_only():
    """Test for getting all models with ecmwf_only"""
    models = get_all_models(get_day_ahead_only=True, use_ocf_data_sampler=False)
    assert len(models) == 1
    assert models[0].day_ahead


def test_get_all_models_run_extra_models():
    """Test for getting all models with ecmwf_only"""
    models = get_all_models(run_extra_models=True)
    assert len(models) == 5


def test_get_all_models_ocf_data_sampler():
    """Test for getting all models with ecmwf_only"""
    models = get_all_models(use_ocf_data_sampler=True, run_extra_models=True)
    assert len(models) == 5

    models = get_all_models(use_ocf_data_sampler=False, run_extra_models=True)
    assert len(models) == 2

    models = get_all_models(
        use_ocf_data_sampler=False, 
        run_extra_models=True, 
        get_day_ahead_only=True
    )
    assert len(models) == 1

