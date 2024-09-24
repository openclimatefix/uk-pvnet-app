""" Test for getting all ml models"""
from pvnet_app.model_configs.pydantic_models import get_all_models


def test_get_all_models():
    """Test for getting all models"""
    models = get_all_models()
    assert len(models.models) == 4


