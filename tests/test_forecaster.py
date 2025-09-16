import torch

from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from pvnet_app.forecaster import Forecaster
from pvnet_app.model_configs.pydantic_models import get_all_models


def test_model_loading():
    """Test that all configured models can be loaded correctly."""

    models = get_all_models(get_critical_only=False)    
    device = torch.device("cpu")
    
    for model_config in models:
        # Extract model info
        model_name = model_config.pvnet.repo
        model_version = model_config.pvnet.commit
        summation_name = model_config.summation.repo if model_config.summation else None
        summation_version = model_config.summation.commit if model_config.summation else None
        
        # Load models via Forecaster
        pvnet_model, summation_model = Forecaster.load_model(
            model_name=model_name,
            model_version=model_version,
            summation_name=summation_name,
            summation_version=summation_version,
            device=device
        )
        
        # Verify models loaded correctly
        assert isinstance(pvnet_model, PVNetBaseModel)
        
        # Verify summation model if configured
        if summation_name:
            assert isinstance(summation_model, SummationBaseModel)
        else:
            assert summation_model is None