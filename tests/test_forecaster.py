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
        pvnet_repo = model_config.pvnet.repo
        pvnet_commit = model_config.pvnet.commit
        summation_repo = model_config.summation.repo if model_config.summation else None
        summation_commit = model_config.summation.commit if model_config.summation else None
        
        # Load models via Forecaster
        pvnet_model, summation_model = Forecaster.load_model(
            pvnet_repo=pvnet_repo,
            pvnet_commit=pvnet_commit,
            summation_repo=summation_repo,
            summation_commit=summation_commit,
            device=device
        )
        
        # Verify models loaded correctly
        assert isinstance(pvnet_model, PVNetBaseModel)
        
        # Verify summation model if configured
        if summation_repo is not None:
            assert isinstance(summation_model, SummationBaseModel)
        else:
            assert summation_model is None