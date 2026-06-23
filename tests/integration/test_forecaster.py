import pandas as pd
import torch
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from pvnet_app.forecaster import Forecaster
from pvnet_app.models.pydantic_models import get_all_models


def test_model_loading():
    """Test that all configured models can be loaded correctly."""

    models = get_all_models(get_critical_only=False)
    device = torch.device("cpu")

    for model_config in models:

        forecaster = Forecaster(
            model_config=model_config,
            data_config_path="dummy.yaml",
            t0=pd.Timestamp.now(),
            gsp_ids=[*range(10)],
            device=device,
            gsp_capacities=None,
            national_capacity=1,
        )

        # Verify models loaded correctly
        assert isinstance(forecaster.model, PVNetBaseModel)

        if model_config.summation.repo is None:
            assert forecaster.summation_model is None
        else:
            assert isinstance(forecaster.summation_model, SummationBaseModel)
