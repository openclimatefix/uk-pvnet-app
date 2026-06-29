import pandas as pd
import torch
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from pvnet_app.forecaster import PVNetForecaster
from pvnet_app.models.registry import get_model_specs


def test_model_loading():
    """Test that all configured models can be loaded correctly."""

    models = get_model_specs(get_critical_only=False)
    device = torch.device("cpu")

    for model_spec in models:
        forecaster = PVNetForecaster(
            model_spec=model_spec,
            data_config_path="dummy.yaml",
            run_data_dir="dummy_dir",
            t0=pd.Timestamp.now(),
            device=device,
            capacities={},
        )

        # Verify models loaded correctly
        assert isinstance(forecaster.model, PVNetBaseModel)

        if model_spec.summation.repo is None:
            assert forecaster.summation_model is None
        else:
            assert isinstance(forecaster.summation_model, SummationBaseModel)
