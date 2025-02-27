import os
import pytest
import torch
import warnings

from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from pvnet_app.forecast_compiler import ForecastCompiler
from pvnet_app.model_configs.pydantic_models import get_all_models


def test_model_loading():
    """Test that all configured models can be loaded correctly."""

    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=True)    
    device = torch.device("cpu")
    
    for model_config in models:
        # Extract model info
        model_name = model_config.pvnet.repo
        model_version = model_config.pvnet.version
        summation_name = model_config.summation.repo if model_config.summation else None
        summation_version = model_config.summation.version if model_config.summation else None
        
        # Load models via ForecastCompiler
        pvnet_model, summation_model = ForecastCompiler.load_model(
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
        
        assert hasattr(pvnet_model, "forecast_len")
        assert hasattr(pvnet_model, "output_quantiles")


def test_model_version_warning():
    """Test that warnings are raised when PVNet and summation model versions don't match."""

    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=True)
    models_with_summation = [m for m in models if m.summation is not None]
    
    if not models_with_summation:
        pytest.skip("No models with summation available to test")
    
    model_config = models_with_summation[0]
    device = torch.device("cpu")
    
    # Mock summation model - different expected PVNet version
    with patch.object(SummationBaseModel, 'pvnet_model_name', new_callable=property, return_value='different/model'), \
         patch.object(SummationBaseModel, 'pvnet_model_version', new_callable=property, return_value='different-version'):
        
        with pytest.warns(UserWarning) as record:
            pvnet_model = PVNetBaseModel.from_pretrained(
                model_id=model_config.pvnet.repo,
                revision=model_config.pvnet.version,
            ).to(device)
            
            summation_model = SummationBaseModel.from_pretrained(
                model_id=model_config.summation.repo,
                revision=model_config.summation.version,
            ).to(device)
            
            # Check if the warning is emitted when comparing models
            from pvnet_app.forecast_compiler import _model_mismatch_msg
            expected_warning = _model_mismatch_msg.format(
                model_config.pvnet.repo, 
                model_config.pvnet.version,
                'different/model', 
                'different-version'
            )
            warnings.warn(expected_warning)
        
        # Verification - warning message
        assert any("may lead to an error" in str(w.message) for w in record)
