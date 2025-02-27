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
    # Get all models from configuration
    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=True)
    
    # Use CPU for testing
    device = torch.device("cpu")
    
    for model_config in models:
        # Extract model information
        model_name = model_config.pvnet.repo
        model_version = model_config.pvnet.version
        summation_name = model_config.summation.repo if model_config.summation else None
        summation_version = model_config.summation.version if model_config.summation else None
        
        # Load the models
        pvnet_model, summation_model = ForecastCompiler.load_model(
            model_name=model_name,
            model_version=model_version,
            summation_name=summation_name,
            summation_version=summation_version,
            device=device
        )
        
        # Verify the models loaded correctly
        assert isinstance(pvnet_model, PVNetBaseModel)
        
        # Check summation model if configured
        if summation_name:
            assert isinstance(summation_model, SummationBaseModel)
        else:
            assert summation_model is None
        
        # Check that essential model attributes exist
        assert hasattr(pvnet_model, "forecast_len")
        assert hasattr(pvnet_model, "output_quantiles")


def test_model_version_warning():
    """Test that warnings are raised when PVNet and summation model versions don't match."""
    # Get one model configuration that includes a summation model
    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=True)
    models_with_summation = [m for m in models if m.summation is not None]
    
    if not models_with_summation:
        pytest.skip("No models with summation available to test")
    
    model_config = models_with_summation[0]
    
    # Create a temporary subclass to force version mismatch
    class TestSummationModel(SummationBaseModel):
        @property
        def pvnet_model_name(self):
            return "different/model"
            
        @property
        def pvnet_model_version(self):
            return "different-version"
    
    # Patch the from_pretrained method to return our test model
    original_from_pretrained = SummationBaseModel.from_pretrained
    
    def mock_from_pretrained(*args, **kwargs):
        model = original_from_pretrained(*args, **kwargs)
        test_model = TestSummationModel()
        # Copy attributes from the real model to our test model
        for attr in dir(model):
            if not attr.startswith('__') and not callable(getattr(model, attr)):
                try:
                    setattr(test_model, attr, getattr(model, attr))
                except AttributeError:
                    pass
        return test_model
    
    # Apply the patch
    SummationBaseModel.from_pretrained = mock_from_pretrained
    
    try:
        # Check that warning is raised
        with pytest.warns(UserWarning) as record:
            ForecastCompiler.load_model(
                model_name=model_config.pvnet.repo,
                model_version=model_config.pvnet.version,
                summation_name=model_config.summation.repo,
                summation_version=model_config.summation.version,
                device=torch.device("cpu")
            )
        
        # Verify the warning message
        assert any("may lead to an error" in str(w.message) for w in record)
    finally:
        # Restore the original method
        SummationBaseModel.from_pretrained = original_from_pretrained