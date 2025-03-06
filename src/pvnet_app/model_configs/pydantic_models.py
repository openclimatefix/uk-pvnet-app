"""A pydantic model for the ML models"""
import logging
from importlib.resources import files

import fsspec
from pyaml_env import parse_config
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


class HuggingFaceCommit(BaseModel):
    """The location of a model on Hugging Face"""
    repo: str = Field(..., description="The Hugging Face repo")
    commit: str = Field(..., description="The commit hash")


class ModelConfig(BaseModel):
    """Configuration of a model and the settings it will be run with in the app"""

    name: str = Field(..., description="The name of the model")
    pvnet: HuggingFaceCommit = Field(..., description="The PVNet model location")
    summation: HuggingFaceCommit = Field(..., description="The summation model location")

    use_adjuster: bool = Field(False, description="Whether to use the adjuster")
    save_gsp_sum: bool = Field(
        False, 
        description="Whether to save the sum of GSPs as welll as the national estimate"
    )
    verbose: bool = Field(False, description="Whether to log verbose output for the model")
    save_gsp_to_recent: bool = Field(
        False, 
        description="Whether to save the GSP results to the `ForecastValueLastSevenDays` table",
    )
    day_ahead: bool = Field(False, description="If this model is for day-ahead forecasts or not")
    ecmwf_only: bool = Field(False, description="If this model is only using ecmwf data")
    uses_satellite_data: bool = Field(
        True, 
        description="If this model uses satellite data (currently this is only used in tests)"
    )
    uses_ocf_data_sampler: bool = Field(
        True,
        description="If this model uses ocf-data-sampler. Else uses ocf_datapipes",
    )


class ModelConfigCollection(BaseModel):
    """A collection of model configurations"""

    models: list[ModelConfig] = Field(
        ..., 
        description="A list of model configs to use for the forecast"
    )

    @field_validator("models")
    @classmethod
    def name_must_be_unique(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        """Ensure that all model names are unique, respect to using ocf_data_sampler or not"""
        names = [(model.name, model.uses_ocf_data_sampler) for model in v]

        if len(names) != len(set(names)):
            raise Exception(f"Model names must be unique, names are {names}")
        return v


def get_all_models(
    allow_use_adjuster: bool = True,
    allow_save_gsp_sum: bool = True,
    get_ecmwf_only: bool = False,
    get_day_ahead_only: bool = False,
    run_extra_models: bool = False,
    use_ocf_data_sampler: bool = True,
) -> list[ModelConfig]:
    """Returns all the models for a given client

    Args:
        allow_use_adjuster: If set to false all models will have use_adjuster set to false
        allow_save_gsp_sum: If set to false all models will have save_gsp_sum set to false
        get_ecmwf_only: If only the ECMWF model should be returned
        get_day_ahead_only: If only the day-ahead model should be returned
        run_extra_models: If all extra models should be returned
        use_ocf_data_sampler: If the ocf-data-sampler models should be returned
    """
    
    filename = files("pvnet_app.model_configs").joinpath("all_models.yaml")

    with fsspec.open(filename, mode="r") as stream:
        try:
            models_dict = parse_config(data=stream)
            model_collection = ModelConfigCollection(**models_dict)
        except Exception as config_error:
            log.error(f"Error parsing model configuration: {config_error}")
            raise config_error

    # Override the use_adjuster and save_gsp_sum properties
    if not allow_use_adjuster:
        for model in model_collection.models:
            model.use_adjuster = False
    if not allow_save_gsp_sum:
        for model in model_collection.models:
            model.save_gsp_sum = False

    # Filter models
    filtered_models = model_collection.models.copy()

    if get_ecmwf_only:
        log.info("Filtering for ECMWF model only")
        filtered_models = [model for model in filtered_models if model.ecmwf_only]

    if get_day_ahead_only:
        log.info("Filtering for Day Ahead model")
        filtered_models = [model for model in filtered_models if model.day_ahead]
    else:
        log.info("Excluding Day Ahead model")
        filtered_models = [model for model in filtered_models if not model.day_ahead]

    if not run_extra_models and not get_day_ahead_only and not get_ecmwf_only:
        log.info("Limiting to default pvnet_v2 model")
        filtered_models = [model for model in filtered_models if model.name == "pvnet_v2"]

    if use_ocf_data_sampler:
        log.info("Filtering for models using OCF Data Sampler")
        filtered_models = [model for model in filtered_models if model.uses_ocf_data_sampler]
    else:
        log.info("Filtering for models not using OCF Data Sampler")
        filtered_models = [model for model in filtered_models if not model.uses_ocf_data_sampler]

    selected_model_info = [
        (model.name, f"uses_ocf_data_sampler={model.uses_ocf_data_sampler}")
        for model in filtered_models
    ]
    log.info(f"Selected models: {selected_model_info}")

    return filtered_models
