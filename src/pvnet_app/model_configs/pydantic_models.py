"""A pydantic model for the ML models."""
import logging
from importlib.resources import files

import fsspec
from pyaml_env import parse_config
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


class HuggingFaceCommit(BaseModel):
    """The location of a model on Hugging Face."""
    repo: str = Field(..., description="The Hugging Face repo")
    commit: str = Field(..., description="The commit hash")


class ModelConfig(BaseModel):
    """Configuration of a model and the settings it will be run with in the app."""

    name: str = Field(..., description="The name of the model")
    pvnet: HuggingFaceCommit = Field(..., description="The PVNet model location")
    summation: HuggingFaceCommit = Field(..., description="The summation model location")

    use_adjuster: bool = Field(False, description="Whether to use the adjuster")
    save_gsp_sum: bool = Field(
        False,
        description="Whether to save the sum of GSPs as welll as the national estimate",
    )
    verbose_logging: bool = Field(False, description="Whether to log verbose output for the model")
    save_gsp_to_recent: bool = Field(
        False,
        description="Whether to save the GSP results to the `ForecastValueLastSevenDays` table",
    )
    is_day_ahead: bool = Field(
        False,
        description="If this model makes day-ahead forecasts (as opposed to intra-day)",
    )
    is_critical: bool = Field(
        False,
        description="If this model must always be part of the critial set of models which should "
        "always be run",
    )
    uses_satellite_data: bool = Field(
        True,
        description="If this model uses satellite data (currently this is only used in tests)",
    )


class ModelConfigCollection(BaseModel):
    """A collection of model configurations."""

    models: list[ModelConfig] = Field(
        ...,
        description="A list of model configs to use for the forecast",
    )

    @field_validator("models")
    @classmethod
    def name_must_be_unique(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        """Ensure that all model names are unique."""
        names = [model.name for model in v]

        if len(names) != len(set(names)):
            raise Exception(f"Model names must be unique, names are {names}")
        return v


def get_all_models(
    allow_adjuster: bool = True,
    allow_save_gsp_sum: bool = True,
    get_critical_only: bool = False,
) -> list[ModelConfig]:
    """Returns all the models for a given client.

    Args:
        allow_adjuster: If set to false, all models will have use_adjuster set to false
        allow_save_gsp_sum: If set to false, all models will have save_gsp_sum set to false
        get_critical_only: If only the critical models should be returned
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
    if not allow_adjuster:
        for model in model_collection.models:
            model.use_adjuster = False
    if not allow_save_gsp_sum:
        for model in model_collection.models:
            model.save_gsp_sum = False

    # Filter models
    filtered_models = model_collection.models.copy()

    if get_critical_only:
        log.info("Filtering to critical models")
        filtered_models = [model for model in filtered_models if model.is_critical]

    # We should always have at least one model
    if len(filtered_models)==0:
        raise Exception("No models found")

    selected_model_info = [model.name for model in filtered_models]
    log.info(f"Selected models: {selected_model_info}")

    return filtered_models
