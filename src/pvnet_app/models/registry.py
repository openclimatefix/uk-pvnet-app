"""A pydantic model for the ML models."""

import logging
from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class HuggingFaceCommit(BaseModel):
    """The location of a model on Hugging Face."""

    repo: str = Field(..., description="The Hugging Face repo")
    commit: str = Field(..., description="The commit hash")


class ModelSpec(BaseModel):
    """Specification of a model variant including artifact references and deployment settings."""

    name: str = Field(..., description="The name of the model")
    pvnet: HuggingFaceCommit = Field(..., description="The PVNet model location")
    summation: HuggingFaceCommit = Field(..., description="The summation model location")

    log_level: Literal["INFO", "DEBUG"] = Field(..., description="Log level to use for the model")
    is_day_ahead: bool = Field(
        False,
        description="If this model makes day-ahead forecasts (as opposed to intra-day)",
    )
    is_critical: bool = Field(
        False,
        description="If this model must always be part of the critical set of models which should "
        "always be run",
    )
    uses_satellite_data: bool = Field(
        True,
        description="If this model uses satellite data (currently this is only used in tests)",
    )


class ModelRegistry(BaseModel):
    """The full collection of model specs loaded from the catalog."""

    models: list[ModelSpec] = Field(
        ...,
        description="A list of model specs to use for the forecast",
    )

    @field_validator("models")
    @classmethod
    def name_must_be_unique(cls, v: list[ModelSpec]) -> list[ModelSpec]:
        """Ensure that all model names are unique."""
        names = [model.name for model in v]

        if len(names) != len(set(names)):
            raise ValueError(f"Model names must be unique, names are {names}")
        return v


def get_model_specs(get_critical_only: bool = False) -> list[ModelSpec]:
    """Return model specs from the catalog.

    Args:
        get_critical_only: If only the critical models should be returned
    """
    with files("pvnet_app.models").joinpath("catalogue.yaml").open("r") as f:
        models_dict = yaml.safe_load(f)

    model_collection = ModelRegistry(**models_dict)

    if get_critical_only:
        logger.info("Filtering to critical models")
        filtered_models = [model for model in model_collection.models if model.is_critical]
    else:
        filtered_models = model_collection.models

    logger.info(f"Selected models: {[m.name for m in filtered_models]}")

    return filtered_models
