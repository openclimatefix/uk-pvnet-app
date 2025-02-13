"""A pydantic model for the ML models"""
import logging
import os
from importlib.resources import files

import fsspec
from pyaml_env import parse_config
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


class ModelHF(BaseModel):
    repo: str = Field(..., title="Repo name", description="The HF Repo")
    version: str = Field(..., title="Repo version", description="The HF version")


class Model(BaseModel):
    """One ML Model"""

    name: str = Field(..., title="Model Name", description="The name of the model")
    pvnet: ModelHF = Field(..., title="PVNet", description="The PVNet model")
    summation: ModelHF = Field(..., title="Summation", description="The Summation model")

    use_adjuster: bool | None = Field(
        False, title="Use Adjuster", description="Whether to use the adjuster model",
    )
    save_gsp_sum: bool | None = Field(
        False, title="Save GSP Sum", description="Whether to save the GSP sum",
    )
    verbose: bool | None = Field(
        False, title="Verbose", description="Whether to print verbose output",
    )
    save_gsp_to_recent: bool | None = Field(
        False,
        title="Save GSP to Forecast Value Last Seven Days",
        description="Whether to save the GSP to Forecast Value Last Seven Days",
    )
    day_ahead: bool | None = Field(
        False, title="Day Ahead", description="If this model is day ahead or not",
    )

    ecmwf_only: bool | None = Field(
        False, title="ECMWF ONly", description="If this model is only using ecmwf data",
    )

    uses_satellite_data: bool | None = Field(
        True, title="Uses Satellite Data", description="If this model uses satellite data",
    )

    uses_ocf_data_sampler: bool | None = Field(
        True,
        title="Uses OCF Data Sampler",
        description="If this model uses data sampler, old one uses ocf_datapipes",
    )


class Models(BaseModel):
    """A group of ml models"""

    models: list[Model] = Field(
        ..., title="Models", description="A list of models to use for the forecast",
    )

    @field_validator("models")
    @classmethod
    def name_must_be_unique(cls, v: list[Model]) -> list[Model]:
        """Ensure that all model names are unique, respect to using ocf_data_sampler or not"""
        names = [(model.name, model.uses_ocf_data_sampler) for model in v]
        unique_names = set(names)

        if len(names) != len(unique_names):
            raise Exception(f"Model names must be unique, names are {names}")
        return v


def get_all_models(
    get_ecmwf_only: bool | None = False,
    get_day_ahead_only: bool | None = False,
    run_extra_models: bool | None = False,
    use_ocf_data_sampler: bool | None = True,
) -> list[Model]:
    """Returns all the models for a given client

    Args:
        get_ecmwf_only: If only the ECMWF model should be returned
        get_day_ahead_only: If only the day ahead model should be returned
        run_extra_models: If extra models should be run
        use_ocf_data_sampler: If the OCF Data Sampler should be used
    """
    try:
        # load models from yaml file
        filename = files("pvnet_app.model_configs").joinpath("all_models.yaml")

        with fsspec.open(filename, mode="r") as stream:
            try:
                models_dict = parse_config(data=stream)
                models = Models(**models_dict)
            except Exception as config_error:
                log.error(f"Error parsing model configuration: {config_error}")
                raise

        models = config_pvnet_v2_model(models)

        # Apply filters based on input parameters
        filtered_models = models.models.copy()

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

    except Exception as e:
        log.error(f"Critical error in get_all_models: {e}")
        raise


def config_pvnet_v2_model(models):
    """Function to adjust pvnet model"""
    # special case for environment variables
    use_adjuster = os.getenv("USE_ADJUSTER", "true").lower() == "true"
    save_gsp_sum = os.getenv("SAVE_GSP_SUM", "false").lower() == "true"
    # find index where name=pvnet_v2
    pvnet_v2_index = 0
    models.models[pvnet_v2_index].use_adjuster = use_adjuster
    models.models[pvnet_v2_index].save_gsp_sum = save_gsp_sum

    return models
