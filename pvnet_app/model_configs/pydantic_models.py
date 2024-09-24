""" A pydantic model for the ML models"""

from typing import List, Optional

import fsspec
from pyaml_env import parse_config
from pydantic import BaseModel, Field, field_validator


class ModelHF(BaseModel):
    repo: str = Field(..., title="Repo name", description="The HF Repo")
    version: str = Field(..., title="Repo version", description="The HF version")


class Model(BaseModel):
    """One ML Model"""

    name: str = Field(..., title="Model Name", description="The name of the model")
    pvnet: ModelHF = Field(
        ..., title="PVNet", description="The PVNet model"
    )
    summation: ModelHF = Field(
        ..., title="Summation", description="The Summation model"
    )

    use_adjuster: bool = Field(
        False, title="Use Adjuster", description="Whether to use the adjuster model"
    )
    save_gsp_sum: bool = Field(
        False, title="Save GSP Sum", description="Whether to save the GSP sum"
    )
    verbose: bool = Field(
        False, title="Verbose", description="Whether to print verbose output"
    )
    save_gsp_to_recent: bool = Field(
        False, title="Save GSP to Forecast Value Last Seven Days",
        description="Whether to save the GSP to Forecast Value Last Seven Days"
    )
    day_ahead: bool = Field(
        False, title="Day Ahead", description="If this model is day ahead or not"
    )


class Models(BaseModel):
    """ A group of ml models """
    models: List[Model] = Field(
        ..., title="Models", description="A list of models to use for the forecast"
    )

    @field_validator('models')
    @classmethod
    def name_must_be_unique(cls, v: List[Model]) -> List[Model]:
        """ Ensure that all model names are unique """
        names = [model.name for model in v]
        unique_names = set(names)

        if len(names) == len(unique_names):
            raise Exception(f"Model names must be unique, names are {names}")
        return v


def get_all_models():
    """
    Returns all the models for a given client
    """

    # load models from yaml file
    import os

    filename = os.path.dirname(os.path.abspath(__file__)) + "/all_models.yaml"

    with fsspec.open(filename, mode="r") as stream:
        models = parse_config(data=stream)
        models = Models(**models)

    return models
