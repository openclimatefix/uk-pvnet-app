"""Functions to load, save, and modify a PVNet model's data-configuration file."""
import yaml

from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.consts import (
    generation_path,
    nwp_cloudcasting_path,
    nwp_ecmwf_path,
    nwp_ukv_path,
    sat_path,
)
from pvnet_app.models.registry import ModelSpec




def load_yaml_config(path: str) -> dict:
    """Load config file from path.

    Args:
        path: The path to the config file
    """
    with open(path) as file:
        return yaml.safe_load(file)


def save_yaml_config(config: dict, path: str) -> None:
    """Save config file to path.

    Args:
        config: The config to save
        path: The path to save the config file
    """
    with open(path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def populate_config_with_data_filepaths(config: dict, run_data_dir: str) -> dict:
    """Populate the data source filepaths in the config.

    Args:
        config: The data config
        run_data_dir: The directory where the downloaded input data is stored
    """
    nwp_paths = {
        "ukv": f"{run_data_dir}/{nwp_ukv_path}",
        "ecmwf": f"{run_data_dir}/{nwp_ecmwf_path}",
        "cloudcasting": f"{run_data_dir}/{nwp_cloudcasting_path}",
    }

    # Set the GSP input path
    config["input_data"]["generation"]["zarr_path"] = f"{run_data_dir}/{generation_path}"

    # Replace satellite data path
    if "satellite" in config["input_data"]:
        config["input_data"]["satellite"]["zarr_path"] = f"{run_data_dir}/{sat_path}"

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config:
            provider = nwp_config[nwp_source]["provider"]
            if provider not in nwp_paths:
                raise ValueError(f"Unknown NWP provider: {provider}")
            nwp_config[nwp_source]["zarr_path"] = nwp_paths[provider]

    return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config dropout parameters for production.

    Args:
        config: The data config
    """
    # Remove satellite dropout
    if "satellite" in config["input_data"]:
        satellite_config = config["input_data"]["satellite"]

        satellite_config["dropout_timedeltas_minutes"] = []
        satellite_config["dropout_fraction"] = 0

    # Remove NWP dropout
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config:
            nwp_config[nwp_source]["dropout_timedeltas_minutes"] = []
            nwp_config[nwp_source]["dropout_fraction"] = 0

    return config


def modify_data_config_for_production(input_path: str, output_path: str, run_data_dir: str) -> None:
    """Resave the data config with the data source filepaths and dropouts overwritten.

    Args:
        input_path: Path to input configuration file
        output_path: Location to save the output configuration file
        run_data_dir: The directory where the downloaded input data is stored
    """
    config = load_yaml_config(input_path)

    config = populate_config_with_data_filepaths(config, run_data_dir=run_data_dir)
    config = overwrite_config_dropouts(config)

    save_yaml_config(config, output_path)


def get_required_nwp_providers(data_configs: list[dict]) -> set[str]:
    """Return the set of NWP providers required by any of the model data configs."""
    providers = set()
    for conf in data_configs:
        for source in conf["input_data"].get("nwp", {}).values():
            providers.add(source["provider"])
    return providers


def load_model_data_configs(
    model_specs: list[ModelSpec],
    hf_token: str | None,
) -> tuple[dict[str, str], list[dict]]:
    """Fetch each model's data config; return (paths by model name, loaded configs)."""
    paths: dict[str, str] = {}
    configs: list[dict] = []
    for spec in model_specs:
        path = PVNetBaseModel.get_data_config(
            spec.pvnet.repo, revision=spec.pvnet.commit, token=hf_token,
        )
        paths[spec.name] = path
        configs.append(load_yaml_config(path))
    return paths, configs


def fetch_model_data_config_paths(
    model_specs: list[ModelSpec],
    hf_token: str | None,
) -> dict[str, str]:
    """Return the local data config path for each model, keyed by model name.

    Downloads each config from Hugging Face if it is not already cached locally.

    Args:
        model_specs: The specs of the models to fetch data config paths for
        hf_token: Hugging Face token, required for models in private repos
    """
    paths: dict[str, str] = {}
    for spec in model_specs:
        path = PVNetBaseModel.get_data_config(
            spec.pvnet.repo, revision=spec.pvnet.commit, token=hf_token,
        )
        paths[spec.name] = path
    return paths
