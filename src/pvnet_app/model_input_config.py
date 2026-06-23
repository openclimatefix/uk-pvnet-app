"""Functions to load, save, and modify a PVNet model's data-configuration file."""
import yaml

from pvnet_app.consts import (
    generation_path,
    nwp_cloudcasting_path,
    nwp_ecmwf_path,
    nwp_ukv_path,
    sat_path,
)


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


def populate_config_with_data_filepaths(config: dict) -> dict:
    """Populate the data source filepaths in the config.

    Args:
        config: The data config
    """
    nwp_paths = {
        "ukv": nwp_ukv_path,
        "ecmwf": nwp_ecmwf_path,
        "cloudcasting": nwp_cloudcasting_path,
    }

    # Set the GSP input path
    config["input_data"]["generation"]["zarr_path"] = generation_path

    # Replace satellite data path
    if "satellite" in config["input_data"]:
        config["input_data"]["satellite"]["zarr_path"] = sat_path

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


def modify_data_config_for_production(input_path: str, output_path: str) -> None:
    """Resave the data config with the data source filepaths and dropouts overwritten.

    Args:
        input_path: Path to input configuration file
        output_path: Location to save the output configuration file
        reformat_config: Reformat config to new format
    """
    config = load_yaml_config(input_path)

    config = populate_config_with_data_filepaths(config)
    config = overwrite_config_dropouts(config)

    save_yaml_config(config, output_path)
