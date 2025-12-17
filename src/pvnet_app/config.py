"""Functions to load and save configuration files."""
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
        config = yaml.safe_load(file)
    return config


def save_yaml_config(config: dict, path: str) -> None:
    """Save config file to path.

    Args:
        config: The config to save
        path: The path to save the config file
    """
    with open(path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def populate_config_with_data_data_filepaths(config: dict) -> dict:
    """Populate the data source filepaths in the config.

    Args:
        config: The data config
    """
    production_paths = {
        "nwp": {
            "ukv": nwp_ukv_path,
            "ecmwf": nwp_ecmwf_path,
            "cloudcasting": nwp_cloudcasting_path,
        },
        "satellite": sat_path,
    }

    # Set the GSP input path to null. We don't need it in production
    config["input_data"]["generation"]["zarr_path"] = generation_path

    # Replace satellite data path
    if "satellite" in config["input_data"] and \
        config["input_data"]["satellite"]["zarr_path"] != "":
            config["input_data"]["satellite"]["zarr_path"] = production_paths["satellite"]

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config:
            provider = nwp_config[nwp_source]["provider"]
            if nwp_config[nwp_source]["zarr_path"] != "" and \
                provider not in production_paths["nwp"]:
                raise ValueError(f"Unknown NWP provider: {provider}")
            nwp_config[nwp_source]["zarr_path"] = production_paths["nwp"][provider]

    return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config drouput parameters for production.

    Args:
        config: The data config
    """
    # Replace data sources
    if "satellite" in config["input_data"]:
        satellite_config = config["input_data"]["satellite"]

        if satellite_config["zarr_path"] != "":
            satellite_config["dropout_timedeltas_minutes"] = []
            satellite_config["dropout_fraction"] = 0

    # NWP is nested so must be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config:
            if nwp_config[nwp_source]["zarr_path"] != "":
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = []
                nwp_config[nwp_source]["dropout_fraction"] = 0

    return config


def modify_data_config_for_production(
    input_path: str,
    output_path: str,
) -> None:
    """Resave the data config with the data source filepaths and dropouts overwritten.

    Args:
        input_path: Path to input configuration file
        output_path: Location to save the output configuration file
        reformat_config: Reformat config to new format
    """
    config = load_yaml_config(input_path)

    config = populate_config_with_data_data_filepaths(config)
    config = overwrite_config_dropouts(config)

    save_yaml_config(config, output_path)
