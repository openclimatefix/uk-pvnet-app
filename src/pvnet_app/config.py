import yaml

from pvnet_app.consts import nwp_ecmwf_path, nwp_ukv_path, sat_path


def load_yaml_config(path: str) -> dict:
    """Load config file from path
    
    Args:
        path: The path to the config file
    """
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_yaml_config(config: dict, path: str) -> None:
    """Save config file to path
    
    Args:
        config: The config to save
        path: The path to save the config file
    """
    with open(path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def populate_config_with_data_data_filepaths(config: dict) -> dict:
    """Populate the data source filepaths in the config

    Args:
        config: The data config
    """
    production_paths = {
        "nwp": {"ukv": nwp_ukv_path, "ecmwf": nwp_ecmwf_path},
        "satellite": sat_path,
    }

    # Set the GSP input path to null. We don't need it in production
    config["input_data"]["gsp"]["zarr_path"] = ""

    # Replace satellite data path
    if "satellite" in config["input_data"]:
        if config["input_data"]["satellite"]["zarr_path"] != "":
            config["input_data"]["satellite"]["zarr_path"] = production_paths["satellite"]

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["zarr_path"] != "":
                assert nwp_source in production_paths["nwp"], f"Missing NWP path: {nwp_source}"
                nwp_config[nwp_source]["zarr_path"] = production_paths["nwp"][nwp_source]

    return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config drouput parameters for production

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
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["zarr_path"] != "":
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = []
                nwp_config[nwp_source]["dropout_fraction"] = 0

    return config




def modify_data_config_for_production(
    input_path: str, 
    output_path: str, 
) -> None:
    """Resave the data config with the data source filepaths and dropouts overwritten

    Args:
        input_path: Path to input configuration file
        output_path: Location to save the output configuration file
        reformat_config: Reformat config to new format
    """
    config = load_yaml_config(input_path)

    config = populate_config_with_data_data_filepaths(config)
    config = overwrite_config_dropouts(config)

    save_yaml_config(config, output_path)


def get_union_of_configs(config_paths: list[str]) -> dict:
    """Find the config which is able to run all models from a list of config paths

    Note that this implementation is very limited and will not work in general unless all models
    have been trained on the same batches. We do not check example if the satellite and NWP channels
    are the same in the different configs, or whether the NWP time slices are the same. Many more
    limitations not mentioned apply
    """
    # Load all the configs
    configs = [load_yaml_config(config_path) for config_path in config_paths]

    # We will ammend this config according to the entries in the other configs
    common_config = configs[0]

    for config in configs[1:]:

        if "satellite" in config["input_data"]:

            if "satellite" in common_config["input_data"]:

                # Find the minimum satellite delay across configs
                common_config["input_data"]["satellite"]["interval_end_minutes"] = max(
                    common_config["input_data"]["satellite"]["interval_end_minutes"],
                    config["input_data"]["satellite"]["interval_end_minutes"],
                )

            else:
                # Add satellite to common config if not there already
                common_config["input_data"]["satellite"] = config["input_data"]["satellite"]

        if "nwp" in config["input_data"]:

            # Add NWP to common config if not there already
            if "nwp" not in common_config["input_data"]:
                common_config["input_data"]["nwp"] = config["input_data"]["nwp"]

            else:
                for nwp_key, nwp_conf in config["input_data"]["nwp"].items():
                    # Add different NWP sources to common config if not there already
                    if nwp_key not in common_config["input_data"]["nwp"]:
                        common_config["input_data"]["nwp"][nwp_key] = nwp_conf

    return common_config


def get_nwp_channels(provider: str, nwp_config: dict) -> None| list[str]:
    """Get the NWP channels from the NWP config

    Args:
        provider: The NWP provider
        nwp_config: The NWP config
    """
    nwp_channels = None
    if "nwp" in nwp_config["input_data"]:
        for label, source in nwp_config["input_data"]["nwp"].items():
            if source["provider"] == provider:
                nwp_channels = source["channels"]
    return nwp_channels

