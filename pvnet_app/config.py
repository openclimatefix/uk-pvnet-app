import yaml

from pvnet_app.consts import sat_path, nwp_ukv_path, nwp_ecmwf_path


def load_yaml_config(path: str) -> dict:
    """Load config file from path"""
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_yaml_config(config: dict, path: str) -> None:
    """Save config file to path"""
    with open(path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def populate_config_with_data_data_filepaths(config: dict, gsp_path: str = "") -> dict:
    """Populate the data source filepaths in the config

    Args:
        config: The data config
        gsp_path: For lagacy usage only
    """

    production_paths = {
        "gsp": gsp_path,
        "nwp": {"ukv": nwp_ukv_path, "ecmwf": nwp_ecmwf_path},
        "satellite": sat_path,
    }

    # Replace data sources
    for source in ["gsp", "satellite"]:
        if source in config["input_data"]:
            if config["input_data"][source][f"{source}_zarr_path"] != "":
                assert source in production_paths, f"Missing production path: {source}"
                config["input_data"][source][f"{source}_zarr_path"] = production_paths[source]

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":
                assert "nwp" in production_paths, "Missing production path: nwp"
                assert nwp_source in production_paths["nwp"], f"Missing NWP path: {nwp_source}"
                nwp_config[nwp_source]["nwp_zarr_path"] = production_paths["nwp"][nwp_source]

    return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config drouput parameters for production

    Args:
        config: The data config
    """

    # Replace data sources
    for source in ["satellite"]:
        if source in config["input_data"]:
            if config["input_data"][source][f"{source}_zarr_path"] != "":
                config["input_data"][source][f"dropout_timedeltas_minutes"] = [0]

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = [0]

    return config


def modify_data_config_for_production(
    input_path: str, output_path: str, gsp_path: str = ""
) -> None:
    """Resave the data config with the data source filepaths and dropouts overwritten

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
        gsp_path: For lagacy usage only
    """
    config = load_yaml_config(input_path)

    config = populate_config_with_data_data_filepaths(config, gsp_path=gsp_path)
    config = overwrite_config_dropouts(config)

    save_yaml_config(config, output_path)


def get_union_of_configs(config_paths: list[str]) -> dict:
    """Find the config which is able to run all models from a list of config paths

    Note that this implementation is very limited and will not work in general unless all models
    have been trained on the same batches. We do not chck example if the satellite and NWP channels
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
                common_config["input_data"]["satellite"]["live_delay_minutes"] = min(
                    common_config["input_data"]["satellite"]["live_delay_minutes"],
                    config["input_data"]["satellite"]["live_delay_minutes"],
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
