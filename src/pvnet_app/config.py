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


def populate_config_with_data_data_filepaths(config: dict, gsp_path: str = "") -> dict:
    """Populate the data source filepaths in the config

    Args:
        config: The data config
        gsp_path: Address of the GSP database (for lagacy usage only)
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
                config["input_data"][source][f"{source}_zarr_path"] = production_paths[source]

    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":
                assert nwp_source in production_paths["nwp"], f"Missing NWP path: {nwp_source}"
                nwp_config[nwp_source]["nwp_zarr_path"] = production_paths["nwp"][nwp_source]

    return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config drouput parameters for production

    Args:
        config: The data config
    """
    # Replace data sources
    if "satellite" in config["input_data"]:

        satellite_config = config["input_data"]["satellite"]

        if satellite_config["satellite_zarr_path"] != "":
            satellite_config["dropout_timedeltas_minutes"] = None
            satellite_config["dropout_fraction"] = 0

    # NWP is nested so must be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = None
                nwp_config[nwp_source]["dropout_fraction"] = 0

    return config


def reformat_config_data_sampler(config: dict) -> dict:
    """Reformat config

    This is to keep the configurations from ocf-data-sampler==0.0.19 working,
    we need to upgrade them a bit to the configuration in ocf-data-sampler>=0.1.5

    Args:
        config: The data config
    """
    # Replace satellite
    if "satellite" in config["input_data"]:

        satellite_config = config["input_data"]["satellite"]

        if satellite_config["satellite_zarr_path"] != "":

            rename_pairs = [
                ("satellite_image_size_pixels_width", "image_size_pixels_width"),
                ("satellite_image_size_pixels_height", "image_size_pixels_height"),
                ("forecast_minutes", "interval_end_minutes"),
                ("satellite_zarr_path", "zarr_path"),
                ("satellite_channels", "channels"),
            ]

            update_config(
                rename_pairs=rename_pairs,
                config=satellite_config,
                remove_keys=["live_delay_minutes"]
            )

    # NWP is nested so must be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"] != "":

                rename_pairs = [
                    ("nwp_image_size_pixels_width", "image_size_pixels_width"),
                    ("nwp_image_size_pixels_height", "image_size_pixels_height"),
                    ("forecast_minutes", "interval_end_minutes"),
                    ("nwp_zarr_path", "zarr_path"),
                    ("nwp_accum_channels", "accum_channels"),
                    ("nwp_channels", "channels"),
                    ("nwp_provider", "provider"),
                ]

                update_config(rename_pairs=rename_pairs, config=nwp_config[nwp_source])

    if "gsp" in config["input_data"]:

        gsp_config = config["input_data"]["gsp"]

        rename_pairs = [
            ("forecast_minutes", "interval_end_minutes"),
            ("gsp_zarr_path", "zarr_path"),
        ]

        update_config(rename_pairs=rename_pairs, config=gsp_config)

    update_config(
        rename_pairs=[],
        config=config["input_data"],
        change_history_minutes=False,
        remove_keys=["default_forecast_minutes", "default_history_minutes"]
    )

    return config


def update_config(rename_pairs: list, config: dict, change_history_minutes: bool = True, remove_keys=None):
    """Update the config in place with rename pairs, and remove keys if they exist

    1. Rename keys in the config
    2. Change history minutes to interval start minutes, with a negative value
    3. Remove keys from the config

    Args:
        rename_pairs: list of pairs to rename
        config: the config dict
        change_history_minutes: option to change history minutes to interval start minutes
        remove_keys: list of key to remove
    """
    for old, new in rename_pairs:
        if old in config:
            config[new] = config[old]
            del config[old]

    if change_history_minutes:
        if "history_minutes" in config:
            config["interval_start_minutes"] = -config["history_minutes"]
            del config["history_minutes"]

    if remove_keys is not None:
        for key in remove_keys:
            if key in config:
                del config[key]


def modify_data_config_for_production(
    input_path: str, 
    output_path: str, 
    gsp_path: str = "", 
    reformat_config: bool = False,
) -> None:
    """Resave the data config with the data source filepaths and dropouts overwritten

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
        gsp_path: Address of the GSP database (for lagacy usage only)
        reformat_config: Reformat config to new format
    """
    config = load_yaml_config(input_path)

    config = populate_config_with_data_data_filepaths(config, gsp_path=gsp_path)
    config = overwrite_config_dropouts(config)
    if reformat_config:
        config = reformat_config_data_sampler(config)

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


def get_nwp_channels(provider: str, nwp_config: dict) -> None| list[str]:
    """Get the NWP channels from the NWP config

    Args:
        provider: The NWP provider
        nwp_config: The NWP config
    """
    nwp_channels = None
    if "nwp" in nwp_config["input_data"]:
        if provider in nwp_config["input_data"]["nwp"]:
            nwp_channels = nwp_config["input_data"]["nwp"][provider]["nwp_channels"]
    return nwp_channels

