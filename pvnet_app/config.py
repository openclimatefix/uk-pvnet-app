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


# def populate_config_with_data_data_filepaths(config: dict, gsp_path: str = "") -> dict:
#     """Populate the data source filepaths in the config

#     Args:
#         config: The data config
#         gsp_path: For lagacy usage only
#     """
#     production_paths = {
#         "gsp": gsp_path,
#         "nwp": {"ukv": nwp_ukv_path, "ecmwf": nwp_ecmwf_path},
#         "satellite": sat_path,
#     }

#     # Replace data sources - GSP and satellite
#     for source in ["gsp", "satellite"]:
#         # v0 and v1 schema
#         if source in config["input_data"]:
#             zarr_path_key = (
#                 f"{source}_zarr_path"
#                 if f"{source}_zarr_path" in config["input_data"][source]
#                 else "zarr_path"
#             )
            
#             if config["input_data"][source][zarr_path_key] != "":
#                 assert source in production_paths, f"Missing production path: {source}"
#                 config["input_data"][source][zarr_path_key] = production_paths[source]

#     # Handle NWP separately - nested
#     if "nwp" in config["input_data"]:
#         nwp_config = config["input_data"]["nwp"]
#         for nwp_source in nwp_config.keys():
#         # v0 and v1 schema
#             zarr_path_key = (
#                 "nwp_zarr_path"
#                 if "nwp_zarr_path" in nwp_config[nwp_source]
#                 else "zarr_path"
#             )
#             provider_key = (
#                 "nwp_provider"
#                 if "nwp_provider" in nwp_config[nwp_source]
#                 else "provider"
#             )

#             if zarr_path_key in nwp_config[nwp_source] and nwp_config[nwp_source][zarr_path_key] != "":
#                 provider = nwp_config[nwp_source][provider_key].lower()
#                 assert provider in production_paths["nwp"], f"Missing NWP path: {provider}"
#                 nwp_config[nwp_source][zarr_path_key] = production_paths["nwp"][provider]

#     return config


def populate_config_with_data_data_filepaths(config: dict, gsp_path: str = "") -> dict:
    """Populate the data source filepaths in the config with backwards compatibility

    Args:
        config: The data config
        gsp_path: For legacy usage only
    """
    production_paths = {
        "gsp": gsp_path,
        "nwp": {"ukv": nwp_ukv_path, "ecmwf": nwp_ecmwf_path},
        "satellite": sat_path,
    }

    # Backward compatibility for attribute handling
    for source in ["gsp", "satellite"]:
        if source in config["input_data"]:
            source_config = config["input_data"][source]
            path_key = f"{source}_zarr_path"
            
            if source_config.get(path_key, ""):
                assert source in production_paths, f"Missing production path: {source}"
                source_config[path_key] = production_paths[source]
                
                if source == "gsp":
                    source_config["handle_legacy_attrs"] = True

    # NWP is nested - treated separately 
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            zarr_path = nwp_config[nwp_source].get("nwp_zarr_path", "")
            if zarr_path:
                assert "nwp" in production_paths, "Missing production path: nwp"
                assert nwp_source in production_paths["nwp"], f"Missing NWP path: {nwp_source}"
                nwp_config[nwp_source]["nwp_zarr_path"] = production_paths["nwp"][nwp_source]

                # v0 and v1 schema
                old_keys = [
                    ("zarr_path", "nwp_zarr_path"),
                    ("channels", "nwp_channels"),
                    ("image_size_pixels_height", "nwp_image_size_pixels_height"),
                    ("image_size_pixels_width", "nwp_image_size_pixels_width"),
                    ("provider", "nwp_provider")
                ]
                
                for old_key, new_key in old_keys:
                    if old_key in nwp_config[nwp_source]:
                        nwp_config[nwp_source][new_key] = nwp_config[nwp_source].pop(old_key)

    return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config drouput parameters for production

    Args:
        config: The data config
    """

    # Replace data source - satellite
    for source in ["satellite"]:
        
        # v0 and v1 schema
        if source in config["input_data"]:
            zarr_path_key = (
                f"{source}_zarr_path"
                if f"{source}_zarr_path" in config["input_data"][source]
                else "zarr_path"
            )
            if config["input_data"][source][zarr_path_key] != "":
                config["input_data"][source]["dropout_timedeltas_minutes"] = None

    # Handle NWP separately - nested
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():

        # v0 and v1 schema
            zarr_path_key = (
                "nwp_zarr_path"
                if "nwp_zarr_path" in nwp_config[nwp_source]
                else "zarr_path"
            )
            if zarr_path_key in nwp_config[nwp_source] and nwp_config[nwp_source][zarr_path_key] != "":
                config["input_data"]["nwp"][nwp_source]["dropout_timedeltas_minutes"] = None

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
