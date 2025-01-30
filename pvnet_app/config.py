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
    """Populate the data source filepaths in the config with schema version handling

    Args:
        config: The data config
        gsp_path: For legacy usage only
    """
    production_paths = {
        "gsp": gsp_path,
        "nwp": {"ukv": nwp_ukv_path, "ecmwf": nwp_ecmwf_path},
        "satellite": sat_path,
    }

    # Handle GSP and satellite sources
    for source in ["gsp", "satellite"]:
        if source in config["input_data"]:
            source_config = config["input_data"][source]
            
            # Determine schema version and appropriate key
            schema_version = source_config.get("config_schema_version", "v0")
            zarr_path_key = "zarr_path" if schema_version == "v1" else f"{source}_zarr_path"
            
            # Specific handling for GSP to ensure data sampler compatibility
            if source == "gsp":
                # Ensure backward compatibility and add missing keys
                # source_config.setdefault('config_schema_version', schema_version)
                
                # Add default keys for OCF data sampler
                gsp_keys_to_ensure = [
                    'installed_capacity_mwp', 
                    'generation_mw', 
                    'effective_capacity_mwp',
                    'capacity_mwp'
                ]
                
                # Ensure zarr path keys are consistent
                if schema_version == "v1":
                    if f"gsp_zarr_path" in source_config:
                        source_config["zarr_path"] = source_config.pop(f"gsp_zarr_path")

                        for key in gsp_keys_to_ensure:
                            source_config.setdefault(key, True)
                else:
                    if "zarr_path" in source_config:
                        source_config["gsp_zarr_path"] = source_config.pop("zarr_path")

            if not source_config.get(zarr_path_key, ""):
                assert source in production_paths, f"Missing production path: {source}"
                source_config[zarr_path_key] = production_paths[source]

    # Handle NWP - nested structure
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():

            # Check schema version for NWP
            schema_version = nwp_config[nwp_source].get("config_schema_version", "v0")
            
            # Set appropriate keys based on schema version
            if schema_version == "v1":
                path_key = "zarr_path"
                provider_key = "provider"
                channel_key = "channels"
                height_key = "image_size_pixels_height"
                width_key = "image_size_pixels_width"
            else:
                path_key = "nwp_zarr_path"
                provider_key = "nwp_provider"
                channel_key = "nwp_channels"
                height_key = "nwp_image_size_pixels_height"
                width_key = "nwp_image_size_pixels_width"

            if not nwp_config[nwp_source].get(path_key, ""):
                provider = nwp_config[nwp_source][provider_key].lower()
                assert provider in production_paths["nwp"], f"Missing NWP path: {provider}"
                nwp_config[nwp_source][path_key] = production_paths["nwp"][provider]

            # Ensure all keys align with schema version
            for v0_key, v1_key in [
                ("nwp_zarr_path", "zarr_path"),
                ("nwp_channels", "channels"),
                ("nwp_image_size_pixels_height", "image_size_pixels_height"),
                ("nwp_image_size_pixels_width", "image_size_pixels_width"),
                ("nwp_provider", "provider")
            ]:
                if schema_version == "v1" and v0_key in nwp_config[nwp_source]:
                    nwp_config[nwp_source][v1_key] = nwp_config[nwp_source].pop(v0_key)
                elif schema_version == "v0" and v1_key in nwp_config[nwp_source]:
                    nwp_config[nwp_source][v0_key] = nwp_config[nwp_source].pop(v1_key)

    return config


# def overwrite_config_dropouts(config: dict) -> dict:
#     """Overwrite the config dropout parameters for production with schema version handling

#     Args:
#         config: The data config
#     """

#     # Replace data source - satellite
#     for source in ["satellite"]:
#         if source in config["input_data"]:
#             source_config = config["input_data"][source]
            
#             # Check schema version
#             schema_version = source_config.get("config_schema_version", "v0")
#             zarr_path_key = "zarr_path" if schema_version == "v1" else f"{source}_zarr_path"
            
#             if source_config.get(zarr_path_key, ""):
#                 source_config["dropout_timedeltas_minutes"] = None

#     # Handle NWP separately - nested structure
#     if "nwp" in config["input_data"]:
#         nwp_config = config["input_data"]["nwp"]
#         for nwp_source in nwp_config.keys():
#             # Check schema version
#             schema_version = nwp_config[nwp_source].get("config_schema_version", "v0")
#             path_key = "zarr_path" if schema_version == "v1" else "nwp_zarr_path"
            
#             if nwp_config[nwp_source].get(path_key, ""):
#                 nwp_config[nwp_source]["dropout_timedeltas_minutes"] = None
                
#     return config


def overwrite_config_dropouts(config: dict) -> dict:
    """Overwrite the config dropout parameters for production with enhanced schema version handling

    Args:
        config: The data config
    """
    # Ensure input_data exists
    if "input_data" not in config:
        config["input_data"] = {}

    # Replace data source - satellite
    for source in ["satellite"]:
        if source in config["input_data"]:
            source_config = config["input_data"][source]
            
            # Determine path keys
            path_keys = [f"{source}_zarr_path", "zarr_path"]
            path_key = next((key for key in path_keys if key in source_config), path_keys[0])
            
            # Ensure dropout key exists
            if "dropout_timedeltas_minutes" not in source_config:
                source_config["dropout_timedeltas_minutes"] = None
            
            # Set dropout to None if path exists
            if source_config.get(path_key, ""):
                source_config["dropout_timedeltas_minutes"] = None

    # Handle NWP separately - nested structure
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            # Determine path keys
            path_keys = ["nwp_zarr_path", "zarr_path"]
            path_key = next((key for key in path_keys if key in nwp_config[nwp_source]), path_keys[0])
            
            # Ensure dropout key exists
            if "dropout_timedeltas_minutes" not in nwp_config[nwp_source]:
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = None
            
            # Set dropout to None if path exists
            if nwp_config[nwp_source].get(path_key, ""):
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = None
                
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


# PURELY FOR TESTING PURPOSE FOR NOW
def ensure_zarr_metadata(zarr_path: str) -> None:
    """
    Ensure that a Zarr store has proper metadata, creating it if necessary.
    
    Args:
        zarr_path (str): Path to the Zarr store
    """
    import os
    import json
    import zarr
    import xarray as xr

    try:
        # Check if the Zarr path exists
        if not os.path.exists(zarr_path):
            return

        # Try to open with xarray first to generate metadata
        try:
            ds = xr.open_zarr(zarr_path)
            ds.to_zarr(zarr_path, mode='a')
        except Exception as e:
            print(f"Error opening Zarr with xarray: {e}")

        # Open the Zarr store
        store = zarr.DirectoryStore(zarr_path)

        # Ensure .zgroup exists
        if '.zgroup' not in store:
            store['.zgroup'] = json.dumps({"zarr_format": 2}).encode('utf-8')

        # Check if .zmetadata exists
        if '.zmetadata' not in store:
            # Create basic metadata
            metadata = {
                "zarr_format": 2,
                "variables": {},
                "attributes": {}
            }

            # Try to list groups/datasets
            try:
                zgroup = zarr.open_group(store=store)
                for key in zgroup.keys():
                    dataset = zgroup[key]
                    metadata["variables"][key] = {
                        "dtype": str(dataset.dtype),
                        "shape": dataset.shape
                    }
            except Exception as e:
                print(f"Error creating metadata: {e}")

            # Write metadata
            try:
                store['.zmetadata'] = json.dumps(metadata).encode('utf-8')
            except Exception as e:
                print(f"Error writing .zmetadata: {e}")

    except Exception as e:
        print(f"Unexpected error in ensure_zarr_metadata: {e}")
