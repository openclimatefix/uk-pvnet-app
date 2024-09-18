import yaml

from pvnet_app.consts import sat_path, nwp_ukv_path, nwp_ecmwf_path


def load_yaml_config(path: str) -> dict:
    """Load config file from path"""
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_yaml_config(config: dict, path: str) -> None:
    """Save config file to path"""
    with open(path, 'w') as file:
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
        if source in config["input_data"] :
            if config["input_data"][source][f"{source}_zarr_path"]!="":
                assert source in production_paths, f"Missing production path: {source}"
                config["input_data"][source][f"{source}_zarr_path"] = production_paths[source]
        
    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"]!="":
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
        if source in config["input_data"] :
            if config["input_data"][source][f"{source}_zarr_path"]!="":
                config["input_data"][source][f"dropout_timedeltas_minutes"] = None
        
    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source]["nwp_zarr_path"]!="":
                nwp_config[nwp_source]["dropout_timedeltas_minutes"] = None
                
    return config
    
    
def modify_data_config_for_production(
    input_path: str, 
    output_path: str, 
    gsp_path: str = ""
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
        
    
def find_min_satellite_delay_config(config_paths: list[str]) -> dict:
    """Find the config with the minimum satallite delay across from list of config paths"""

    # Load all the configs
    configs = [load_yaml_config(config_path) for config_path in config_paths]
    
    min_sat_delay = None
    
    for config in configs:
        
        if "satellite" in config["input_data"]:
            
            sat_delay = config["input_data"]["satellite"]["live_delay_minutes"]
            if min_sat_delay is None:
                min_sat_delay = sat_delay
            else:
                min_sat_delay = min(min_sat_delay, sat_delay)
        
    config = configs[0] 
    config["input_data"]["satellite"]["live_delay_minutes"] = min_sat_delay
    return config