from datetime import timezone, datetime
import fsspec.asyn
import yaml
import os
import copy
import xarray as xr
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
import logging

def config_union(config_paths):
    from ocf_datapipes.config.load import load_yaml_configuration

from nowcasting_datamodel.models import (
    ForecastSQL,
    ForecastValue,
)
from nowcasting_datamodel.read.read import (
    get_latest_input_data_last_updated,
    get_location,
)
from nowcasting_datamodel.read.read_models import get_model



from pvnet_app.consts import sat_path, nwp_ukv_path, nwp_ecmwf_path


logger = logging.getLogger(__name__)



def worker_init_fn(worker_id):
    """
    Clear reference to the loop and thread.
    This is a nasty hack that was suggested but NOT recommended by the lead fsspec developer!
    This appears necessary otherwise gcsfs hangs when used after forking multiple worker processes.
    Only required for fsspec >= 0.9.0
    See:
    - https://github.com/fsspec/gcsfs/issues/379#issuecomment-839929801
    - https://github.com/fsspec/filesystem_spec/pull/963#issuecomment-1131709948
    TODO: Try deleting this two lines to make sure this is still relevant.
    """
    fsspec.asyn.iothread[0] = None
    fsspec.asyn.loop[0] = None

    
def load_yaml_config(path):
    """Load config file from path"""
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def save_yaml_config(config, path):
    """Save config file to path"""
    with open(path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def populate_data_config_sources(input_path, output_path):
    """Resave the data config and replace the source filepaths

    Args:
        input_path: Path to input datapipes configuration file
        output_path: Location to save the output configuration file
    """
    config = load_yaml_config(input_path)
        
    production_paths = {
        "gsp": os.environ["DB_URL"],
        "nwp": {"ukv": nwp_ukv_path, "ecmwf": nwp_ecmwf_path},
        "satellite": sat_path,
        # TODO: include hrvsatellite
    }        
    
    # Replace data sources
    for source in ["gsp", "satellite", "hrvsatellite"]:
        if source in config["input_data"] :
            if config["input_data"][source][f"{source}_zarr_path"]!="":
                assert source in production_paths, f"Missing production path: {source}"
                config["input_data"][source][f"{source}_zarr_path"] = production_paths[source]
        
    # NWP is nested so much be treated separately
    if "nwp" in config["input_data"]:
        nwp_config = config["input_data"]["nwp"]
        for nwp_source in nwp_config.keys():
            if nwp_config[nwp_source][f"nwp_zarr_path"]!="":
                assert "nwp" in production_paths, f"Missing production path: nwp"
                assert nwp_source in production_paths["nwp"], f"Missing NWP path: {nwp_source}"
                nwp_config[nwp_source][f"nwp_zarr_path"] = production_paths["nwp"][nwp_source]

    # We do not need to set PV path right now. This currently done through datapipes
    # TODO - Move the PV path to here
    
    save_yaml_config(config, output_path)
        
    
def find_min_satellite_delay_config(config_paths):
    """Find the config with the minimum satallite delay across from list of config paths"""
    # Load all the configs
    configs = [load_yaml_config(config_path) for config_path in config_paths]
    
    min_sat_delay = np.inf
    
    for config in configs:

        min_sat_delay = min(
            min_sat_delay,
            config["input_data"]["satellite"]["live_delay_minutes"]
        )
        
    config = configs[0] 
    config["input_data"]["satellite"]["live_delay_minutes"] = min_sat_delay
    return config

        
def preds_to_dataarray(preds, model, valid_times, gsp_ids):
    """Put numpy array of predictions into a dataarray"""
    
    if model.use_quantile_regression:
        output_labels = model.output_quantiles
        output_labels = [f"forecast_mw_plevel_{int(q*100):02}" for q in model.output_quantiles]
        output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
    else:
        output_labels = ["forecast_mw"]
        preds = preds[..., np.newaxis]

    da = xr.DataArray(
        data=preds,
        dims=["gsp_id", "target_datetime_utc", "output_label"],
        coords=dict(
            gsp_id=gsp_ids,
            target_datetime_utc=valid_times,
            output_label=output_labels,
        ),
    )
    return da
        
        
def convert_dataarray_to_forecasts(
    forecast_values_dataarray: xr.DataArray, session: Session, model_name: str, version: str
) -> list[ForecastSQL]:
    """
    Make a ForecastSQL object from a DataArray.

    Args:
        forecast_values_dataarray: Dataarray of forecasted values. Must have `target_datetime_utc`
            `gsp_id`, and `output_label` coords. The `output_label` coords must have `"forecast_mw"`
            as an element.
        session: database session
        model_name: the name of the model
        version: the version of the model
    Return:
        List of ForecastSQL objects
    """
    logger.debug("Converting DataArray to list of ForecastSQL")

    assert "target_datetime_utc" in forecast_values_dataarray.coords
    assert "gsp_id" in forecast_values_dataarray.coords
    assert "forecast_mw" in forecast_values_dataarray.output_label

    # get last input data
    input_data_last_updated = get_latest_input_data_last_updated(session=session)

    # get model name
    model = get_model(name=model_name, version=version, session=session)

    forecasts = []

    for gsp_id in forecast_values_dataarray.gsp_id.values:
        gsp_id = int(gsp_id)
        # make forecast values
        forecast_values = []

        # get location
        location = get_location(session=session, gsp_id=gsp_id)

        gsp_forecast_values_da = forecast_values_dataarray.sel(gsp_id=gsp_id)

        for target_time in pd.to_datetime(gsp_forecast_values_da.target_datetime_utc.values):
            # add timezone
            target_time_utc = target_time.replace(tzinfo=timezone.utc)
            this_da = gsp_forecast_values_da.sel(target_datetime_utc=target_time)

            forecast_value_sql = ForecastValue(
                target_time=target_time_utc,
                expected_power_generation_megawatts=(
                    this_da.sel(output_label="forecast_mw").item()
                ),
            ).to_orm()

            forecast_value_sql.adjust_mw = 0.0

            properties = {}

            if "forecast_mw_plevel_10" in gsp_forecast_values_da.output_label:
                val = this_da.sel(output_label="forecast_mw_plevel_10").item()
                # `val` can be NaN if PVNet has probabilistic outputs and PVNet_summation doesn't,
                # or if PVNet_summation has probabilistic outputs and PVNet doesn't.
                # Do not log the value if NaN
                if not np.isnan(val):
                    properties["10"] = val

            if "forecast_mw_plevel_90" in gsp_forecast_values_da.output_label:
                val = this_da.sel(output_label="forecast_mw_plevel_90").item()

                if not np.isnan(val):
                    properties["90"] = val
                    
            if len(properties)>0:
                forecast_value_sql.properties = properties

            forecast_values.append(forecast_value_sql)

        # make forecast object
        forecast = ForecastSQL(
            model=model,
            forecast_creation_time=datetime.now(tz=timezone.utc),
            location=location,
            input_data_last_updated=input_data_last_updated,
            forecast_values=forecast_values,
            historic=False,
        )

        forecasts.append(forecast)

    return forecasts



    