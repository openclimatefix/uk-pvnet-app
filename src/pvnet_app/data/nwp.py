import logging
import os
import shutil
from importlib.resources import files

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from ocf_datapipes.config.load import load_yaml_configuration

from pvnet_app.consts import nwp_ecmwf_path, nwp_ukv_path

logger = logging.getLogger(__name__)


def _download_nwp_data(source: str, destination: str, provider: str):
    logger.info(f"Downloading NWP data from {source} to {destination}, for {provider}")

    if source is None:
        logger.warning(f"Source file for NWP provider {provider} is not set. "
                       f"Skipping download. One possible way to fix this is to "
                       f"set the environment variable NWP_{provider}_ZARR_PATH")
        return

    fs = fsspec.open(source).fs
    if fs.exists(source):
        fs.get(source, destination, recursive=True)
    else:
        logger.warning(f"NWP data from {source} does not exist")


def download_all_nwp_data():
    """Download the NWP data"""

    _download_nwp_data(os.getenv("NWP_UKV_ZARR_PATH"), nwp_ukv_path, 'UKV')
    _download_nwp_data(os.getenv("NWP_ECMWF_ZARR_PATH"), nwp_ecmwf_path, 'ECMWF')


def regrid_nwp_data(nwp_zarr, target_coords_path, method):
    """This function loads the  NWP data, then regrids and saves it back out if the data is not
    on the same grid as expected. The data is resaved in-place.
    """
    logger.info(f"Regridding NWP data {nwp_zarr} to expected grid to {target_coords_path}")

    ds_raw = xr.open_zarr(nwp_zarr)

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(target_coords_path)

    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude)
        and ds_raw.longitude.equals(ds_target_coords.longitude)
    )

    if not needs_regridding:
        logger.info(f"No NWP regridding required for {nwp_zarr} - skipping this step")
        return

    logger.info(f"Regridding NWP {nwp_zarr} to expected grid")

    # Pull the raw data into RAM
    ds_raw = ds_raw.compute()

    # Regrid in RAM efficient way by chunking first. Each step is regridded separately
    regrid_chunk_dict = {
        "step": 1,
        "latitude": -1,
        "longitude": -1,
        "x": -1,
        "y": -1,
    }

    regridder = xe.Regridder(ds_raw, ds_target_coords, method=method)
    ds_regridded = regridder(
        ds_raw.chunk(
            {k: regrid_chunk_dict[k] for k in list(ds_raw.xindexes) if k in regrid_chunk_dict},
        ),
    ).compute(scheduler="single-threaded")

    # Re-save - including rechunking
    shutil.rmtree(nwp_zarr, ignore_errors=True)
    ds_regridded["variable"] = ds_regridded["variable"].astype(str)

    # Rechunk to these dimensions when saving
    save_chunk_dict = {
        "step": 5,
        "latitude": 100,
        "longitude": 100,
        "x": 100,
        "y": 100,
    }

    ds_regridded.chunk(
        {k: save_chunk_dict[k] for k in list(ds_raw.xindexes) if k in save_chunk_dict},
    ).to_zarr(nwp_zarr)


def fix_ecmwf_data():

    ds = xr.open_zarr(nwp_ecmwf_path).compute()
    ds["variable"] = ds["variable"].astype(str)

    name_sub = {"t": "t2m", "clt": "tcc"}

    if any(v in name_sub for v in ds["variable"].values):
        logger.info("Renaming the ECMWF variables")
        ds["variable"] = np.array(
            [name_sub[v] if v in name_sub else v for v in ds["variable"].values],
        )
    else:
        logger.info("No ECMWF renaming required - skipping this step")

    logger.info("Extending the ECMWF data to reach the shetlands")
    # Thw data must be extended to reach the shetlands. This will fill missing lats with NaNs
    # and reflects what the model saw in training
    ds = ds.reindex(latitude=np.concatenate([np.arange(62, 60, -0.05), ds.latitude.values]))

    # Re-save inplace
    shutil.rmtree(nwp_ecmwf_path, ignore_errors=True)
    ds.to_zarr(nwp_ecmwf_path)


def fix_ukv_data():
    """Extra steps to align UKV production data with training

    - In training the UKV data is float16. This causes it to overflow into inf values which are then
      clipped.
    """
    ds = xr.open_zarr(nwp_ukv_path).compute()
    ds = ds.astype(np.float16)

    ds["variable"] = ds["variable"].astype(str)

    # Re-save inplace
    shutil.rmtree(nwp_ukv_path, ignore_errors=True)
    ds.to_zarr(nwp_ukv_path)


def preprocess_nwp_data():

    if os.path.exists(nwp_ukv_path):

        rename_ukv_variables()

        # Regrid the UKV data
        regrid_nwp_data(
            nwp_zarr=nwp_ukv_path,
            target_coords_path=files("pvnet_app.data").joinpath("nwp_ukv_target_coords.nc"),
            method="bilinear",
        )

        # UKV data must be float16 to allow overflow to inf like in training
        fix_ukv_data()

    if os.path.exists(nwp_ecmwf_path):

        # rename dataset variable from  HRES-IFS_uk to ECMWF_UK
        rename_ecmwf_variables()

        # Regrid the ECMWF data
        regrid_nwp_data(
            nwp_zarr=nwp_ecmwf_path,
            target_coords_path=files("pvnet_app.data").joinpath("nwp_ecmwf_target_coords.nc"),
            method="conservative",  # this is needed to avoid zeros around edges of ECMWF data
        )

        # Names need to be aligned between training and prod, and we need to infill the shetlands
        fix_ecmwf_data()


def rename_ecmwf_variables():
    """Rename the ECMWF variables to what we use in the ML Model"""
    d = xr.open_zarr(nwp_ecmwf_path)
    # if the variable HRES-IFS_uk is there
    if ("HRES-IFS_uk" in d.data_vars) or ("hres-ifs_uk" in d.data_vars):
        logger.info("Renaming the ECMWF variables")
        if "HRES-IFS_uk" in d.data_vars:
            d = d.rename({"HRES-IFS_uk": "ECMWF_UK"})
        else:
            d = d.rename({"hres-ifs_uk": "ECMWF_UK"})

        # remove anything >60 in latitude
        logger.info("Removing data above 60 latitude")
        d = d.where(d.latitude <= 60, drop=True)

        # remove anything step > 83
        logger.info("Removing data after step 83, step 84 is nan")
        d = d.where(d.step <= d.step[83], drop=True)

        # rename variable names in the variable coordinate
        # This is a renaming from ECMWF variables to what we use in the ML Model
        # This change happened in the new nwp-consumer>=1.0.0
        # Ideally we won't need this step in the future
        variable_coords = d.variable.values
        rename = {"cloud_cover_high": "hcc",
                  "cloud_cover_low": "lcc",
                  "cloud_cover_medium": "mcc",
                  "cloud_cover_total": "tcc",
                  "snow_depth_gl": "sde",
                  "direct_shortwave_radiation_flux_gl": "sr",
                  "downward_longwave_radiation_flux_gl": "dlwrf",
                  "downward_shortwave_radiation_flux_gl": "dswrf",
                  "downward_ultraviolet_radiation_flux_gl": "duvrs",
                  "temperature_sl": "t",
                  "total_precipitation_rate_gl": "prate",
                  "visibility_sl": "vis",
                  "wind_u_component_100m": "u100",
                  "wind_u_component_10m": "u10",
                  "wind_u_component_200m": "u200",
                  "wind_v_component_100m": "v100",
                  "wind_v_component_10m": "v10",
                  "wind_v_component_200m": "v200"}

        for k, v in rename.items():
            variable_coords[variable_coords == k] = v

        # assign the new variable names
        d = d.assign_coords(variable=variable_coords)
        d = d.compute()

        # save back to path
        shutil.rmtree(nwp_ecmwf_path, ignore_errors=True)
        d.to_zarr(nwp_ecmwf_path)


def rename_ukv_variables():
    d = xr.open_zarr(nwp_ukv_path)

    # if um-ukv is in the datavars, then this comes from the new new-consuerm
    # We need to rename the data variables, and
    # load in lat and lon, ready for regridding later.
    if 'um-ukv' in d.data_vars:

        logger.info("Renaming the UKV variables")

        # rename to UKV
        d = d.rename({"um-ukv": "UKV"})

        variable_coords = d.variable.values
        rename = {"cloud_cover_high": "hcc",
                  "cloud_cover_low": "lcc",
                  "cloud_cover_medium": "mcc",
                  "cloud_cover_total": "tcc",
                  "snow_depth_gl": "sde",
                  "direct_shortwave_radiation_flux_gl": "sr",
                  "downward_longwave_radiation_flux_gl": "dlwrf",
                  "downward_shortwave_radiation_flux_gl": "dswrf",
                  "downward_ultraviolet_radiation_flux_gl": "duvrs",
                  "relative_humidity_sl": "r",
                  "temperature_sl": "t",
                  "total_precipitation_rate_gl": "prate",
                  "visibility_sl": "vis",
                  "wind_direction_10m": "wdir10",
                  "wind_speed_10m": "si10",
                  "wind_v_component_10m": "v10",
                  "wind_u_component_10m": "u10"}

        for k, v in rename.items():
            variable_coords[variable_coords == k] = v

        # assign the new variable names
        d = d.assign_coords(variable=variable_coords)

        # this is all taken from the metoffice website, apart from the x and y values
        lat = xr.open_dataset(files("pvnet_app.data").joinpath("nwp-consumer-mo-ukv-lat.nc"))
        lon = xr.open_dataset(files("pvnet_app.data").joinpath("nwp-consumer-mo-ukv-lon.nc"))

        # combine with d
        d = d.assign_coords(latitude=lat.latitude)
        d = d.assign_coords(longitude=lon.longitude)
        d = d.compute()

        # save back to path
        shutil.rmtree(nwp_ukv_path, ignore_errors=True)
        d.to_zarr(nwp_ukv_path)


def check_model_nwp_inputs_available(
    data_config_filename: str,
    t0: pd.Timestamp,
) -> bool:
    """Checks whether the model can be run given the available NWP data

    Args:
        data_config_filename: Path to the data configuration file
        t0: The init-time of the forecast

    Returns:
        bool: Whether the NWP timestamps satisfy that specified in the config
    """
    input_config = load_yaml_configuration(data_config_filename).input_data

    available = True

    # check satellite if using
    if hasattr(input_config, "nwp") and (input_config.nwp is not None):

        for nwp_source, nwp_zarr_path in zip(["ukv", "ecmwf"], [nwp_ukv_path, nwp_ecmwf_path]):

            if nwp_source in input_config.nwp:

                if not os.path.exists(nwp_zarr_path):
                    available = False
                
                else:

                    ds_nwp = xr.open_zarr(nwp_zarr_path)

                    nwp_config = input_config.nwp[nwp_source]

                    # Find the available valid times of the NWP data
                    assert len(ds_nwp.init_time) == 1, "These checks assume a single init_time"
                    available_nwp_times = (
                        pd.to_datetime(ds_nwp.init_time.values[0]) + pd.to_timedelta(ds_nwp.step)
                    )

                    # Get the NWP valid times required by the model
                    freq = pd.Timedelta(f"{nwp_config.time_resolution_minutes}min")

                    req_start_time = (
                        t0 - pd.Timedelta(f"{nwp_config.history_minutes}min")
                    ).ceil(freq)

                    req_end_time = (
                        t0 + pd.Timedelta(f"{nwp_config.forecast_minutes}min")
                    ).ceil(freq) 
                    
                    # If we diff accumulated channels in time we'll need one more timestamp
                    if len(nwp_config.nwp_accum_channels)>0:
                        req_end_time = req_end_time + freq

                    required_nwp_times = pd.date_range(
                        start=req_start_time, 
                        end=req_end_time, 
                        freq=freq,
                    )

                    # Check if any of the expected datetimes are missing
                    missing_time_steps = np.setdiff1d(
                        required_nwp_times, 
                        available_nwp_times, 
                        assume_unique=True
                    )

                    available = available and (len(missing_time_steps)==0)

                    if len(missing_time_steps) > 0:
                        logger.warning(
                            f"Some {nwp_source} timesteps for {t0=} missing: \n{missing_time_steps}"
                        )

    return available

