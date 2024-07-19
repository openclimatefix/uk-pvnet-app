import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import logging
import os
import fsspec
from datetime import timedelta, datetime
import ocf_blosc2
from ocf_datapipes.config.load import load_yaml_configuration

from pvnet_app.consts import sat_path, nwp_ukv_path, nwp_ecmwf_path

logger = logging.getLogger(__name__)

this_dir = os.path.dirname(os.path.abspath(__file__))

sat_5_path = "sat_5_min.zarr"
sat_15_path = "sat_15_min.zarr"


def download_all_sat_data():
    """Download the sat data"""
    
    # Clean out old files
    os.system(f"rm -r {sat_path} {sat_5_path} {sat_15_path}")

    # download 5 minute satellite data
    sat_download_path = os.environ["SATELLITE_ZARR_PATH"]
    fs = fsspec.open(sat_download_path).fs
    if fs.exists(sat_download_path):
        fs.get(sat_download_path, "sat_5_min.zarr.zip")
        os.system(f"unzip -qq sat_5_min.zarr.zip -d {sat_5_path}")
        os.system(f"rm sat_5_min.zarr.zip")
    
    # Also download 15-minute satellite if it exists
    sat_15_dl_path = os.environ["SATELLITE_ZARR_PATH"]\
        .replace("sat.zarr", "sat_15.zarr").replace("latest.zarr", "latest_15.zarr")
    if fs.exists(sat_15_dl_path):
        logger.info(f"Downloading 15-minute satellite data {sat_15_dl_path}")
        fs.get(sat_15_dl_path, "sat_15_min.zarr.zip")
        os.system(f"unzip sat_15_min.zarr.zip -d {sat_15_path}")
        os.system(f"rm sat_15_min.zarr.zip")

        
def _get_latest_time_and_mins_delay(sat_zarr_path, t0):
    ds_sat = xr.open_zarr(sat_zarr_path)
    latest_time = pd.to_datetime(ds_sat.time.max().item())
    delay = t0 - latest_time
    delay_mins = int(delay.total_seconds() / 60)
    all_datetimes = pd.to_datetime(ds_sat.time.values)
    return latest_time, delay_mins, all_datetimes

def combine_5_and_15_sat_data(t0) -> [datetime, int, int]:
    """Select and/or combine the 5 and 15-minutely satellite data

    The return is
    - the most recent timestamp of the data
    - the delay in minutes of the most recent timestamp from t0
    - the data frequency, 5 or 15
    """
    
    # Check which satellite data exists
    exists_5_minute = os.path.exists(sat_5_path)
    exists_15_minute = os.path.exists(sat_15_path)  

    if not exists_5_minute and not exists_15_minute:
        raise FileNotFoundError("Neither 5- nor 15-minutely data was found.")
    
    # Find the delay in the 5- and 15-minutely data
    if exists_5_minute:
        latest_time_5, delay_mins_5, all_datetimes_5 = _get_latest_time_and_mins_delay(sat_5_path, t0)
        logger.info(f"Latest 5-minute timestamp is {latest_time_5} for t0 time {t0}. All the datetimes are {all_datetimes_5}")
    else:
        latest_time_5, delay_mins_5, all_datetimes_5 = datetime.min, np.inf, []
        logger.info(f"No 5-minute data was found.")
        
    if exists_15_minute:
        latest_time_15, delay_mins_15, all_datetimes_15 = _get_latest_time_and_mins_delay(sat_15_path, t0)
        logger.info(f"Latest 5-minute timestamp is {latest_time_15} for t0 time {t0}. All the datetimes are  {all_datetimes_15}")
    else:
        latest_time_15, delay_mins_15, all_datetimes_15 = datetime.min, np.inf, []
        logger.info(f"No 15-minute data was found.")
        
    # Move the data with the most recent timestamp to the expected path
    if latest_time_5>=latest_time_15:
        logger.info(f"Using 5-minutely data.")
        os.system(f"mv {sat_5_path} {sat_path}")
        latest_time = latest_time_5
        delay_mins = delay_mins_5
        data_freq_minutes = 5
        all_datetimes = all_datetimes_5
    else:
        logger.info(f"Using 15-minutely data.")
        os.system(f"mv {sat_15_path} {sat_path}")
        latest_time = latest_time_15
        delay_mins = delay_mins_15
        data_freq_minutes = 15
        all_datetimes = all_datetimes_15
        
    return latest_time, delay_mins, data_freq_minutes, all_datetimes


def extend_satellite_data_with_nans(t0):
    """Fill the satellite data with NaNs out to time t0"""

    # Find how delayed the satellite data is
    _, delay_mins, _ = _get_latest_time_and_mins_delay(sat_path, t0)
    
    if delay_mins > 0:
        logger.info(f"Filling most recent {delay_mins} mins with NaNs")
        
        # Load into memory so we can delete it on disk
        ds_sat = xr.open_zarr(sat_path).compute()
        
        # Pad with zeros
        fill_times = pd.date_range(
            t0+timedelta(minutes=(-delay_mins+5)), 
            t0, 
            freq="5min"
        )
        
        ds_sat = ds_sat.reindex(time=np.concatenate([ds_sat.time, fill_times]), fill_value=np.nan)    

        # Re-save inplace
        os.system(f"rm -rf {sat_path}")
        ds_sat.to_zarr(sat_path)
    
    return delay_mins
        
    
def check_model_inputs_available(data_config_filename, all_satellite_datetimes, t0, data_freq_minutes ):
    """Checks whether the model can be run given the current satellite delay

    Args:
        data_config_filename: Path to the data configuration file
        all_satellite_datetimes: All the satellite datetimes available
        t0: The time the model is trying to forecast
        data_freq_minutes: The frequency of the satellite data. This can be 5 or 15 minutes.

    """
    
    data_config = load_yaml_configuration(data_config_filename)
    
    available = True
    
    # check satellite if using
    if hasattr(data_config.input_data, "satellite"):
        if data_config.input_data.satellite is not None:

            # Take into account how recently the model tries to slice satellite data from
            max_sat_delay_allowed_mins = data_config.input_data.satellite.live_delay_minutes

            # Take into account the dropout the model was trained with, if any
            if data_config.input_data.satellite.dropout_fraction>0:
                max_sat_delay_allowed_mins = max(
                    max_sat_delay_allowed_mins,
                    np.abs(data_config.input_data.satellite.dropout_timedeltas_minutes).max()
                )

            # get start and end satellite times
            history_minutes = data_config.input_data.satellite.history_minutes

            # we only check every 15 minutes, as ocf_datapipes resample from 15 to 5 if necessary.
            # We need to include the end point also
            freq = f"{data_freq_minutes}min"
            logger.info(f"Checking satellite data for {t0} with history {history_minutes} and freq {freq}")
            expected_datetimes = pd.date_range(
                t0 - timedelta(minutes=history_minutes),
                t0 - timedelta(minutes=max_sat_delay_allowed_mins) + timedelta(minutes=data_freq_minutes),
                freq=freq
            )

            # Check if all expected datetimes are in the available satellite data
            all_satellite_data_present = all([t in all_satellite_datetimes for t in expected_datetimes])
            if not all_satellite_data_present:
                # log something. e,g x,y timestamps are missing
                logger.info(f"Missing satellite data for {expected_datetimes} in {all_satellite_datetimes}")

            available = all_satellite_data_present

    return available


def preprocess_sat_data(t0):
    """Combine and 5- and 15-minutely satellite data and extend to t0 if required"""
    
    # Deal with switching between the 5 and 15 minutely satellite data
    _, _, data_freq_minutes, all_datetimes = combine_5_and_15_sat_data(t0)
    
    # Extend the satellite data with NaNs if needed by the model and record the delay of most recent
    # non-nan timestamp
    extend_satellite_data_with_nans(t0)

    # TODO return actual datetimes
    
    return all_datetimes, data_freq_minutes

    
def _download_nwp_data(source, destination):

    logger.info(f"Downloading NWP data from {source} to {destination}")

    fs = fsspec.open(source).fs
    fs.get(source, destination, recursive=True)

    
def download_all_nwp_data():
    """Download the NWP data"""
    _download_nwp_data(os.environ["NWP_UKV_ZARR_PATH"], nwp_ukv_path)
    _download_nwp_data(os.environ["NWP_ECMWF_ZARR_PATH"], nwp_ecmwf_path)


def regrid_nwp_data(nwp_zarr, target_coords_path, method):
    """This function loads the  NWP data, then regrids and saves it back out if the data is not
    on the same grid as expected. The data is resaved in-place.
    """

    logger.info(f'Regridding NWP data {nwp_zarr} to expected grid to {target_coords_path}')
    
    ds_raw = xr.open_zarr(nwp_zarr)

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(target_coords_path)
    
    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude) and
         ds_raw.longitude.equals(ds_target_coords.longitude)
        
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
            {k: regrid_chunk_dict[k] for k in list(ds_raw.xindexes) if k in regrid_chunk_dict}
        )
    ).compute(scheduler="single-threaded")

    # Re-save - including rechunking
    os.system(f"rm -rf {nwp_zarr}")
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
        {k: save_chunk_dict[k] for k in list(ds_raw.xindexes) if k in save_chunk_dict}
    ).to_zarr(nwp_zarr)

    
def fix_ecmwf_data():
    
    ds = xr.open_zarr(nwp_ecmwf_path).compute()
    ds["variable"] = ds["variable"].astype(str)
    
    name_sub = {
        "t": "t2m",
        "clt": "tcc"
    }
    
    if any(v in name_sub for v in ds["variable"].values):
        logger.info(f"Renaming the ECMWF variables")
        ds["variable"] = np.array([name_sub[v] if v in name_sub else v for v in ds["variable"].values])
    else:
        logger.info(f"No ECMWF renaming required - skipping this step")
    
    logger.info(f"Extending the ECMWF data to reach the shetlands")
    # Thw data must be extended to reach the shetlands. This will fill missing lats with NaNs
    # and reflects what the model saw in training
    ds = ds.reindex(latitude=np.concatenate([np.arange(62, 60, -0.05), ds.latitude.values]))
    
    # Re-save inplace
    os.system(f"rm -rf {nwp_ecmwf_path}")
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
    os.system(f"rm -rf {nwp_ukv_path}")
    ds.to_zarr(nwp_ukv_path)
    

def preprocess_nwp_data():
    
    # Regrid the UKV data
    regrid_nwp_data(
        nwp_zarr=nwp_ukv_path, 
        target_coords_path=f"{this_dir}/../data/nwp_ukv_target_coords.nc",
        method="bilinear"
    )
    
    # Regrid the ECMWF data
    regrid_nwp_data(
        nwp_zarr=nwp_ecmwf_path, 
        target_coords_path=f"{this_dir}/../data/nwp_ecmwf_target_coords.nc",
        method="conservative" # this is needed to avoid zeros around edges of ECMWF data
    )
    
    # UKV data must be float16 to allow overflow to inf like in training
    fix_ukv_data()
    
    # Names need to be aligned between training and prod, and we need to infill the shetlands
    fix_ecmwf_data()