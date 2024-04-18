import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import logging
import os
import fsspec
from datetime import timedelta
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
    
    # Also download 15-minute satellite if it exists
    sat_15_dl_path = os.environ["SATELLITE_ZARR_PATH"]\
        .replace("sat.zarr", "sat_15.zarr").replace("latest.zarr", "latest_15.zarr")
    if fs.exists(sat_15_dl_path):
        logger.info(f"Downloading 15-minute satellite data {sat_15_dl_path}")
        fs.get(sat_15_dl_path, "sat_15_min.zarr.zip")
        os.system(f"unzip sat_15_min.zarr.zip -d {sat_15_path}")

        
def _get_latest_time_and_mins_delay(sat_zarr_path, t0):
    ds_sat = xr.open_zarr(sat_zarr_path)
    latest_time = pd.to_datetime(ds_sat.time.max().item())
    delay = t0 - latest_time
    delay_mins = int(delay.total_seconds() / 60)
    return latest_time, delay_mins


def combine_5_and_15_sat_data(t0, max_sat_delay_allowed_mins):
    """Select and/or combine the 5 and 15-minutely satellite data"""

    use_5_minute = os.path.exists(sat_5_path)
    if not use_5_minute:
        logger.info(f"5-minute satellite data not found at {sat_5_path}. Trying 15-minute data.")
    else:
        latest_time_5, delay_mins_5 = _get_latest_time_and_mins_delay(sat_5_path, t0)
        logger.info(f"Latest 5-minute timestamp is {latest_time_5} for t0 time {t0}.")
        
        if delay_mins_5 <= max_sat_delay_allowed_mins:  
            logger.info(
                f"5-min satellite delay is only {delay_mins_5} minutes. "
                f"Maximum delay for this model is {max_sat_delay_allowed_mins} minutes - "
                "Using 5-minutely data."
            )
            os.system(f"mv {sat_5_path} {sat_path}")
        else:
            logger.info(
                f"5-min satellite delay is {delay_mins_5} minutes. "
                f"Maximum delay for this model is {max_sat_delay_allowed_mins} minutes - "
                "Trying 15-minutely data."
            )
            use_5_minute = False

    if not use_5_minute:
        # Make sure the 15-minute data is actually there
        if not os.path.exists(sat_15_path):
            raise ValueError(f"5-minute satellite data not found at {sat_15_path}")
        
        latest_time_15, delay_mins_15 = _get_latest_time_and_mins_delay(sat_15_path, t0)     
        logger.info(f"Latest 15-minute timestamp is {latest_time_15} for t0 time {t0}.")
        
        # If the 15-minute satellite data is too delayed the run fails
        if delay_mins_15 > max_sat_delay_allowed_mins:
            raise ValueError(
                f"15-min satellite delay is {delay_mins_15} minutes. "
                f"Maximum delay for this model is {max_sat_delay_allowed_mins} minutes"
            )
        
        ds_sat_15 = xr.open_zarr(sat_15_path)
        
        #logger.debug("Resampling 15 minute data to 5 mins")
        #ds_sat_15.resample(time="5T").interpolate("linear").to_zarr(sat_path)
        ds_sat_15.attrs["source"] = "15-minute"

        ds_sat_15.to_zarr(sat_path)
        

def extend_satellite_data_with_nans(t0, min_sat_delay_used_mins):
    """Fill the satellite data with NaNs if needed by the model"""

    # Check how the expected satellite delay compares with the satellite data available and fill
    # if required
    latest_time, delay_mins = _get_latest_time_and_mins_delay(sat_path, t0)
    
    if min_sat_delay_used_mins < delay_mins:
        fill_mins = delay_mins - min_sat_delay_used_mins
        logger.info(f"Filling most recent {fill_mins} mins with NaNs")
        
        # Load into memory so we can delete it on disk
        ds_sat = xr.open_zarr(sat_path).compute()
        
        # Pad with zeros
        fill_times = pd.date_range(
            latest_time+timedelta(minutes=5), 
            latest_time+timedelta(minutes=fill_mins), 
            freq="5min"
        )
        
        
        ds_sat = ds_sat.reindex(time=np.concatenate([ds_sat.time, fill_times]), fill_value=np.nan)    

        # Re-save inplace
        os.system(f"rm -rf {sat_path}")
        ds_sat.to_zarr(sat_path)
        

def preprocess_sat_data(t0, data_config_filename):
    
    # Find the max delay w.r.t t0 that this model was trained with
    data_config = load_yaml_configuration(data_config_filename)
        
    # Take into account how recently the model tries to slice data from
    max_sat_delay_allowed_mins = data_config.input_data.satellite.live_delay_minutes
    
    # Take into account the dropout the model was trained with, if any
    if data_config.input_data.satellite.dropout_fraction>0:
        max_sat_delay_allowed_mins = max(
            max_sat_delay_allowed_mins, 
            np.abs(data_config.input_data.satellite.dropout_timedeltas_minutes).max()
        )
    
    # The model will not ever try to use data more recent than this
    min_sat_delay_used_mins = data_config.input_data.satellite.live_delay_minutes
    
    # Deal with switching between the 5 and 15 minutely satellite data
    combine_5_and_15_sat_data(t0, max_sat_delay_allowed_mins)
    
    # Extend the satellite data with NaNs if needed by the model
    extend_satellite_data_with_nans(t0, min_sat_delay_used_mins)
    
    ds_sat = xr.open_zarr(sat_path)
    ds_sat.data.isnull().mean().compute()
    #assert False

    
def _download_nwp_data(source, destination):
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