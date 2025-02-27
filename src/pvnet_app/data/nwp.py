import logging
import shutil
from importlib.resources import files

from abc import ABC, abstractmethod
from typing_extensions import override

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from ocf_datapipes.config.load import load_yaml_configuration

from pvnet_app.consts import nwp_ecmwf_path, nwp_ukv_path

logger = logging.getLogger(__name__)


def download_data(source: str, destination: str) -> bool:

    fs = fsspec.open(source).fs

    file_exists = fs.exists(source)
    if file_exists:
        fs.get(source, destination, recursive=True)
    return file_exists


def regrid_nwp_data(
    ds: xr.Dataset, 
    target_coords_path: str, 
    method: str, 
    nwp_source: str
) -> xr.Dataset:
    """This function regrids the input NWP data to the grid of the target path

    Args:
        ds: The NWP data to regrid
        target_coords_path: The path to the target grid
        method: The regridding method to use
        nwp_source: The source of the NWP data - only used for logging messages
    """
    logger.info(f"Regridding{nwp_source} to expected grid to {target_coords_path}")

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(target_coords_path)

    # Check if regridding step needs to be done
    needs_regridding = not (
        ds.latitude.equals(ds_target_coords.latitude)
        and ds.longitude.equals(ds_target_coords.longitude)
    )

    if not needs_regridding:
        logger.info(f"No regridding required for {nwp_source} - skipping this step")
        return

    logger.info(f"Regridding {nwp_source} to expected grid")

    # Regrid in RAM efficient way by chunking first. Each step is regridded separately
    regrid_chunk_dict = {
        "step": 1,
        "latitude": -1,
        "longitude": -1,
        "x": -1,
        "y": -1,
    }

    ds_rechunked = ds.chunk(
        {k: regrid_chunk_dict[k] for k in list(ds.xindexes) if k in regrid_chunk_dict}
    )

    regridder = xe.Regridder(ds, ds_target_coords, method=method)
    return regridder(ds_rechunked).compute(scheduler="single-threaded")


def get_nwp_valid_times(ds: xr.Dataset) -> pd.DatetimeIndex:
    return pd.to_datetime(ds.init_time.values[0]) + pd.to_timedelta(ds.step)


def check_model_nwp_inputs_available(
    data_config_filename: str,
    t0: pd.Timestamp,
    nwp_source: str,
    nwp_valid_times: pd.DatetimeIndex | None,
) -> bool:
    """Checks whether the model can be run given the available NWP data

    Args:
        data_config_filename: Path to the data configuration file
        t0: The init-time of the forecast

    Returns:
        bool: Whether the NWP timestamps satisfy that specified in the config
    """
    input_config = load_yaml_configuration(data_config_filename).input_data

    # Only check if using NWP data
    model_uses_nwp = (
        hasattr(input_config, "nwp") 
        and (input_config.nwp is not None)
        and (nwp_source in input_config.nwp)
    )

    if model_uses_nwp and (nwp_valid_times is None):
        available = False

    elif model_uses_nwp:

        nwp_config = input_config.nwp[nwp_source]

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
            nwp_valid_times, 
            assume_unique=True
        )

        available = len(missing_time_steps)==0

        if len(missing_time_steps) > 0:
            logger.warning(
                f"Some {nwp_source} timesteps for {t0=} missing: \n{missing_time_steps}"
            )
    
    else:
        available = True

    return available



class NWPDownloader(ABC):

    destination_path: str = None
    nwp_source: str = None
    save_chunk_dict: dict = None

    def __init__(self, source_path: str | None):
        self.source_path = source_path
        self.valid_times = None
    
    @abstractmethod
    def process(self, ds: xr.Dataset) -> xr.Dataset:
        """"Apply all processing steps to the NWP data in order to match the training data"""
        pass


    def resave(self, ds: xr.Dataset) -> None:
        """Resave the NWP data to the destination path"""

        ds["variable"] = ds["variable"].astype(str)
        
        # Overwrite the old data
        shutil.rmtree(self.destination_path, ignore_errors=True)
        ds.chunk(self.save_chunk_dict).to_zarr(self.destination_path)


    def run(self) -> None:
        """Download, process, and save the NWP data"""

        logger.info(f"Downloading and processing the {self.nwp_source} data")

        if self.source_path is None:
            logger.warning(f"Source file for {self.nwp_source} is not set. Skipping download.")
            return
        
        file_exists = download_data(self.source_path, self.destination_path)
        if not file_exists:
            logger.warning(
                f"Source file {self.source_path} for {self.nwp_source} does not exist. "
                "Skipping download."
            )
            return

        ds = xr.open_zarr(self.destination_path).compute()

        ds = self.process(ds)

        # Store the valid times for the NWP data
        self.valid_times = get_nwp_valid_times(ds)

        self.resave(ds)
    
    
    def check_model_inputs_available(
        self,
        data_config_filename: str,
        t0: pd.Timestamp,
    ) -> bool:

        return check_model_nwp_inputs_available(
            data_config_filename=data_config_filename,
            t0=t0,
            nwp_source=self.nwp_source,
            nwp_valid_times=self.valid_times,
        ) 



class ECMWFDownloader(NWPDownloader):

    destination_path = nwp_ecmwf_path
    nwp_source = "ecmwf"
    save_chunk_dict = {
        "step": 10,
        "latitude": 50,
        "longitude": 50,
    }

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        return regrid_nwp_data(
            ds=ds,
            target_coords_path=files("pvnet_app.data").joinpath("nwp_ecmwf_target_coords.nc"),
            method="conservative",  # this is needed to avoid zeros around edges of ECMWF data
            nwp_source="ECMWF",
        )


    def extend_to_shetlands(self, ds: xr.Dataset) -> xr.Dataset:
        """Extend the ECMWF data to reach the shetlands (with NaNS) as in the training data
        
        The training data stoped at 60 degrees latitude but extended with NaNs to reach the 
        Shetlands. We repeat this here.
        """

        logger.info("Extending the ECMWF data to reach the shetlands")

        # The data must be extended to reach the shetlands. This will fill missing lats with NaNs
        # and reflects what the model saw in training
        return ds.reindex(latitude=np.concatenate([np.arange(62, 60, -0.05), ds.latitude.values]))


    def rename_variables(self, ds):
        """Rename the ECMWF variables to match the training data"""
        
        logger.info("Renaming the ECMWF variables")
        ds = ds.rename({"hres-ifs_uk": "ECMWF_UK"})

        # rename variable names in the variable coordinate
        # This is a renaming from ECMWF variables to what we use in the ML Model
        # This change happened in the new nwp-consumer>=1.0.0
        # Ideally we won't need this step in the future
        variable_coords = ds.variable.values
        rename = {
            "cloud_cover_high": "hcc",
            "cloud_cover_low": "lcc",
            "cloud_cover_medium": "mcc",
            "cloud_cover_total": "tcc",
            "snow_depth_gl": "sde",
            "direct_shortwave_radiation_flux_gl": "sr",
            "downward_longwave_radiation_flux_gl": "dlwrf",
            "downward_shortwave_radiation_flux_gl": "dswrf",
            "downward_ultraviolet_radiation_flux_gl": "duvrs",
            "temperature_sl": "t2m",
            "total_precipitation_rate_gl": "prate",
            "visibility_sl": "vis",
            "wind_u_component_100m": "u100",
            "wind_u_component_10m": "u10",
            "wind_u_component_200m": "u200",
            "wind_v_component_100m": "v100",
            "wind_v_component_10m": "v10",
            "wind_v_component_200m": "v200"
        }

        for k, v in rename.items():
            variable_coords[variable_coords == k] = v

        # assign the new variable names
        ds = ds.assign_coords(variable=variable_coords)
        
        return ds


    def remove_nans(self, ds: xr.Dataset) -> xr.Dataset:
        """Remove the NaNs introduced the the NWP consumer bug

        - The last step of the ECMWF data is NaN
        - All data above 60 degrees latitude is NaN
        
        See: https://github.com/openclimatefix/nwp-consumer/issues/218
        """

        logger.info("Removing data above 60 latitude")
        ds = ds.where(ds.latitude <= 60, drop=True)

        logger.info("Removing data after step 84, step 85 is nan")
        ds = ds.isel(step=slice(None, 84))

        return ds
    
    @override
    def process(self, ds: xr.Dataset) -> xr.Dataset:

        # This regridding explicitly puts the data on the exact same grid as in training
        # Regridding must be done before .extend_to_shetlands() is called
        ds = self.remove_nans(ds)
        ds = self.regrid(ds)
        ds = self.extend_to_shetlands(ds)
        ds = self.rename_variables(ds)

        return ds
    

class UKVDownloader(NWPDownloader):

    destination_path = nwp_ukv_path
    nwp_source = "ukv"
    save_chunk_dict = {
        "step": 10,
        "x": 100,
        "y": 100,
    }

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        """Regrid the UKV data to the target grid
        
        In production the UKV data is on a different grid structure to the training data. The
        trraining data is on a regular OSGB grid. The production data is on some other curvilinear 
        grid.        
        """
        return regrid_nwp_data(
            ds=ds,
            target_coords_path=files("pvnet_app.data").joinpath("nwp_ukv_target_coords.nc"),
            method="bilinear",
            nwp_source="UKV",
        )


    def fix_dtype(self, ds: xr.Dataset) -> xr.Dataset:
        """Fix the dtype of the UKV data.
        
        In training the UKV data is float16. This causes it to overflow into inf values which we 
        now need to force in production to be consistent with training.
        """
        return ds.astype(np.float16)

    @override
    def process(self, ds: xr.Dataset) -> xr.Dataset:

        ds = self.regrid(ds)
        ds = self.fix_dtype(ds)

        return ds