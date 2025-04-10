import logging
import shutil
from importlib.resources import files

from abc import ABC, abstractmethod
from typing_extensions import override

import fsspec
import xesmf as xe

from ocf_data_sampler.config.load import load_yaml_configuration

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from pvnet_app.consts import nwp_ecmwf_path, nwp_ukv_path, nwp_cloudcasting_path


logger = logging.getLogger(__name__)


def download_data(source: str, destination: str) -> bool:
    """Download data from a source to a destination
    
    Args:
        source: The source path
        destination: The destination path
    """

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
        nwp_source: The source of the NWP data (only used for logging messages)
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

        req_start_time = (t0 + pd.Timedelta(f"{nwp_config.interval_start_minutes}min")).ceil(freq)

        req_end_time = (t0 + pd.Timedelta(f"{nwp_config.interval_end_minutes}min")).ceil(freq) 
            
        # If we diff accumulated channels in time we'll need one more timestamp
        if len(nwp_config.accum_channels)>0:
            req_end_time = req_end_time + freq

        required_nwp_times = pd.date_range(start=req_start_time, end=req_end_time, freq=freq)

        # Check if any of the expected datetimes are missing
        missing_time_steps = np.setdiff1d(required_nwp_times, nwp_valid_times, assume_unique=True)

        available = len(missing_time_steps)==0

        if len(missing_time_steps) > 0:
            logger.warning(f"Some {nwp_source} timesteps for {t0=} missing: \n{missing_time_steps}")
    
    else:
        available = True

    return available



class NWPDownloader(ABC):

    destination_path: str = None
    nwp_source: str = None
    save_chunk_dict: dict = None

    def __init__(self, source_path: str | None, nwp_variables: list[str] | None = None):
        self.source_path = source_path
        self.nwp_variables = nwp_variables
        # Initially no valid times are available. This will only change is the data can be 
        # downloaded, processed, and saved successfully
        self.valid_times = None
    
    @abstractmethod
    def process(self, ds: xr.Dataset) -> xr.Dataset:
        """"Apply all processing steps to the NWP data in order to match the training data"""
        pass

    @abstractmethod
    def data_is_okay(self, ds: xr.Dataset) -> bool:
        """Apply quality checks to the NWP data
        
        Args:
            ds: The NWP data

        Returns:
            bool: Whether the data passes the quality checks
        """
        pass

    def resave(self, ds: xr.Dataset) -> None:
        """Resave the NWP data to the destination path"""

        ds["variable"] = ds["variable"].astype(str)

        for var in ds.data_vars:
            # Remove the chunks from the data variables
            ds[var].encoding.pop("chunks", None)
        
        # Overwrite the old data
        shutil.rmtree(self.destination_path, ignore_errors=True)
        ds.chunk(self.save_chunk_dict).to_zarr(self.destination_path)

    def filter_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Filter the NWP data to only include the variables needed by the models"""

        if self.nwp_variables is not None:
            logger.info(f"Selecting variables: {self.nwp_variables} from {ds.variable.values}")
            ds = ds.sel(variable=self.nwp_variables)

        return ds

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

        init_time = pd.to_datetime(ds.init_time.values[0])
        valid_times = init_time + pd.to_timedelta(ds.step)
        logger.info(
            f"{self.nwp_source} has init-time {init_time} and valid times: {valid_times}"
        )

        # Check the data is okay before processing
        if not self.data_is_okay(ds):
            logger.warning(f"{self.nwp_source} NWP data did not pass quality checks.")
            return

        ds = self.process(ds)
        self.resave(ds)

        # Only store the valid_times if the NWP data has been successfully downloaded, 
        # quality checked, and processed. Else valid_times will be None
        self.valid_times = valid_times            


    def clean_up(self) -> None:
        """Remove the downloaded data"""
        shutil.rmtree(self.destination_path, ignore_errors=True)
    
    
    def check_model_inputs_available(
        self,
        data_config_filename: str,
        t0: pd.Timestamp,
    ) -> bool:
        """Check if the NWP data the model needs is available
        
        Args:
            data_config_filename: The path to the data configuration file
            t0: The init-time of the forecast
        """

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
    
    @staticmethod
    def regrid(ds: xr.Dataset) -> xr.Dataset:
        """Regrid the ECMWF data to the target grid
        
        In training the ECMWF was at twice the resolution as we have available in production. This
        regridding step will put the data on the same grid as the training data
        """
        return regrid_nwp_data(
            ds=ds,
            target_coords_path=files("pvnet_app.data").joinpath("nwp_ecmwf_target_coords.nc"),
            method="conservative",  # this is needed to avoid zeros around edges of ECMWF data
            nwp_source="ECMWF",
        )

    @staticmethod
    def extend_to_shetlands(ds: xr.Dataset) -> xr.Dataset:
        """Extend the ECMWF data to reach the shetlands (with NaNS) as in the training data
        
        The training data stopped at 60 degrees latitude but was extended with NaNs to reach the 
        Shetlands. We repeat this here.
        """

        logger.info("Extending the ECMWF data to reach the shetlands")

        # The data must be extended to reach the shetlands. This will fill missing lats with NaNs
        # and reflects what the model saw in training
        return ds.reindex(latitude=np.concatenate([np.arange(62, 60, -0.05), ds.latitude.values]))

    @staticmethod
    def rename_variables(ds):
        """Rename the ECMWF variables to match the training data
        
        Rename variable names in the variable coordinate to match the names the model expects and 
        was trained on.
        
        This change happened in the new nwp-consumer>=1.0.0. Ideally we won't need this step in the
        future once the training data is updated.
        """
        
        logger.info("Renaming the ECMWF variables")
        ds = ds.rename({"hres-ifs_uk": "ECMWF_UK"})

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

    @staticmethod
    def remove_nans(ds: xr.Dataset) -> xr.Dataset:
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
        ds = self.rename_variables(ds)
        ds = self.filter_variables(ds)
        ds = self.regrid(ds)
        ds = self.extend_to_shetlands(ds)

        return ds
    
    @override
    def data_is_okay(self, ds: xr.Dataset) -> bool:
        # Need to slice off known nans first
        ds = self.remove_nans(ds)
        contains_nans = ds[list(ds.data_vars.keys())[0]].isnull().any().compute().item()
        return not contains_nans



class UKVDownloader(NWPDownloader):

    destination_path = nwp_ukv_path
    nwp_source = "ukv"
    save_chunk_dict = {
        "step": 10,
        "x": 100,
        "y": 100,
    }

    @staticmethod
    def regrid(ds: xr.Dataset) -> xr.Dataset:
        """Regrid the UKV data to the target grid
        
        In production the UKV data is on a different grid structure to the training data. The
        training data from CEDA is on a regular OSGB grid. The production data is on some other 
        curvilinear grid.        
        """
        return regrid_nwp_data(
            ds=ds,
            target_coords_path=files("pvnet_app.data").joinpath("nwp_ukv_target_coords.nc"),
            method="bilinear",
            nwp_source="UKV",
        )

    @staticmethod
    def fix_dtype(ds: xr.Dataset) -> xr.Dataset:
        """Fix the dtype of the UKV data.
        
        In training the UKV data is float16. This caused it to overflow into inf values for the 
        visibility channel which is measured in metres and can be above 2**16=65km. We 
        need to force this overflow to happen in production to be consistent with training.
        """
        return ds.astype(np.float16)

    @staticmethod
    def rename_variables(ds):
        """Change the UKV variable names to match the training data"""

        # This is for nwp-consumer>=1.0.0
        logger.info("Renaming the UKV variables")

        ds = ds.rename({"um-ukv": "UKV"})

        varname_mapping = {
            "cloud_cover_high": "hcc",
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
            "wind_u_component_10m": "u10"
        }

        variable_coords = [varname_mapping.get(v, v) for v in ds.variable.values]
            
        ds = ds.assign_coords(variable=variable_coords)

        return ds

    @staticmethod
    def add_lon_lat_coords(ds: xr.Dataset) -> xr.Dataset:
        """Add latitude and longitude coords to the UKV data
        
        The training UKV data is on a regular OSGB grid but the live data is on a Lambert Azimuthal
        Equal Area grid. We need to add longitudes and latitudes coords so we can regrid the data
        to the training grid.
        """

        # This is for nwp-consumer>=1.0.0

        logger.info("Adding lon-lat coords to the UKV data")

        ds = ds.rename({'x_laea': 'x', 'y_laea': 'y'})

        # This is the Lambert Azimuthal Equal Area projection used in the UKV live data
        laea = pyproj.Proj(
            proj='laea',
            lat_0=54.9,
            lon_0=-2.5,
            x_0=0.,
            y_0=0.,
            ellps="WGS84",
            datum="WGS84",
        )

        # WGS84 is short for "World Geodetic System 1984". This is a lon-lat coord system
        wgs84 = pyproj.Proj(f"+init=EPSG:4326")

        laea_to_lon_lat = pyproj.Transformer.from_proj(laea, wgs84, always_xy=True).transform

        # Calculate longitude and latitude from x_laea and y_laea
        # - x is an array of shape (455,)
        # - y is an array of shape (639,)
        # We need to change x and y to a 2D arrays of shape (455, 639)
        x, y = ds.x.values, ds.y.values
        x = x.reshape(1, -1).repeat(len(ds.y.values), axis=0)
        y = y.reshape(-1, 1).repeat(len(ds.x.values), axis=1)

        lons, lats = laea_to_lon_lat(xx=x, yy=y)

        ds = ds.assign_coords(
            longitude=(["y", "x"], lons),
            latitude=(["y", "x"], lats),
        )

        return ds

    @override
    def process(self, ds: xr.Dataset) -> xr.Dataset:

        ds = self.rename_variables(ds)
        ds = self.filter_variables(ds)
        ds = self.add_lon_lat_coords(ds)
        ds = self.regrid(ds)
        ds = self.fix_dtype(ds)

        return ds
    
    @override
    def data_is_okay(self, ds: xr.Dataset) -> bool:
        contains_nans = ds[list(ds.data_vars.keys())[0]].isnull().any().compute().item()
        return not contains_nans


class CloudcastingDownloader(NWPDownloader):

    destination_path = nwp_cloudcasting_path
    nwp_source = "cloudcasting"
    save_chunk_dict = {
        "step": -1,
        "x_geostationary": 100,
        "y_geostationary": 100,
    }

    @override
    def process(self, ds: xr.Dataset) -> xr.Dataset:
        # The cloudcasting data needs no changes
        return ds
    
    @override
    def data_is_okay(self, ds: xr.Dataset) -> bool:
        contains_nans = ds[list(ds.data_vars.keys())[0]].isnull().any().compute().item()
        return not contains_nans
