"""Functions and classes to download and process NWP data."""

import logging
import shutil
from abc import ABC, abstractmethod
from importlib.resources import files
from typing import override

import fsspec
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
import xesmf as xe
from ocf_data_sampler.config.load import load_yaml_configuration

from pvnet_app.data.utils import slice_to_pvnet_spatial_area

logger = logging.getLogger(__name__)


def download_data(source: str, destination: str) -> bool:
    """Download data from a source to a destination.

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
    ds_target_coords: xr.Dataset,
    method: str,
    nwp_source: str,
) -> xr.Dataset:
    """This function regrids the input NWP data to the grid of the target path.

    Args:
        ds: The NWP data to regrid
        ds_target_coords: The target grid dataset
        method: The regridding method to use
        nwp_source: The source of the NWP data (only used for logging messages)
    """
    # Check if regridding step needs to be done
    needs_regridding = not (
        ds.latitude.equals(ds_target_coords.latitude)
        and ds.longitude.equals(ds_target_coords.longitude)
    )

    if not needs_regridding:
        logger.info(f"No regridding required for {nwp_source} - skipping this step")
        return ds

    regridder = xe.Regridder(ds, ds_target_coords, method=method, unmapped_to_nan=True)

    return regridder(ds)


def check_model_nwp_inputs_available(
    data_config_filename: str,
    t0: pd.Timestamp,
    nwp_source: str,
    nwp_valid_times: pd.DatetimeIndex | None,
) -> bool:
    """Checks whether the model can be run given the available NWP data.

    Args:
        data_config_filename: Path to the data configuration file
        t0: The init-time of the forecast
        nwp_source: The NWP data source to check (e.g. "ukv", "ecmwf", "cloudcasting")
        nwp_valid_times: The valid times available in the NWP data

    Returns:
        bool: Whether the NWP timestamps satisfy that specified in the config
    """
    input_config = load_yaml_configuration(data_config_filename).input_data

    # Only check if using NWP data
    model_uses_nwp = (input_config.nwp is not None) and (
        nwp_source in [c.provider for _, c in input_config.nwp.items()]
    )

    if model_uses_nwp and (nwp_valid_times is None):
        available = False

    elif model_uses_nwp:
        nwp_config = next(c for _, c in input_config.nwp.items() if c.provider == nwp_source)

        # Get the NWP valid times required by the model
        freq = pd.Timedelta(f"{nwp_config.time_resolution_minutes}min")

        # ocf-data-sampler uses ceil to round up to the nearest timestep; match that here
        req_start_time = (t0 + pd.Timedelta(f"{nwp_config.interval_start_minutes}min")).ceil(freq)
        req_end_time = (t0 + pd.Timedelta(f"{nwp_config.interval_end_minutes}min")).ceil(freq)

        # If we diff accumulated channels in time we'll need one more timestamp
        if len(nwp_config.accum_channels) > 0:
            req_end_time = req_end_time + freq

        required_nwp_times = pd.date_range(start=req_start_time, end=req_end_time, freq=freq)

        # Check if any of the expected datetimes are missing
        missing_time_steps = np.setdiff1d(required_nwp_times, nwp_valid_times, assume_unique=True)

        available = len(missing_time_steps) == 0

        if len(missing_time_steps) > 0:
            logger.warning(f"Some {nwp_source} timesteps for {t0=} missing: \n{missing_time_steps}")

    else:
        available = True

    return available


class NWPDownloader(ABC):
    """Abstract base class to download and process NWP data."""

    nwp_source: str
    save_chunk_dict: dict[str, int]

    def __init__(
        self,
        source_path: str | None,
        destination_path: str,
        window_size_pixels: int | None = None,
    ) -> None:
        """Initialise the NWP downloader."""
        self.source_path = source_path
        self.destination_path = destination_path
        # Initially no valid times are available. This will only change is the data can be
        # downloaded, processed, and saved successfully
        self.valid_times = None
        self.window_size_pixels = window_size_pixels

    @abstractmethod
    def process(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply all processing steps to the NWP data in order to match the training data."""
        pass

    def data_is_okay(self, ds: xr.Dataset) -> bool:
        """Apply quality checks to the NWP data.

        Args:
            ds: The NWP data

        Returns:
            bool: Whether the data passes the quality checks
        """
        # ocf-data-sampler expects there to be only one variable and no NaNs
        vars = list(ds.data_vars)
        if len(vars) != 1:
            logger.warning(f"{self.nwp_source} data has unexpected variables: {vars}")
            return False
        else:
            contains_nans = ds[vars[0]].isnull().any().compute().item()

        return not contains_nans

    def resave(self, ds: xr.Dataset) -> None:
        """Resave the NWP data to the destination path."""
        # Overwrite the old data
        shutil.rmtree(self.destination_path, ignore_errors=True)

        ds["variable"] = ds["variable"].astype(str)

        # Clear old encoding
        for v in list(ds.variables.keys()):
            ds[v].encoding.clear()

        ds.chunk(self.save_chunk_dict).to_zarr(self.destination_path)

    def run(self) -> None:
        """Download, process, and save the NWP data."""
        logger.info(f"Downloading and processing the {self.nwp_source} data")

        if self.source_path is None:
            logger.warning(f"Source file for {self.nwp_source} is not set. Skipping download.")
            return

        file_exists = download_data(self.source_path, self.destination_path)
        if not file_exists:
            logger.warning(
                f"Source file {self.source_path} for {self.nwp_source} does not exist. "
                "Skipping download.",
            )
            return

        ds = xr.open_zarr(self.destination_path).compute()

        init_time = pd.to_datetime(ds.init_time.values[0])
        valid_times = init_time + pd.to_timedelta(ds.step)
        logger.info(
            f"{self.nwp_source} has init-time {init_time} and valid times: {valid_times}",
        )

        # Process the data to match the training data, then check the quality of the data, and
        # resave if it passes
        ds = self.process(ds)

        if self.data_is_okay(ds):
            self.resave(ds)

            # Only store the valid_times if the NWP data has been successfully downloaded,
            # quality checked, and processed. Else valid_times will be None
            self.valid_times = valid_times

        else:
            logger.warning(f"{self.nwp_source} NWP data did not pass quality checks.")

    def check_model_inputs_available(
        self,
        data_config_filename: str,
        t0: pd.Timestamp,
    ) -> bool:
        """Check if the NWP data the model needs is available.

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
    """Class to download and process the ECMWF data."""

    nwp_source = "ecmwf"
    save_chunk_dict = {  # noqa: RUF012
        "step": 10,
        "latitude": 50,
        "longitude": 50,
    }

    @override
    def process(self, ds: xr.Dataset) -> xr.Dataset:

        if self.window_size_pixels is not None:
            # Slice the data to the spatial extent used in PVNet
            ds = slice_to_pvnet_spatial_area(
                ds,
                width_pixels=self.window_size_pixels,
                height_pixels=self.window_size_pixels,
            )

        return ds


class UKVDownloader(NWPDownloader):
    """Class to download and process the UKV data."""

    nwp_source = "ukv"
    save_chunk_dict = {  # noqa: RUF012
        "step": 10,
        "x_osgb": 100,
        "y_osgb": 100,
    }

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        """Regrid the UKV data to the target grid.

        In production the UKV data is on a different grid structure to the training data. The
        training data from CEDA is on a regular OSGB grid. The production data is on a Lambert
        Azimuthal Equal Area grid
        """
        ds_target_coords = xr.load_dataset(
            files("pvnet_app.data").joinpath("nwp_ukv_target_coords.nc")
        )

        if self.window_size_pixels is not None:
            # Slice the data to the spatial extent used in PVNet
            ds_target_coords = slice_to_pvnet_spatial_area(
                ds_target_coords,
                width_pixels=self.window_size_pixels,
                height_pixels=self.window_size_pixels,
            )

        return regrid_nwp_data(
            ds=ds,
            ds_target_coords=ds_target_coords,
            method="bilinear",
            nwp_source="UKV",
        )

    @staticmethod
    def add_lon_lat_coords(ds: xr.Dataset) -> xr.Dataset:
        """Add latitude and longitude coords to the UKV data.

        The training UKV data is on a regular OSGB grid but the live data is on a Lambert Azimuthal
        Equal Area grid. We need to add longitudes and latitudes coords so we can regrid the data
        to the training grid.
        """
        # This is for nwp-consumer>=1.0.0
        logger.info("Adding lon-lat coords to the UKV data")

        # This is the Lambert Azimuthal Equal Area projection used in the UKV live data
        laea = pyproj.Proj(
            proj="laea",
            lat_0=54.9,
            lon_0=-2.5,
            x_0=0.0,
            y_0=0.0,
            ellps="WGS84",
            datum="WGS84",
        )

        # WGS84 is short for "World Geodetic System 1984". This is a lon-lat coord system
        wgs84 = pyproj.CRS("EPSG:4326")

        laea_to_lon_lat = pyproj.Transformer.from_crs(
            laea.crs,
            wgs84,
            always_xy=True,
        ).transform

        # Calculate longitude and latitude from x_laea and y_laea
        # - x is an array of shape (455,)
        # - y is an array of shape (639,)
        # We need to change x and y to a 2D arrays
        x_laea, y_laea = np.meshgrid(ds.x_laea, ds.y_laea)

        lons, lats = laea_to_lon_lat(xx=x_laea, yy=y_laea)

        ds = ds.assign_coords(
            longitude=(["y_laea", "x_laea"], lons),
            latitude=(["y_laea", "x_laea"], lats),
        )

        return ds

    @override
    def process(self, ds: xr.Dataset) -> xr.Dataset:
        ds = self.add_lon_lat_coords(ds)
        # The regrid step also slices the data to the spatial extent used in PVNet if
        # self.window_size_pixels is not None
        ds = self.regrid(ds)
        return ds


class CloudcastingDownloader(NWPDownloader):
    """Class to download and process the cloudcasting data."""

    nwp_source = "cloudcasting"
    save_chunk_dict: dict[str, int] = {  # noqa: RUF012
        "step": -1,
        "x_geostationary": 100,
        "y_geostationary": 100,
    }

    @override
    def process(self, ds: xr.Dataset) -> xr.Dataset:
        # The cloudcasting data needs no changes
        if self.window_size_pixels is not None:
            # Slice the data to the spatial extent used in PVNet
            ds = slice_to_pvnet_spatial_area(
                ds,
                width_pixels=self.window_size_pixels,
                height_pixels=self.window_size_pixels,
            )

        return ds
