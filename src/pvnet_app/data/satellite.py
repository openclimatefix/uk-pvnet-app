import logging
import os
import shutil

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from ocf_datapipes.config.load import load_yaml_configuration

from pvnet_app.consts import sat_path

logger = logging.getLogger(__name__)


def get_satellite_timestamps(zarr_path: str) -> pd.DatetimeIndex:
    """Get the datetimes of the satellite data at the given path

    Args:
        zarr_path: The path to the satellite zarr

    Returns:
        pd.DatetimeIndex: All available timestamps
    """
    ds = xr.open_zarr(zarr_path)
    return pd.to_datetime(ds.time.values)


def fill_1d_bool_gaps(x, max_gap):
    """In a boolean array, fill consecutive False elements if their number is less than the gap_size

    Args:
        x: A 1-dimensional boolean array
        max_gap: integer of the maximum gap size which will be filled with True

    Returns:
        A 1-dimensional boolean array

    Examples:
        >>> x = np.array([0, 1, 0, 0, 1, 0, 1, 0])
        >>> fill_1d_bool_gaps(x, max_gap=2).astype(int)
        array([0, 1, 1, 1, 1, 1, 1, 0])

        >>> x = np.array([1, 0, 0, 0, 1, 0, 1, 0])
        >>> fill_1d_bool_gaps(x, max_gap=2).astype(int)
        array([1, 0, 0, 0, 1, 1, 1, 0])
    """
    should_fill = np.zeros(len(x), dtype=bool)

    i_start = None

    last_b = False
    for i, b in enumerate(x):
        if last_b and not b:
            i_start = i
        elif b and not last_b and i_start is not None:
            if i - i_start <= max_gap:
                should_fill[i_start:i] = True
            i_start = None
        last_b = b

    return np.logical_or(should_fill, x)


def interpolate_missing_satellite_timestamps(ds: xr.Dataset, max_gap: pd.Timedelta) -> xr.Dataset:
    """Linearly interpolate missing satellite timestamps

    The max gap is inclusive of timestamps either side. E.g. if max gap is 15 minutes and the  
    satellite includes timestamps 12:00 and 12:15, then 12:05 and 12:10 will be filled. If the max 
    gap was 10 minutes, then none of the timestamps would be filled. A max gap if 5 minutes will do 
    nothing since the normal spacing is already 5 minutes.
    
    Args:
        ds: The satellite data
        max_gap: The maximum gap size which will be filled via interpolation. 
    """

    # If any of these times are missing, we will try to interpolate them
    dense_times = pd.date_range(ds.time.values.min(), ds.time.values.max(), freq="5min")

    # Create mask array of which timestamps are available
    timestamp_available = np.isin(dense_times, ds.time)

    # If all the requested times are present we avoid running interpolation
    if timestamp_available.all():
        logger.info("No gaps in the available satllite sequence - no interpolation run")
        return ds

    # If less than 2 of the buffer requested times are present we cannot infill
    elif timestamp_available.sum() < 2:
        logger.warning("Cannot run interpolate infilling with less than 2 time steps available")
        return ds

    else:
        logger.info("Some requested times are missing - running interpolation")

        # Run the interpolation to all 5-minute timestamps between the first and last
        ds_interp = ds.interp(time=dense_times, method="linear", assume_sorted=True)

        # Find the timestamps which are within max gap size
        max_gap_steps = int(max_gap / pd.Timedelta("5min")) - 1
        valid_fill_times = fill_1d_bool_gaps(timestamp_available, max_gap_steps)

        # Mask the timestamps outside the max gap size
        valid_fill_times_xr = xr.zeros_like(ds_interp.time, dtype=bool)
        valid_fill_times_xr.values[:] = valid_fill_times
        ds_sat_filtered = ds_interp.where(valid_fill_times_xr, drop=True)

        time_was_filled = np.logical_and(valid_fill_times_xr, ~timestamp_available)

        if time_was_filled.any():
            infilled_times = time_was_filled.where(time_was_filled, drop=True)
            logger.info(
                "The following times were filled by interpolation:\n"
                f"{infilled_times.time.values}",
            )

        if not valid_fill_times_xr.all():
            not_infilled_times = valid_fill_times_xr.where(~valid_fill_times_xr, drop=True)
            logger.info(
                "After interpolation the following times are still missing:\n"
                f"{not_infilled_times.time.values}",
            )

        return ds_sat_filtered


def extend_satellite_data_with_nans(
    ds: xr.Dataset, 
    t0: pd.Timestamp, 
    limit: pd.Timedelta = pd.Timedelta("3h"),
) -> xr.Dataset:
    """Fill missing satellite timestamps with NaNs

    The satellite data is filled with NaNs after its last avilable timestamp. The data is
    extended forwards in time either up to t0 or up to the limit, whichever is smaller.

    Args:
        t0: The init-time of the forecast
        limit: The maximum time to extend the data with NaNs
    """
    # Find how delayed the satellite data is
    sat_max_time = pd.to_datetime(ds.time).max()
    delay = t0 - sat_max_time

    if delay > pd.Timedelta(0):
        logger.info(f"Filling most recent {delay} with NaNs")

        if delay > limit:
            logger.warning(
                f"The satellite data is delayed by more than {limit}. "
                f"Will only infill {limit} forward from the latest satellite timestamp.",
            )
        fill_timedelta = min(delay, limit)

        # We will fill the data with NaNs for these timestamps
        fill_times = pd.date_range(
            sat_max_time + pd.Timedelta("5min"), 
            sat_max_time+fill_timedelta, 
            freq="5min"
        )

        # Extend the data with NaNs
        ds = ds.reindex(time=np.concatenate([ds.time, fill_times]), fill_value=np.nan)
    
    return ds


def check_model_satellite_inputs_available(
    data_config_filename: str,
    t0: pd.Timestamp,
    sat_datetimes: pd.DatetimeIndex | None,
) -> bool:
    """Checks whether the model can be run given the current satellite delay

    Args:
        data_config_filename: Path to the data configuration file
        t0: The init-time of the forecast
        available_sat_datetimes: The available satellite timestamps

    Returns:
        bool: Whether the satellite data satisfies that specified in the config
    """
    input_config = load_yaml_configuration(data_config_filename).input_data

    available = True

    # Only check if using satellite data
    model_uses_satellite = (
        hasattr(input_config, "satellite") 
        and (input_config.satellite is not None)
    )

    # In case the model does not require satellite
    if not model_uses_satellite:
        available = True

    # In case the model requires satellite but none is available
    elif model_uses_satellite and (sat_datetimes is None):
        available = False

    # In case the model requires satellite and some is available
    elif model_uses_satellite:

        # Take into account how recently the model tries to slice satellite data from
        max_sat_delay_allowed_mins = input_config.satellite.live_delay_minutes

        # Take into account the dropout the model was trained with, if any
        if input_config.satellite.dropout_fraction > 0:
            max_sat_delay_allowed_mins = max(
                max_sat_delay_allowed_mins,
                np.abs(input_config.satellite.dropout_timedeltas_minutes).max(),
            )

        # Get all expected datetimes
        history_minutes = input_config.satellite.history_minutes

        expected_datetimes = pd.date_range(
            t0 - pd.Timedelta(f"{int(history_minutes)}min"),
            t0 - pd.Timedelta(f"{int(max_sat_delay_allowed_mins)}min"),
            freq="5min",
        )

        # Check if any of the expected datetimes are missing
        missing_time_steps = np.setdiff1d(expected_datetimes, sat_datetimes, assume_unique=True)

        available = len(missing_time_steps) == 0

        if len(missing_time_steps) > 0:
            logger.info(f"Some satellite timesteps for {t0=} missing: \n{missing_time_steps}")

    return available


def contains_too_many_of_value(
    ds: xr.Dataset, 
    value: float, 
    threshold: float,
) -> bool:
    """Check if the input data contains more than a certain fraction of a given value.

    Args:
        ds: The satellite data
        value: The value to check for
        threshold: The maximum fraction of the value allowed
    """

    logger.info(f"Checking satellite data for value ({value})")

    too_many_values = False

    # We will calculate fractional ocurence for each time
    reduction_dims = set(ds.data.dims) - {"time"}
    if np.isnan(value):
        # np.nan != np.nan, so we have to use isnan
        fraction_values = np.isnan(ds.data).mean(dim=reduction_dims)
    else:
        fraction_values = (ds.data == value).mean(dim=reduction_dims)

    if fraction_values.max() > threshold:
        logger.warning(
            f"Satellite data contains values {value} greater than {threshold:.2%} of the time"
            f"{fraction_values.to_series().to_string()}"
        )

        too_many_values=True

    return too_many_values


def scale_satellite_data(ds: xr.Dataset) -> xr.Dataset:
    """Scale the satellite data to be between 0 and 1.

    The production satellite data is scaled between 0 and 1023. This function scales it to be 
    between 0 and 1 as in the training data.
    
    Args:
        ds: The satellite data
    """
    return ds / 1023


class SatelliteDownloader:

    destination_path_5: str = "sat_5_min.zarr.zip"
    destination_path_15: str = "sat_15_min.zarr.zip"
    destination_path: str = sat_path

    def __init__(self, 
        t0: pd.Timestamp, 
        source_path_5: str | None, 
        source_path_15: str | None,  
        legacy: bool = False
    ):
        self.t0 = t0
        self.source_path_5 = source_path_5
        self.source_path_15 = source_path_15
        self.legacy = legacy
        self.valid_times = None

    def download_data(self) -> bool:
        """Download the sat data if available and return whether it was successful

        Returns:
            bool: Whether satellite data was available to download
        """

        # Set variable to track whether the satellite download is successful
        data_available = False

        # Download 5 minute satellite data if it exists
        if self.source_path_5 is not None:
            fs = fsspec.open(self.source_path_5).fs
            if fs.exists(self.source_path_5):
                logger.info("Downloading 5-minute satellite data")
                fs.get(self.source_path_5, self.destination_path_5)
                data_available = True
            else:
                logger.info("No 5-minute data available")

        # Also download 15-minute satellite if it exists
        if self.source_path_15 is not None:
            fs = fsspec.open(self.source_path_15).fs
            if fs.exists(self.source_path_15):
                logger.info("Downloading 15-minute satellite data")
                fs.get(self.source_path_15, self.destination_path_15)
                data_available = True
            else:
                logger.info("No 15-minute data available")

        return data_available
    

    def choose_and_load_satellite_data(self) -> xr.Dataset:
        """Select from the 5 and 15-minutely satellite data for the most recent data"""
        # Check which satellite data exists
        exists_5_minute = os.path.exists(self.destination_path_5)
        exists_15_minute = os.path.exists(self.destination_path_15)

        if not exists_5_minute and not exists_15_minute:
            raise FileNotFoundError("Neither 5- nor 15-minutely data was found.")

        # Find the delay in the 5- and 15-minutely data
        if exists_5_minute:
            datetimes_5min = get_satellite_timestamps(self.destination_path_5)
            logger.info(
                f"Latest 5-minute timestamp is {datetimes_5min.max()}. "
                f"All the datetimes are: \n{datetimes_5min}",
            )
        else:
            logger.info("No 5-minute data was found.")

        if exists_15_minute:
            datetimes_15min = get_satellite_timestamps(self.destination_path_15)
            logger.info(
                f"Latest 15-minute timestamp is {datetimes_15min.max()}. "
                f"All the datetimes are: \n{datetimes_15min}",
            )
        else:
            logger.info("No 15-minute data was found.")

        # If both 5- and 15-minute data exists, use the most recent
        if exists_5_minute and exists_15_minute:
            use_5_minute = datetimes_5min.max() > datetimes_15min.max()
        else:
            # If only one exists, use that
            use_5_minute = exists_5_minute

        # Move the selected data to the expected path
        if use_5_minute:
            logger.info("Using 5-minutely data.")
            ds = xr.open_zarr(self.destination_path_5).compute()
        else:
            logger.info("Using 15-minutely data.")
            ds = xr.open_zarr(self.destination_path_15).compute()
        
        return ds
    
    @staticmethod
    def data_is_okay(ds: xr.Dataset) -> bool:
        """Apply quality checks to the satellite data
        
        Args:
            ds: The satellite data

        Returns:
            bool: Whether the data passes the quality checks
        """
        too_many_nans = contains_too_many_of_value(ds, value=np.nan, threshold=0.1)
        # Note that in the UK, even at night, the values are not zero
        too_many_zeros = contains_too_many_of_value(ds, value=0, threshold=0.1)
        return (not too_many_nans) and (not too_many_zeros)
        
    
    def process(self, ds: xr.Dataset) -> xr.Dataset:
        """"Apply all processing steps to the satellite data in order to match the training data

        Args:
            ds: The satellite data

        Returns:
            xr.Dataset: The processed satellite data
        """

        # Interpolate missing satellite timestamps
        ds = interpolate_missing_satellite_timestamps(ds, max_gap=pd.Timedelta("15min"))

        if not self.legacy:
            # Scale the satellite data if not legacy. The legacy dataloader does production data 
            # scaling inside it. The new dataloader does not
            ds = scale_satellite_data(ds)

        # Store the available satellite timestamps before we extend with NaNs
        self.valid_times = pd.to_datetime(ds.time.values)

        # Extend the satellite data with NaNs if needed by the model and record the delay of most 
        # recent non-nan timestamp
        ds = extend_satellite_data_with_nans(ds, t0=self.t0)

        return ds


    def resave(self, ds: xr.Dataset) -> None:
        """Resave the satellite data to the destination path"""

        ds["variable"] = ds["variable"].astype(str)
        
        # Overwrite the old data
        shutil.rmtree(self.destination_path, ignore_errors=True)

        save_chunk_dict = {
            "x_geostationary": 100,
            "y_geostationary": 100,
            "time": 6,
            "variable": -1,
        }

        # Clear old encoding
        for v in list(ds.variables.keys()):
            ds[v].encoding.clear()

        ds.chunk(save_chunk_dict).to_zarr(self.destination_path)


    def run(self) -> None:
        """Download, process, and save the satellite data"""

        logger.info(f"Downloading and processing the satellite data")
        data_available = self.download_data()

        if not data_available:
            logger.warning("No satellite data available for download")
            return
        
        # Select the most recent satellite data and load it into memory
        ds = self.choose_and_load_satellite_data()
        
        if self.data_is_okay(ds):
            ds = self.process(ds).compute()
            self.resave(ds)

        else:
            logger.warning("Satellite data did not pass quality checks.")

    
    def check_model_inputs_available(
        self,
        data_config_filename: str,
        t0: pd.Timestamp,
    ) -> bool:
        """Check if the satellite data the model needs is available
        
        Args:
            data_config_filename: The path to the data configuration file
            t0: The init-time of the forecast
        """
        return check_model_satellite_inputs_available(
            data_config_filename=data_config_filename,
            t0=t0,
            sat_datetimes=self.valid_times,
        ) 
    
    def clean_up(self) -> None:
        """Remove the downloaded data"""
        for path in [self.destination_path, self.destination_path_5, self.destination_path_15]:
            shutil.rmtree(path, ignore_errors=True)