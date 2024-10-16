import numpy as np
import pandas as pd
import xarray as xr
import logging
from typing import Optional
import os
import fsspec
import ocf_blosc2
from ocf_datapipes.config.load import load_yaml_configuration

from pvnet_app.consts import sat_path

logger = logging.getLogger(__name__)

sat_5_path = "sat_5_min.zarr"
sat_15_path = "sat_15_min.zarr"


def download_all_sat_data() -> bool:
    """Download the sat data and return whether it was successful

    Returns:
        bool: Whether the download was successful
    """

    # Clean out old files
    os.system(f"rm -r {sat_path} {sat_5_path} {sat_15_path}")

    # Set variable to track whether the satellite download is successful
    sat_available = False
    if "SATELLITE_ZARR_PATH" not in os.environ:
        logger.info("SATELLITE_ZARR_PATH has not be set. " "No satellite data will be downloaded.")
        return False

    # download 5 minute satellite data
    sat_5_dl_path = os.environ["SATELLITE_ZARR_PATH"]
    fs = fsspec.open(sat_5_dl_path).fs
    if fs.exists(sat_5_dl_path):
        sat_available = True
        logger.info(f"Downloading 5-minute satellite data")
        fs.get(sat_5_dl_path, "sat_5_min.zarr.zip")
        os.system(f"unzip -qq sat_5_min.zarr.zip -d {sat_5_path}")
        os.system(f"rm sat_5_min.zarr.zip")
    else:
        logger.info(f"No 5-minute data available")

    # Also download 15-minute satellite if it exists
    sat_15_dl_path = os.environ["SATELLITE_ZARR_PATH"].replace(".zarr", "_15.zarr")
    if fs.exists(sat_15_dl_path):
        sat_available = True
        logger.info(f"Downloading 15-minute satellite data")
        fs.get(sat_15_dl_path, "sat_15_min.zarr.zip")
        os.system(f"unzip -qq sat_15_min.zarr.zip -d {sat_15_path}")
        os.system(f"rm sat_15_min.zarr.zip")
    else:
        logger.info(f"No 15-minute data available")

    return sat_available


def get_satellite_timestamps(sat_zarr_path: str) -> pd.DatetimeIndex:
    """Get the datetimes of the satellite data

    Args:
        sat_zarr_path: The path to the satellite zarr

    Returns:
        pd.DatetimeIndex: All available satellite timestamps
    """
    ds_sat = xr.open_zarr(sat_zarr_path)
    return pd.to_datetime(ds_sat.time.values)


def combine_5_and_15_sat_data() -> None:
    """Select and/or combine the 5 and 15-minutely satellite data and move it to the expected path"""

    # Check which satellite data exists
    exists_5_minute = os.path.exists(sat_5_path)
    exists_15_minute = os.path.exists(sat_15_path)

    if not exists_5_minute and not exists_15_minute:
        raise FileNotFoundError("Neither 5- nor 15-minutely data was found.")

    # Find the delay in the 5- and 15-minutely data
    if exists_5_minute:
        datetimes_5min = get_satellite_timestamps(sat_5_path)
        logger.info(
            f"Latest 5-minute timestamp is {datetimes_5min.max()}. "
            f"All the datetimes are: \n{datetimes_5min}"
        )
    else:
        logger.info("No 5-minute data was found.")

    if exists_15_minute:
        datetimes_15min = get_satellite_timestamps(sat_15_path)
        logger.info(
            f"Latest 5-minute timestamp is {datetimes_15min.max()}. "
            f"All the datetimes are: \n{datetimes_15min}"
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
        logger.info(f"Using 5-minutely data.")
        os.system(f"mv {sat_5_path} {sat_path}")
    else:
        logger.info(f"Using 15-minutely data.")
        os.system(f"mv {sat_15_path} {sat_path}")


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


def interpolate_missing_satellite_timestamps(max_gap: pd.Timedelta) -> None:
    """Interpolate missing satellite timestamps"""

    ds_sat = xr.open_zarr(sat_path)

    # If any of these times are missing, we will try to interpolate them
    dense_times = pd.date_range(
        ds_sat.time.values.min(),
        ds_sat.time.values.max(),
        freq="5min",
    )

    # Create mask array of which timestamps are available
    timestamp_available = np.isin(dense_times, ds_sat.time)

    # If all the requested times are present we avoid running interpolation
    if timestamp_available.all():
        logger.warning("No gaps in the available satllite sequence - no interpolation run")
        return

    # If less than 2 of the buffer requested times are present we cannot infill
    elif timestamp_available.sum() < 2:
        logger.warning("Cannot run interpolate infilling with less than 2 time steps available")
        return

    else:
        logger.info("Some requested times are missing - running interpolation")

        # Compute before interpolation for efficiency
        ds_sat = ds_sat.compute()

        # Run the interpolation to all 5-minute timestamps between the first and last
        ds_interp = ds_sat.interp(time=dense_times, method="linear", assume_sorted=True)

        # Find the timestamps which are within max gap size
        max_gap_steps = int(max_gap / pd.Timedelta("5min")) - 1
        valid_fill_times = fill_1d_bool_gaps(timestamp_available, max_gap_steps)

        # Mask the timestamps outside the max gap size
        valid_fill_times_xr = xr.zeros_like(ds_interp.time, dtype=bool)
        valid_fill_times_xr.values[:] = valid_fill_times
        ds_sat = ds_interp.where(valid_fill_times_xr)

        time_was_filled = np.logical_and(valid_fill_times_xr, ~timestamp_available)

        if time_was_filled.any():
            infilled_times = time_was_filled.where(time_was_filled, drop=True)
            logger.info(
                "The following times were filled by interpolation:\n"
                f"{infilled_times.time.values}"
            )

        if not valid_fill_times_xr.all():
            not_infilled_times = valid_fill_times_xr.where(~valid_fill_times_xr, drop=True)
            logger.info(
                "After interpolation the following times are still missing:\n"
                f"{not_infilled_times.time.values}"
            )

        # Save the interpolated data
        os.system(f"rm -rf {sat_path}")
        ds_sat.to_zarr(sat_path)


def extend_satellite_data_with_nans(
    t0: pd.Timestamp, satellite_data_path: Optional[str] = sat_path
) -> None:
    """Fill the satellite data with NaNs out to time t0

    Args:
        t0: The init-time of the forecast
    """

    # Find how delayed the satellite data is
    ds_sat = xr.open_zarr(satellite_data_path)
    sat_max_time = pd.to_datetime(ds_sat.time).max()
    delay = t0 - sat_max_time

    if delay > pd.Timedelta(0):
        logger.info(f"Filling most recent {delay} with NaNs")

        if delay > pd.Timedelta("3h"):
            logger.warning(
                "The satellite data is delayed by more than 3 hours. "
                "Will only infill last 3 hours."
            )
            delay = pd.Timedelta("3h")

        # Load into memory so we can delete it on disk
        ds_sat = ds_sat.compute()

        # We will fill the data with NaNs for these timestamps
        fill_times = pd.date_range(t0 - delay + pd.Timedelta("5min"), t0, freq="5min")

        # Extend the data with NaNs
        ds_sat = ds_sat.reindex(time=np.concatenate([ds_sat.time, fill_times]), fill_value=np.nan)

        # Re-save inplace
        os.system(f"rm -rf {satellite_data_path}")
        ds_sat.to_zarr(satellite_data_path)


def check_model_satellite_inputs_available(
    data_config_filename: str,
    t0: pd.Timestamp,
    sat_datetimes: pd.DatetimeIndex,
) -> bool:
    """Checks whether the model can be run given the current satellite delay

    Args:
        data_config_filename: Path to the data configuration file
        t0: The init-time of the forecast
        available_sat_datetimes: The available satellite timestamps

    Returns:
        bool: Whether the satellite data satisfies that specified in the config
    """

    data_config = load_yaml_configuration(data_config_filename)

    available = True

    # check satellite if using
    if hasattr(data_config.input_data, "satellite"):
        if data_config.input_data.satellite:

            # Take into account how recently the model tries to slice satellite data from
            max_sat_delay_allowed_mins = data_config.input_data.satellite.live_delay_minutes

            # Take into account the dropout the model was trained with, if any
            if data_config.input_data.satellite.dropout_fraction > 0:
                max_sat_delay_allowed_mins = max(
                    max_sat_delay_allowed_mins,
                    np.abs(data_config.input_data.satellite.dropout_timedeltas_minutes).max(),
                )

            # Get all expected datetimes
            history_minutes = data_config.input_data.satellite.history_minutes

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


def preprocess_sat_data(t0: pd.Timestamp, use_legacy: bool = False) -> pd.DatetimeIndex:
    """Combine and 5- and 15-minutely satellite data and extend to t0 if required

    Args:
        t0: The init-time of the forecast
        use_legacy: Whether to prepare the data as required for the legacy dataloader

    Returns:
        pd.DatetimeIndex: The available satellite timestamps
        int: The spacing between data samples in minutes
    """

    # Deal with switching between the 5 and 15 minutely satellite data
    combine_5_and_15_sat_data()

    # Interpolate missing satellite timestamps
    interpolate_missing_satellite_timestamps(pd.Timedelta("15min"))

    if not use_legacy:
        # scale the satellite data if not legacy. The legacy dataloader does production data scaling
        # inside it. The new dataloader does not
        scale_satellite_data()

    # Get the available satellite timestamps before we extend with NaNs
    sat_timestamps = get_satellite_timestamps(sat_path)

    # Extend the satellite data with NaNs if needed by the model and record the delay of most recent
    # non-nan timestamp
    extend_satellite_data_with_nans(t0)

    return sat_timestamps


def scale_satellite_data() -> None:
    """Scale the satellite data to be between 0 and 1"""

    ds_sat = xr.open_zarr(sat_path)
    ds_sat = ds_sat / 1024
    ds_sat = ds_sat.compute()

    # save
    os.system(f"rm -rf {sat_path}")
    ds_sat.to_zarr(sat_path)
