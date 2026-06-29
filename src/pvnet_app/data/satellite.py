"""Functions to download and process satellite data."""
import logging
import shutil

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.load.utils import make_spatial_coords_increasing
from ocf_data_sampler.select.geospatial import convert_coordinates
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.select_spatial_slice import select_spatial_slice_pixels_multiple

from pvnet_app.data.gsp import get_gsp_locations

logger = logging.getLogger(__name__)


def open_satellite_data(s3_icechunk_path: str, region: str) -> xr.Dataset | None:
    """Open the satellite data from the given s3 icechunk path.

    Args:
        s3_icechunk_path: The s3 path to the icechunk containing the satellite
        region: The s3 region where the icechunk is stored
    """
    bucket, _, path = s3_icechunk_path.removeprefix("s3://").partition("/")

    store = icechunk.s3_storage(
        bucket=bucket,
        prefix=path,
        from_env=True,
        region=region,
    )

    try:
        repo = icechunk.Repository.open(store)
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store)
    except icechunk.IcechunkError as e:
        logger.error(f"Error opening icechunk repository: {e}")
        ds = None

    return ds


def fill_1d_bool_gaps(x: np.array, max_gap: int) -> np.array:
    """Fill consecutive False elements if their number is less than the gap_size.

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
    """Linearly interpolate missing satellite timestamps.

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
                f"The following times were filled by interpolation:\n{infilled_times.time.values}",
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
    limit: pd.Timedelta | None = None,
) -> xr.Dataset:
    """Fill missing satellite timestamps with NaNs.

    The satellite data is filled with NaNs after its last avilable timestamp. The data is
    extended forwards in time either up to t0 or up to the limit, whichever is smaller.

    Args:
        ds: The satellite data
        t0: The init-time of the forecast
        limit: The maximum time to extend the data with NaNs
    """
    # Find how delayed the satellite data is
    sat_max_time = pd.to_datetime(ds.time).max()
    delay = t0 - sat_max_time

    if limit is None:
        limit = pd.Timedelta("3h")

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
            sat_max_time + fill_timedelta,
            freq="5min",
        )

        # Extend the data with NaNs
        ds = ds.reindex(time=np.concatenate([ds.time, fill_times]), fill_value=np.nan)

    return ds


def check_model_satellite_inputs_available(
    data_config_filename: str,
    t0: pd.Timestamp,
    sat_datetimes: pd.DatetimeIndex | None,
) -> bool:
    """Checks whether the model can be run given the current satellite delay.

    Args:
        data_config_filename: Path to the data configuration file
        t0: The init-time of the forecast
        sat_datetimes: The available satellite timestamps

    Returns:
        bool: Whether the satellite data satisfies that specified in the config
    """
    input_config = load_yaml_configuration(data_config_filename).input_data

    available = True

    # Only check if using satellite data
    model_uses_satellite = hasattr(input_config, "satellite") and (
        input_config.satellite is not None
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
        # interval_[start/end]_minutes is relative to t0 so negative means before t0
        interval_start_minutes = input_config.satellite.interval_start_minutes
        interval_end_minutes = input_config.satellite.interval_end_minutes

        # Take into account the dropout the model was trained with
        # If the model was trained with dropout, we can allow the satellite data to be
        # delayed by most negative dropout time
        if input_config.satellite.dropout_fraction > 0:
            interval_end_minutes = min(
                interval_end_minutes,
                np.array(input_config.satellite.dropout_timedeltas_minutes).min(),
            )

        expected_datetimes = pd.date_range(
            t0 + pd.Timedelta(f"{int(interval_start_minutes)}min"),
            t0 + pd.Timedelta(f"{int(interval_end_minutes)}min"),
            freq="5min",
        )

        # Check if any of the expected datetimes are missing
        missing_time_steps = np.setdiff1d(expected_datetimes, sat_datetimes, assume_unique=True)

        available = len(missing_time_steps) == 0

        if len(missing_time_steps) > 0:
            logger.info(f"Some satellite timesteps for {t0=} missing: \n{missing_time_steps}")

    return available


def get_pvnet_satellite_spatial_bounds(
    ds: xr.Dataset,
    width_pixels: int = 24,
    height_pixels: int = 24,
) -> xr.Dataset:
    """Get the spatial extent of the satellite data used in PVNet.

    Args:
        ds: The satellite data
        width_pixels: The width of the spatial slice in pixels
        height_pixels: The height of the spatial slice in pixels

    Returns:
        xr.Dataset: The spatial slice of the dataset used by PVNet
    """
    # Cut down the slice for efficiency and reorder the coordinates if needed
    ds = make_spatial_coords_increasing(
        ds,
        x_coord="x_geostationary",
        y_coord="y_geostationary",
    )

    # We will loop over all the GSP locations and find the min and max x and y coordinates
    # This gives us a bounding box used by PVNet
    df_locs = get_gsp_locations().loc[1:]

    geo_xs, geo_ys = convert_coordinates(
        x=df_locs.longitude.values,
        y=df_locs.latitude.values,
        from_coords="lon_lat",
        target_coords="geostationary",
        area_string=str(ds.attrs["area"]),
    )

    # Add the projection to the locations objects
    locations = []
    for x, y, loc_id in zip(geo_xs, geo_ys, df_locs.index.values, strict=True):
        locations.append(Location(x=x, y=y, coord_system="geostationary", id=loc_id))

    return select_spatial_slice_pixels_multiple(ds, locations, width_pixels, height_pixels)


def contains_too_many_of_value(ds: xr.Dataset, value: float, threshold: float) -> bool:
    """Check if the input data contains more than a certain fraction of a given value.

    Args:
        ds: The satellite data
        value: The value to check for
        threshold: The maximum fraction of the value allowed
    """
    logger.info(f"Checking satellite data for value ({value})")

    # We calculate fraction for each time
    reduction_dims = set(ds.data.dims) - {"time"}
    if np.isnan(value):
        # np.nan != np.nan, so we have to use isnan
        fraction_values = np.isnan(ds.data).mean(dim=reduction_dims)
    else:
        fraction_values = (ds.data == value).mean(dim=reduction_dims)

    exceeds_threshold = fraction_values.values.max() > threshold

    if exceeds_threshold:
        logger.warning(
            f"Satellite data contains values {value} greater than {threshold:.2%} of the time"
            f"{fraction_values.to_series().to_string()}",
        )

    return exceeds_threshold


class SatelliteDownloader:
    """Class to download and process satellite data."""

    def __init__(
        self,
        t0: pd.Timestamp,
        source_path_5: str | None,
        source_path_15: str | None,
        s3_region: str,
        destination_path: str,
    ) -> None:
        """Class to download and process satellite data."""
        self.t0 = t0
        self.source_path_5 = source_path_5
        self.source_path_15 = source_path_15
        self.s3_region = s3_region
        self.time_window = pd.Timedelta("1h")
        self.valid_times = None
        self.sat_choice = None
        self.destination_path = destination_path

    @staticmethod
    def data_is_okay(ds: xr.Dataset) -> bool:
        """Apply quality checks to the satellite data.

        Args:
            ds: The satellite data

        Returns:
            bool: Whether the data passes the quality checks
        """
        # Slice the data to the spatial extent used in PVNet
        ds = get_pvnet_satellite_spatial_bounds(ds)

        too_many_nans = contains_too_many_of_value(ds, value=np.nan, threshold=0.05)

        return (not too_many_nans)

    def process(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply all processing steps to the satellite data in order to match the training data.

        Args:
            ds: The satellite data

        Returns:
            xr.Dataset: The processed satellite data
        """
        # Filter out unused variables
        ds = ds[["data"]]

        # Interpolate missing satellite timestamps
        ds = interpolate_missing_satellite_timestamps(ds, max_gap=pd.Timedelta("15min"))

        # Store the available satellite timestamps before we extend with NaNs
        self.valid_times = pd.to_datetime(ds.time.values)

        # Extend the satellite data with NaNs if needed by the model and record the delay of most
        # recent non-nan timestamp
        ds = extend_satellite_data_with_nans(ds, t0=self.t0)

        # Add the top level area attribute to the data var(s). This is needed by ocf_data_sampler
        for v in list(ds.data_vars.keys()):
            ds[v].attrs["area"] = str(ds.attrs["area"])

        return ds

    def resave(self, ds: xr.Dataset) -> None:
        """Resave the satellite data to the destination path."""
        # Overwrite the old data
        shutil.rmtree(self.destination_path, ignore_errors=True)

        save_chunk_dict = {
            "x_geostationary": 100,
            "y_geostationary": 100,
            "time": 6,
            "channel": -1,
        }

        # Clear old encoding
        for v in list(ds.variables.keys()):
            ds[v].encoding.clear()

        ds.chunk(save_chunk_dict).to_zarr(self.destination_path)

    def run(self) -> None:
        """Download, process, and save the satellite data."""
        logger.info("Downloading and processing the satellite data")

        ds_dict = {}

        for path, label in [(self.source_path_5, "5-min"), (self.source_path_15, "15-min")]:

            if path is not None:
                ds = open_satellite_data(
                    s3_icechunk_path=path,
                    region=self.s3_region,
                )

                if ds is not None:
                    ds_dict[label] = ds
                    logger.info(
                        f"{label} satellite data contains times:"
                        f"\n...\n{ds_dict[label].time.values[-24:]}",
                    )

        if not ds_dict:
            logger.warning("No satellite data available from either source")
            return

        # Select the source with the most recent data, and use 5-minute data if equal recency
        best_source = max(ds_dict, key=lambda k: (ds_dict[k].time.max(), k=="5-min"))
        self.sat_choice = best_source
        logger.info(f"Using {best_source} satellite data")

        # Slice and load into memory for processing
        ds = (
            ds_dict[best_source]
            .sortby("time")
            .drop_duplicates("time", keep="last")
            .sel(time=slice(self.t0 - self.time_window, self.t0))
            .load()
        )

        if self.data_is_okay(ds):
            ds = self.process(ds)
            self.resave(ds)

        else:
            logger.warning("Satellite data did not pass quality checks.")

    def check_model_inputs_available(
        self,
        data_config_filename: str,
        t0: pd.Timestamp,
    ) -> bool:
        """Check if the satellite data the model needs is available.

        Args:
            data_config_filename: The path to the data configuration file
            t0: The init-time of the forecast
        """
        return check_model_satellite_inputs_available(
            data_config_filename=data_config_filename,
            t0=t0,
            sat_datetimes=self.valid_times,
        )

