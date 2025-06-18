import os
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from pvnet_app.consts import sat_path
from pvnet_app.data.satellite import (
    SatelliteDownloader,
    check_model_satellite_inputs_available,
    contains_too_many_of_value,
    extend_satellite_data_with_nans,
    interpolate_missing_satellite_timestamps,
)

# ------------------------------------------------------------
# Utility functions for the tests


def save_to_zarr_zip(ds: xr.Dataset, filename: str) -> None:
    """Save the given xarray dataset to a zarr file in a zip archive

    Args:
        ds: Dataset to save
        filename: Name of the zip archive
    """
    with zarr.ZipStore(filename) as store:
        ds.to_zarr(store, compute=True)


def timesteps_match_expected_freq(
    sat_path: str,
    expected_freq_mins: int | list[int],
) -> bool:
    """Check that the satellite data at the given path has the expected frequency of timesteps.

    Args:
        sat_path: Path to the satellite data
        expected_freq_mins: Expected frequency of timesteps in minutes
    """
    ds_sat = xr.open_zarr(sat_path)

    if not isinstance(expected_freq_mins, list):
        expected_freq_mins = [expected_freq_mins]

    dts = pd.to_datetime(ds_sat.time).diff()[1:]
    return np.isin(dts, pd.to_timedelta(expected_freq_mins, unit="min")).all()


# ------------------------------------------------------------
# Tests begin here


def test_download_sat_5_data(sat_5_data, test_t0):
    """Download only the 5 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # Make 5-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.download_data()

        # Assert that the 5-minute file exists
        assert os.path.exists(sat_downloader.destination_path_5)
        assert not os.path.exists(sat_downloader.destination_path_15)

        # Check the satellite data is 5-minutely
        assert timesteps_match_expected_freq(
            sat_downloader.destination_path_5,
            expected_freq_mins=5,
        )


def test_download_sat_15_data(sat_15_data, test_t0):
    """Download only the 15 minute satellite data"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.download_data()

        # Assert that the 15-minute file exists
        assert not os.path.exists(sat_downloader.destination_path_5)
        assert os.path.exists(sat_downloader.destination_path_15)

        # Check the satellite data is 15-minutely
        assert timesteps_match_expected_freq(
            sat_downloader.destination_path_15,
            expected_freq_mins=15,
        )


def test_download_sat_5_and_15_data(sat_5_data, sat_15_data, test_t0):
    """Download 5 minute sat and 15 minute satellite data"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # Make 5- and 15-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.download_data()

        assert os.path.exists(sat_downloader.destination_path_5)
        assert os.path.exists(sat_downloader.destination_path_15)

        # Check this satellite data is 5-minutely
        assert timesteps_match_expected_freq(
            sat_downloader.destination_path_5,
            expected_freq_mins=5,
        )

        # Check this satellite data is 15-minutely
        assert timesteps_match_expected_freq(
            sat_downloader.destination_path_15,
            expected_freq_mins=15,
        )


def test_run_sat_5_data(sat_5_data, test_t0):
    """Download and process only the 5 minute satellite data"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # Make 5-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.run()

        # Check the satellite data is 5-minutely and is saved in the correct place
        assert timesteps_match_expected_freq(sat_path, expected_freq_mins=5)


def test_run_sat_15_data(sat_15_data, test_t0):
    """Download and process only the 15 minute satellite data"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.run()

        assert not os.path.exists(sat_downloader.destination_path_5)
        assert os.path.exists(sat_downloader.destination_path_15)
        assert os.path.exists(sat_downloader.destination_path)

        # We infill the satellite data to 5 minutes in the process step
        assert timesteps_match_expected_freq(sat_path, expected_freq_mins=5)


def test_run_sat_delayed_5_and_15_data(sat_5_data_delayed, sat_15_data, test_t0):
    """Download and process 5 and 15 minute satellite data. Use the 15 minute data since the
    5 minute data is too delayed
    """

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        save_to_zarr_zip(sat_5_data_delayed, filename="latest.zarr.zip")
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.run()

        # We infill the satellite data to 5 minutes in the process step
        assert timesteps_match_expected_freq(sat_path, expected_freq_mins=5)


def test_run_zeros_in_sat_data(sat_15_data, test_t0):
    """Check that the satellite data is considered invalid if it contains too many zeros"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # Make half the values zeros
        ds = sat_15_data.copy(deep=True)
        ds.data[::2] = 0

        # Make satellite data available
        save_to_zarr_zip(ds, filename="latest.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.run()

        # If the satellite data is invalid the valid_times attribute should be None
        assert sat_downloader.valid_times is None


def test_run_nan_in_sat_data(sat_15_data, test_t0):
    """Check that the satellite data is considered invalid if it contains too many NaNs"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # Make half the values zeros
        ds = sat_15_data.copy(deep=True)
        ds.data[::2] = np.nan

        # Make satellite data available
        save_to_zarr_zip(ds, filename="latest.zarr.zip")

        sat_downloader = SatelliteDownloader(
            t0=test_t0,
            source_path_5="latest.zarr.zip",
            source_path_15="latest_15.zarr.zip",
        )
        sat_downloader.run()

        # If the satellite data is invalid the valid_times attribute should be None
        assert sat_downloader.valid_times is None


def test_check_model_satellite_inputs_available(config_filename):

    t0 = pd.Timestamp("2023-01-01 00:00")
    sat_datetime_1 = pd.date_range(
        t0 - pd.Timedelta("120min"),
        t0 - pd.Timedelta("5min"),
        freq="5min",
    )
    sat_datetime_2 = pd.date_range(
        t0 - pd.Timedelta("120min"),
        t0 - pd.Timedelta("15min"),
        freq="5min",
    )
    sat_datetime_3 = pd.date_range(
        t0 - pd.Timedelta("120min"),
        t0 - pd.Timedelta("35min"),
        freq="5min",
    )
    sat_datetime_4 = pd.to_datetime(
        [t for t in sat_datetime_1 if t != t0 - pd.Timedelta("30min")],
    )
    sat_datetime_5 = pd.to_datetime(
        [t for t in sat_datetime_1 if t != t0 - pd.Timedelta("60min")],
    )

    assert check_model_satellite_inputs_available(config_filename, t0, sat_datetime_1)
    assert check_model_satellite_inputs_available(config_filename, t0, sat_datetime_2)
    assert not check_model_satellite_inputs_available(
        config_filename,
        t0,
        sat_datetime_3,
    )
    assert not check_model_satellite_inputs_available(
        config_filename,
        t0,
        sat_datetime_4,
    )
    assert not check_model_satellite_inputs_available(
        config_filename,
        t0,
        sat_datetime_5,
    )


def test_extend_satellite_data_with_nans(sat_5_data):
    limit = pd.Timedelta("3h")
    max_sat_time = pd.to_datetime(sat_5_data.time).max()

    # This test should do nothing since the satellite data is not delayed
    t0 = max_sat_time
    ds_extended = extend_satellite_data_with_nans(sat_5_data, t0=t0, limit=limit)

    assert (ds_extended.time.values == sat_5_data.time.values).all()

    # This test should add nans to the end of the satellite data
    delay = pd.Timedelta("2h")
    t0 = max_sat_time + delay
    ds_extended = extend_satellite_data_with_nans(sat_5_data, t0=t0, limit=limit)

    assert ds_extended.time.values[-1] == t0
    assert len(sat_5_data.time) + int(delay / pd.Timedelta("5min")) == len(
        ds_extended.time,
    )

    # This test should add nans to the end of the satellite data but only up to a limit
    delay = pd.Timedelta("4h")
    t0 = max_sat_time + delay
    ds_extended = extend_satellite_data_with_nans(sat_5_data, t0=t0, limit=limit)

    assert ds_extended.time.values[-1] == (t0 - delay + limit)
    assert len(sat_5_data.time) + int(limit / pd.Timedelta("5min")) == len(
        ds_extended.time,
    )


def test_interpolate_missing_satellite_timestamps():
    """Test that missing timestamps are interpolated"""

    # Create a 15 minutely dataset with many missing (5-minutely) timestamps
    t_start = "2023-01-01 00:00"
    t_end = "2023-01-01 03:00"
    times = np.delete(pd.date_range(start=t_start, end=t_end, freq="15min"), 1)

    ds = xr.DataArray(
        data=np.ones(times.shape),
        dims=["time"],
        coords={time,times},
    ).to_dataset(name="data")

    ds_interp = interpolate_missing_satellite_timestamps(
        ds,
        max_gap=pd.Timedelta("15min"),
    )

    # The function interpolates to 5 minute intervals but will only interpolate between
    # timestamps if there is less than 15 minutes between them. In this case, the 5 minute
    # intervals between the first two timestamps should not have been interpolated because
    # there is a 30 minute gap
    expected_times = pd.date_range(start=t_start, end=t_end, freq="5min")
    expected_times = [t for t in expected_times if not (times[0] < t < times[1])]

    assert (
        (pd.to_datetime(ds_interp.time) == pd.to_datetime(expected_times)).all().item()
    )

    assert (ds_interp.data.values == 1).all().item()


def test_contains_too_many_of_value(sat_5_data):

    # The original data has no zeros or NaNs
    assert not contains_too_many_of_value(sat_5_data, value=0, threshold=0.0)
    assert not contains_too_many_of_value(sat_5_data, value=np.nan, threshold=0.0)

    # Check it can detect too many zeros
    ds = sat_5_data.copy(deep=True)
    ds["data"].values[:] = 0
    assert contains_too_many_of_value(ds, value=0, threshold=0.1)

    # Check it can detect too many NaNs
    ds = sat_5_data.copy(deep=True)
    ds["data"].values[:] = np.nan
    assert contains_too_many_of_value(ds, value=np.nan, threshold=0.1)
