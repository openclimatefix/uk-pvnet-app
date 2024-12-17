"""
Tests for download_sat_data and preprocess_sat_data

1. Download just 5 minute sat
2. Download just 15 minute sat
3. Download 5 minute sat, then 15 minute sat
4. Download and process 5 minute
5. Download and process 15 minute
6. Download and process 5 and 15 minute, then use 15 minute

Note that I'm not sure these tests will work in parallel, due to files being saved in the same places
"""
from datetime import datetime, timedelta

import os
import tempfile

import pytest
import zarr
import numpy as np
import pandas as pd
import xarray as xr

from pvnet_app.data.satellite import (
    download_all_sat_data,
    preprocess_sat_data,
    check_model_satellite_inputs_available,
    sat_path,
    sat_5_path,
    sat_15_path,
    extend_satellite_data_with_nans,
)


def save_to_zarr_zip(ds, filename):
    encoding = {"data": {"dtype": "int16"}}
    with zarr.ZipStore(filename) as store:
        ds.to_zarr(store, compute=True, mode="w", encoding=encoding, consolidated=True)


def check_timesteps(sat_path, expected_freq_mins):
    ds_sat = xr.open_zarr(sat_path)

    if not isinstance(expected_freq_mins, list):
        expected_freq_mins = [expected_freq_mins]

    dts = pd.to_datetime(ds_sat.time).diff()[1:]
    assert (np.isin(dts, [np.timedelta64(m, "m") for m in expected_freq_mins])).all(), dts


def test_download_sat_5_data(sat_5_data):
    """Download only the 5 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # Make 5-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()

        # Assert that the file 'sat_5_path' exists
        assert os.path.exists(sat_5_path)
        assert not os.path.exists(sat_15_path)

        # Check the satellite data is 5-minutely
        check_timesteps(sat_5_path, expected_freq_mins=5)


def test_download_sat_15_data(sat_15_data):
    """Download only the 15 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"

        download_all_sat_data()

        # Assert that the file 'sat_15_path' exists
        assert not os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)

        # Check the satellite data is 15-minutely
        check_timesteps(sat_15_path, expected_freq_mins=15)


def test_download_sat_both_data(sat_5_data, sat_15_data):
    """Download 5 minute sat and 15 minute satellite data"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # Make 5- and 15-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"

        download_all_sat_data()

        assert os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)

        # Check this satellite data is 5-minutely
        check_timesteps(sat_5_path, expected_freq_mins=5)

        # Check this satellite data is 15-minutely
        check_timesteps(sat_15_path, expected_freq_mins=15)


def test_preprocess_sat_data(sat_5_data, test_t0):
    """Download and process only the 5 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # Make 5-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()

        preprocess_sat_data(test_t0)

        # Check the satellite data is 5-minutely
        check_timesteps(sat_path, expected_freq_mins=5)


def test_preprocess_sat_15_data(sat_15_data, test_t0):
    """Download and process only the 15 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()

        preprocess_sat_data(test_t0)

        # We infill the satellite data to 5 minutes in the process step
        check_timesteps(sat_path, expected_freq_mins=5)


def test_preprocess_old_sat_5_data(sat_5_data_delayed, sat_15_data, test_t0):
    """Download and process 5 and 15 minute satellite data. Use the 15 minute data since the
    5 minute data is too delayed
    """

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        save_to_zarr_zip(sat_5_data_delayed, filename="latest.zarr.zip")
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()

        preprocess_sat_data(test_t0)

        # We infill the satellite data to 5 minutes in the process step
        check_timesteps(sat_path, expected_freq_mins=5)


def test_check_model_satellite_inputs_available(config_filename):

    t0 = datetime(2023,1,1)
    sat_datetime_1 = pd.date_range(t0 - timedelta(minutes=120), t0- timedelta(minutes=5), freq="5min")
    sat_datetime_2 = pd.date_range(t0 - timedelta(minutes=120), t0 - timedelta(minutes=15), freq="5min")
    sat_datetime_3 = pd.date_range(t0 - timedelta(minutes=120), t0 - timedelta(minutes=35), freq="5min")

    assert check_model_satellite_inputs_available(config_filename, t0, sat_datetime_1)
    assert check_model_satellite_inputs_available(config_filename, t0, sat_datetime_2)
    assert not check_model_satellite_inputs_available(config_filename, t0, sat_datetime_3)


def test_extend_satellite_data_with_nans(sat_5_data, test_t0):

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # save sat to zarr
        filename = "sat_5_data.zarr"
        sat_5_data.to_zarr(filename)

        time = sat_5_data.time.values
        t0 = pd.to_datetime(sat_5_data.time).max()
        extend_satellite_data_with_nans(t0=t0, satellite_data_path=filename)

        # load new file
        ds = xr.open_zarr(filename)
        assert (ds.time.values == time).all()


def test_extend_satellite_data_with_nans_over_3_hours(sat_5_data, test_t0):

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # save sat to zarr
        filename = "sat_5_data.zarr"
        sat_5_data.to_zarr(filename)

        time = sat_5_data.time.values
        t0 = pd.to_datetime(sat_5_data.time).max() + pd.Timedelta(hours=4)
        extend_satellite_data_with_nans(t0=t0, satellite_data_path=filename)

        # load new file
        ds = xr.open_zarr(filename)
        assert len(time) + 3*12 == len(ds.time)
        assert ds.time.values[-1] == t0


def test_zeros_in_sat_data(sat_15_data_small, test_t0):
    """Check error is made if data has zeros"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:

        # Change to temporary working directory
        os.chdir(tmpdirname)

        # make half the values zeros
        sat_15_data_small.data[::2] = 0

        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data_small, filename="latest.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()

        # check an error is made
        with pytest.raises(Exception):
            preprocess_sat_data(test_t0)


def test_remove_satellite_data(sat_15_data_small, test_t0):
    """Check error is made if data has zeros"""
    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Change to temporary working directory
        os.chdir(tmpdirname)

        # make half the values zeros
        sat_15_data_small.data[::2] = np.nan

        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data_small, filename="latest.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()

        # check an error is made
        with pytest.raises(Exception):
            preprocess_sat_data(test_t0)
