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
