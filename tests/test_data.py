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
import os
import tempfile
import pytest
import zarr
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta

from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_app.data import (
    download_all_sat_data, preprocess_sat_data, sat_path, sat_5_path, sat_15_path
)
from pvnet_app.app import default_model_name, default_model_version


@pytest.fixture()
def data_config_filename():
    # Pull the data config from huggingface
    filename = PVNetBaseModel.get_data_config(
        default_model_name,
        revision=default_model_version,
    )
    return filename


def save_to_zarr_zip(ds, filename):
    encoding = {"data": {"dtype": "int16"}}
    with zarr.ZipStore(filename) as store:
        ds.to_zarr(store, compute=True, mode="w", encoding=encoding, consolidated=True)


def check_timesteps(sat_path, expected_mins, skip_nans=False):
    ds_sat = xr.open_zarr(sat_path)
    
    if not isinstance(expected_mins, list):
        expected_mins = [expected_mins]
    
    dts = pd.to_datetime(ds_sat.time).diff()[1:]
    assert (np.isin(dts, [np.timedelta64(m, "m") for m in expected_mins])).all(), dts


def test_download_sat_5_data(sat_5_data):
    """Download only the 5 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # Change to temporary working directory
        os.chdir(tmpdirname)
        
        # Make 5-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename= "latest.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()
        
        # Assert that the file 'sat_5_path' exists
        assert os.path.exists(sat_5_path)
        assert not os.path.exists(sat_15_path)
        
        # Check the satellite data is 5-minutely
        check_timesteps(sat_5_path, expected_mins=5)


def test_download_sat_15_data(sat_15_data):
    """Download only the 15 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # Change to temporary working directory
        os.chdir(tmpdirname)
        
        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] =  "latest.zarr.zip"
        
        download_all_sat_data()
        
        # Assert that the file 'sat_15_path' exists
        assert not os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)
        
        # Check the satellite data is 15-minutely
        check_timesteps(sat_15_path, expected_mins=15)


def test_download_sat_both_data(sat_5_data, sat_15_data):
    """Download 5 minute sat and 15 minute satellite data"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # Change to temporary working directory
        os.chdir(tmpdirname)
                
        # Make 5- and 15-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] ="latest.zarr.zip"
        
        download_all_sat_data()

        assert os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)
        
        # Check this satellite data is 5-minutely
        check_timesteps(sat_5_path, expected_mins=5)
        
        # Check this satellite data is 15-minutely
        check_timesteps(sat_15_path, expected_mins=15)


def test_preprocess_sat_data(sat_5_data, data_config_filename, test_t0):
    """Download and process only the 5 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # Change to temporary working directory
        os.chdir(tmpdirname)
        
        # Make 5-minutely satellite data available
        save_to_zarr_zip(sat_5_data, filename="latest.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()
        
        preprocess_sat_data(test_t0, data_config_filename)
        
         # Check the satellite data is 5-minutely
        check_timesteps(sat_path, expected_mins=5)


def test_preprocess_sat_15_data(sat_15_data, data_config_filename, test_t0):
    """Download and process only the 15 minute satellite data"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # Change to temporary working directory
        os.chdir(tmpdirname)
        
        # Make 15-minutely satellite data available
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()
        
        preprocess_sat_data(test_t0, data_config_filename)
        
         # Check the satellite data being used is 15-minutely
        check_timesteps(sat_path, expected_mins=15)


def test_preprocess_old_sat_5_data(sat_5_data_delayed, sat_15_data, data_config_filename, test_t0):
    """Download and process 5 and 15 minute satellite data. Use the 15 minute data since the
    5 minute data is too delayed
    """

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # Change to temporary working directory
        os.chdir(tmpdirname)
        
        save_to_zarr_zip(sat_5_data_delayed, filename="latest.zarr.zip")
        save_to_zarr_zip(sat_15_data, filename="latest_15.zarr.zip")

        os.environ["SATELLITE_ZARR_PATH"] = "latest.zarr.zip"
        download_all_sat_data()
        
        preprocess_sat_data(test_t0, data_config_filename)

         # Check the satellite data being used is 15-minutely
        check_timesteps(sat_path, expected_mins=15)
