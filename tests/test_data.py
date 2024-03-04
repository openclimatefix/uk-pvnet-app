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

from pvnet_app.data import download_all_sat_data, preprocess_sat_data, sat_path, sat_5_path, sat_15_path
import zarr
import os
import pandas as pd
import tempfile


def save_to_zarr_zip(sat_5_data, filename):
    encoding = {"data": {"dtype": "int16"}}
    with zarr.ZipStore(filename) as store:
        sat_5_data.to_zarr(store, compute=True, mode="w", encoding=encoding, consolidated=True)


def test_download_sat_data(sat_5_data):
    """1. Download just 5 minute sat"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # tempfile
        filename = os.path.join(tmpdirname, "latest.zarr.zip")

        # zip sat_5_data file to 'sat_5_data.zarr.zip'
        save_to_zarr_zip(sat_5_data, filename=filename)

        os.environ["SATELLITE_ZARR_PATH"] = filename
        download_all_sat_data()

        # assert that the file 'sat_5_path' exists
        assert os.path.exists(sat_5_path)
        assert not os.path.exists(sat_15_path)


def test_download_sat_15_data(sat_5_data):
    """2. Download just 15 minute sat"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        # tempfile
        filename = os.path.join(tmpdirname, "latest_15.zarr.zip")

        # zip sat_5_data file to 'sat_5_data.zarr.zip'
        save_to_zarr_zip(sat_5_data, filename=filename)

        os.environ["SATELLITE_ZARR_PATH"] = os.path.join(tmpdirname, "latest.zarr.zip")
        download_all_sat_data()

        assert not os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)


def test_download_sat_both_data(sat_5_data):
    """3. Download 5 minute sat, then 15 minute sa"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        # tempfile
        filename = os.path.join(tmpdirname, "latest.zarr.zip")
        save_to_zarr_zip(sat_5_data, filename=filename)

        filename = os.path.join(tmpdirname, "latest_15.zarr.zip")
        save_to_zarr_zip(sat_5_data, filename=filename)

        os.environ["SATELLITE_ZARR_PATH"] = os.path.join(tmpdirname, "latest.zarr.zip")
        download_all_sat_data()

        assert os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)


def test_preprocess_sat_data(sat_5_data):
    """4. Download and process 5 minute"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # tempfile
        filename = os.path.join(tmpdirname, "latest.zarr.zip")

        # zip sat_5_data file to 'sat_5_data.zarr.zip'
        save_to_zarr_zip(sat_5_data, filename=filename)

        os.environ["SATELLITE_ZARR_PATH"] = filename
        download_all_sat_data()
        use_15_minute = preprocess_sat_data(pd.Timestamp.now(tz=None))
        assert use_15_minute == False


def test_preprocess_sat_15_data(sat_5_data):
    """5. Download and process 15 minute"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # tempfile
        filename = os.path.join(tmpdirname, "latest_15.zarr.zip")

        save_to_zarr_zip(sat_5_data, filename=filename)

        os.environ["SATELLITE_ZARR_PATH"] = os.path.join(tmpdirname, "latest.zarr.zip")
        download_all_sat_data()
        assert not os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)

        use_15_minute = preprocess_sat_data(pd.Timestamp.now(tz=None))
        assert use_15_minute == True

        # assert that the file 'sat_5_path' exists
        assert os.path.exists(sat_path)
        assert not os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)


def test_preprocess_old_sat_5_data_(sat_5_data):
    """6. Download and process 5 and 15 minute, then use 15 minute"""

    # make temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # tempfile
        filename = os.path.join(tmpdirname, "latest.zarr.zip")
        save_to_zarr_zip(sat_5_data, filename=filename)

        filename = os.path.join(tmpdirname, "latest_15.zarr.zip")
        save_to_zarr_zip(sat_5_data, filename=filename)

        os.environ["SATELLITE_ZARR_PATH"] = os.path.join(tmpdirname, "latest.zarr.zip")
        download_all_sat_data()
        assert os.path.exists(sat_5_path)
        assert os.path.exists(sat_15_path)

        use_15_minute = preprocess_sat_data(pd.Timestamp.now(tz=None) + pd.Timedelta(days=1))
        assert use_15_minute == True
