from pathlib import Path

import pandas as pd
import xarray as xr

from pvnet_app.data.nwp import CloudcastingDownloader, ECMWFDownloader, UKVDownloader


def test_download_nwp(
    nwp_ukv_data: xr.Dataset,
    nwp_ecmwf_data: xr.Dataset,
    cloudcasting_data: xr.Dataset,
    tmp_path: Path,
):
    source_ukv_path = f"{tmp_path}/source_ukv.zarr"
    source_ecmwf_path = f"{tmp_path}/source_ecmwf.zarr"
    source_cloudcasting_path = f"{tmp_path}/source_cloudcasting.zarr"

    nwp_ukv_data.to_zarr(source_ukv_path)
    nwp_ecmwf_data.to_zarr(source_ecmwf_path)
    cloudcasting_data.to_zarr(source_cloudcasting_path)

    ukv_downloader = UKVDownloader(
        source_path=source_ukv_path,
        destination_path=f"{tmp_path}/ukv.zarr",
    )
    ukv_downloader.run()

    ecmwf_downloader = ECMWFDownloader(
        source_path=source_ecmwf_path,
        destination_path=f"{tmp_path}/ecmwf.zarr",
    )
    ecmwf_downloader.run()

    cloudcasting_downloader = CloudcastingDownloader(
        source_path=source_cloudcasting_path,
        destination_path=f"{tmp_path}/cloudcasting.zarr",
    )
    cloudcasting_downloader.run()


def test_check_model_nwp_inputs_available(
    config_filename: str,
    test_t0: pd.Timestamp,
    nwp_ukv_data: xr.Dataset,
    nwp_ecmwf_data: xr.Dataset,
    tmp_path: Path,
):
    # ---- Test case where all inputs are available

    # Create the required NWP data
    source_ukv_path = f"{tmp_path}/source_ukv.zarr"
    source_ecmwf_path = f"{tmp_path}/source_ecmwf.zarr"

    nwp_ukv_data.to_zarr(source_ukv_path)
    nwp_ecmwf_data.to_zarr(source_ecmwf_path)

    ukv_downloader = UKVDownloader(
        source_path=source_ukv_path,
        destination_path=f"{tmp_path}/ukv.zarr",
        window_size_pixels=2,
    )
    ukv_downloader.run()

    ecmwf_downloader = ECMWFDownloader(
        source_path=source_ecmwf_path,
        destination_path=f"{tmp_path}/ecmwf.zarr",
        window_size_pixels=2,
    )
    ecmwf_downloader.run()

    # The inputs are all available so these should return True
    assert ukv_downloader.check_model_inputs_available(config_filename, test_t0)
    assert ecmwf_downloader.check_model_inputs_available(config_filename, test_t0)

    # ---- Test case where no NWP data is available
    ukv_downloader = UKVDownloader(
        source_path="empty_ukv_path.zarr",
        destination_path="dummy.zarr",
        window_size_pixels=2,
    )
    ukv_downloader.run()

    ecmwf_downloader = ECMWFDownloader(
        source_path="empty_ecmwf_path.zarr",
        destination_path="dummy.zarr",
        window_size_pixels=2,
    )
    ecmwf_downloader.run()

    # No inputs are available so these should return False
    assert not ukv_downloader.check_model_inputs_available(config_filename, test_t0)
    assert not ecmwf_downloader.check_model_inputs_available(config_filename, test_t0)

    # ---- Test case where NWP data is available but not all the required time steps

    # Save the NWP data, but with less time steps
    source_ukv_path = f"{tmp_path}/short_source_ukv.zarr"
    source_ecmwf_path = f"{tmp_path}/short_source_ecmwf.zarr"

    nwp_ukv_data.isel(step=slice(0, 4)).to_zarr(source_ukv_path)
    nwp_ecmwf_data.isel(step=slice(0, 4)).to_zarr(source_ecmwf_path)

    ukv_downloader = UKVDownloader(
        source_path=source_ukv_path,
        destination_path=f"{tmp_path}/short_ukv.zarr",
        window_size_pixels=2,
    )
    ukv_downloader.run()

    ecmwf_downloader = ECMWFDownloader(
        source_path=source_ecmwf_path,
        destination_path=f"{tmp_path}/short_ecmwf.zarr",
        window_size_pixels=2,
    )
    ecmwf_downloader.run()

    # Some steps are missing so these should return False
    assert not ukv_downloader.check_model_inputs_available(config_filename, test_t0)
    assert not ecmwf_downloader.check_model_inputs_available(config_filename, test_t0)
