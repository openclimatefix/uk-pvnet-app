import os
import tempfile

from pvnet_app.data.nwp import download_all_nwp_data, check_model_nwp_inputs_available
from pvnet_app.consts import nwp_ukv_path, nwp_ecmwf_path

import xarray as xr


def test_download_nwp(nwp_ukv_data, nwp_ecmwf_data):
    """Download only the 5 minute satellite data"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # The function loads NWP data from environment variable
        os.environ["NWP_UKV_ZARR_PATH"] = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(os.environ["NWP_UKV_ZARR_PATH"])

        os.environ["NWP_ECMWF_ZARR_PATH"] = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(os.environ["NWP_ECMWF_ZARR_PATH"])

        download_all_nwp_data()

        ds_loaded_ukv = xr.open_zarr(nwp_ukv_path).compute()
        ds_loaded_ecmwf = xr.open_zarr(nwp_ecmwf_path).compute()

        assert ds_loaded_ukv.identical(nwp_ukv_data)
        assert ds_loaded_ecmwf.identical(nwp_ecmwf_data)



def test_check_model_nwp_inputs_available(config_filename, test_t0, nwp_ukv_data, nwp_ecmwf_data):

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # The function checks NWP data from specific paths
        nwp_ukv_data.to_zarr(nwp_ukv_path)
        nwp_ecmwf_data.to_zarr(nwp_ecmwf_path)

        # The inputs are all available so this should return True
        assert check_model_nwp_inputs_available(config_filename, test_t0)

        # Make no inputs available
        os.system(f"rm -r {nwp_ukv_path}")
        os.system(f"rm -r {nwp_ecmwf_path}")

        assert not check_model_nwp_inputs_available(config_filename, test_t0)

        # Save the NWP data, but with less time steps
        nwp_ukv_data.isel(step=slice(0, 4)).to_zarr(nwp_ukv_path)
        nwp_ecmwf_data.isel(step=slice(0, 4)).to_zarr(nwp_ecmwf_path)

        assert not check_model_nwp_inputs_available(config_filename, test_t0)

        

