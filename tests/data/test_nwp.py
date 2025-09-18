import os
import tempfile

from pvnet_app.data.nwp import CloudcastingDownloader, ECMWFDownloader, UKVDownloader


def test_download_nwp(nwp_ukv_data, nwp_ecmwf_data, cloudcasting_data):
    temp_ukv_path = "temp_nwp_ukv.zarr"
    temp_ecmwf_path = "temp_nwp_ecmwf.zarr"
    temp_cloudcasting_path = "temp_cloudcasting.zarr"

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        nwp_ukv_data.to_zarr(temp_ukv_path)
        nwp_ecmwf_data.to_zarr(temp_ecmwf_path)
        cloudcasting_data.to_zarr(temp_cloudcasting_path)

        ukv_downloader = UKVDownloader(source_path=temp_ukv_path)
        ukv_downloader.run()

        ecmwf_downloader = ECMWFDownloader(source_path=temp_ecmwf_path)
        ecmwf_downloader.run()

        cloudcasting_downloader = CloudcastingDownloader(source_path=temp_cloudcasting_path)
        cloudcasting_downloader.run()


def test_check_model_nwp_inputs_available(config_filename, test_t0, nwp_ukv_data, nwp_ecmwf_data):
    temp_ukv_path = "temp_nwp_ukv.zarr"
    temp_ecmwf_path = "temp_nwp_ecmwf.zarr"

    # Test in a case where all inputs are available
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        # Create the required NWP data
        nwp_ukv_data.to_zarr(temp_ukv_path)
        nwp_ecmwf_data.to_zarr(temp_ecmwf_path)

        ukv_downloader = UKVDownloader(source_path=temp_ukv_path)
        ukv_downloader.run()

        ecmwf_downloader = ECMWFDownloader(source_path=temp_ecmwf_path)
        ecmwf_downloader.run()

        # The inputs are all available so these should return True
        assert ukv_downloader.check_model_inputs_available(config_filename, test_t0)
        assert ecmwf_downloader.check_model_inputs_available(config_filename, test_t0)

    # Test in a case where no NWP data is available
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        ukv_downloader = UKVDownloader(source_path=temp_ukv_path)
        ukv_downloader.run()

        ecmwf_downloader = ECMWFDownloader(source_path=temp_ecmwf_path)
        ecmwf_downloader.run()

        # No inputs are available so these should return False
        assert not ukv_downloader.check_model_inputs_available(config_filename, test_t0)
        assert not ecmwf_downloader.check_model_inputs_available(config_filename, test_t0)

    # Test in a case where NWP data is available but not all the required time steps
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        # Save the NWP data, but with less time steps
        nwp_ukv_data.isel(step=slice(0, 4)).to_zarr(temp_ukv_path)
        nwp_ecmwf_data.isel(step=slice(0, 4)).to_zarr(temp_ecmwf_path)

        ukv_downloader = UKVDownloader(source_path=temp_ukv_path)
        ukv_downloader.run()

        ecmwf_downloader = ECMWFDownloader(source_path=temp_ecmwf_path)
        ecmwf_downloader.run()

        # Some steps are missing so these should return False
        assert not ukv_downloader.check_model_inputs_available(config_filename, test_t0)
        assert not ecmwf_downloader.check_model_inputs_available(config_filename, test_t0)
