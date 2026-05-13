import os
import tempfile

import pytest
import zarr
import datetime
from ocf import dp

from pvnet_app.app import run
from pvnet_app.model_configs.pydantic_models import get_all_models
from pvnet_app.save import fetch_dp_gsp_uuid_map

NUM_GSPS = 331


async def check_forecasts_in_data_platform(client, model_configs, test_t0):
    gsp_uuid_map = await fetch_dp_gsp_uuid_map(client=client)
    national_uuid = gsp_uuid_map[0]

    list_response = await client.list_forecasters(dp.ListForecastersRequest())
    forecaster_names = {f.forecaster_name for f in list_response.forecasters}

    for model_config in model_configs:
        model_name = model_config.name.replace("-", "_")

        assert model_name in forecaster_names, f"Missing forecaster '{model_name}' in data platform"
        assert f"{model_name}_adjust" in forecaster_names

        forecasts_response = await client.get_latest_forecasts(
            dp.GetLatestForecastsRequest(
                energy_source=dp.EnergySource.SOLAR,
                pivot_timestamp_utc=test_t0.to_pydatetime().replace(tzinfo=datetime.UTC),
                location_uuid=national_uuid,
            ),
        )
        forecast_model_names = {f.forecaster.forecaster_name for f in forecasts_response.forecasts}
        assert model_name in forecast_model_names


@pytest.mark.asyncio(loop_scope="session")
async def test_app(
    dp_client,  # noqa: ARG001
    setup_dp_locations,  # noqa: ARG001
    test_t0,
    nwp_ukv_data,
    nwp_ecmwf_data,
    sat_5_data_zero_delay,
    cloudcasting_data,
):
    """Test the app running the intraday models"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(temp_nwp_path)

        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        os.environ["CLOUDCASTING_ZARR_PATH"] = temp_nwp_path = "temp_cloudcasting.zarr"
        cloudcasting_data.to_zarr(temp_nwp_path)

        # In production sat zarr is zipped
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path = "temp_sat.zarr.zip"
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data_zero_delay.to_zarr(store)

        # Set environmental variables
        os.environ["RUN_CRITICAL_MODELS_ONLY"] = "False"
        os.environ["ALLOW_SAVE_GSP_SUM"] = "True"
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"
        os.environ["FORECAST_VALIDATION_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        # Run prediction
        await run(t0=test_t0)

    model_configs = get_all_models(get_critical_only=False)
    await check_forecasts_in_data_platform(dp_client, model_configs, test_t0)


@pytest.mark.asyncio(loop_scope="session")
async def test_app_no_sat(
    dp_client,  # noqa: ARG001
    setup_dp_locations,  # noqa: ARG001
    test_t0,
    nwp_ukv_data,
    nwp_ecmwf_data,
):
    """Test the app for the case when no satellite data is available"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(temp_nwp_path)

        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        # There is no satellite data available at the environ path
        os.environ["SATELLITE_ZARR_PATH"] = "nonexistent_sat.zarr.zip"

        os.environ["RUN_CRITICAL_MODELS_ONLY"] = "False"
        os.environ["ALLOW_SAVE_GSP_SUM"] = "True"
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"
        os.environ["FORECAST_VALIDATION_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        await run(t0=test_t0)

    # Only the models which don't use satellite will be run in this case
    model_configs = get_all_models()
    model_configs = [model for model in model_configs if not model.uses_satellite_data]

    await check_forecasts_in_data_platform(dp_client, model_configs, test_t0)
