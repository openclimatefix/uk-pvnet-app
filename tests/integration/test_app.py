import datetime
import os
import tempfile

import pytest
import zarr
from ocf import dp

from pvnet_app.app import run
from pvnet_app.model_configs.pydantic_models import get_all_models
from pvnet_app.save import fetch_dp_gsp_uuid_map

NUM_GSPS = 331


async def _get_streamed_forecast_values(client, location_uuid, forecaster, test_t0, num_horizons):
    stream_response = client.stream_forecast_data(
        dp.StreamForecastDataRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=location_uuid,
            forecasters=[forecaster],
            time_window=dp.StreamForecastDataRequestTimeWindow(
                start_timestamp_utc=test_t0.to_pydatetime().replace(tzinfo=datetime.UTC),
                end_timestamp_utc=(
                    test_t0 + datetime.timedelta(minutes=30 * (num_horizons + 1))
                ).to_pydatetime().replace(tzinfo=datetime.UTC),
            ),
        ),
    )

    init_time_utc = test_t0.to_pydatetime().replace(tzinfo=datetime.UTC)
    forecast_values = {}
    async for value in stream_response:
        if value.init_timestamp == init_time_utc:
            forecast_values[value.horizon_mins] = value

    return forecast_values


async def check_number_of_forecasts(client, model_configs, test_t0):
    gsp_uuid_map = await fetch_dp_gsp_uuid_map(client=client)
    national_uuid = gsp_uuid_map[0]
    location_uuids = list(gsp_uuid_map.values())
    first_target_time_utc = (
        test_t0 + datetime.timedelta(minutes=30)
    ).to_pydatetime().replace(tzinfo=datetime.UTC)

    list_response = await client.list_forecasters(dp.ListForecastersRequest())
    forecasters_by_name = {f.forecaster_name: f for f in list_response.forecasters}

    for model_config in model_configs:
        model_name = model_config.name.replace("-", "_")
        model_adjust = f"{model_name}_adjust"
        expected_num_horizons = 72 if model_config.is_day_ahead else 16

        assert model_name in forecasters_by_name
        assert model_adjust in forecasters_by_name

        forecasts_response = await client.get_latest_forecasts(
            dp.GetLatestForecastsRequest(
                energy_source=dp.EnergySource.SOLAR,
                pivot_timestamp_utc=test_t0.to_pydatetime().replace(tzinfo=datetime.UTC),
                location_uuid=national_uuid,
            ),
        )
        forecast_model_names = {f.forecaster.forecaster_name for f in forecasts_response.forecasts}
        assert model_name in forecast_model_names

        forecast_at_first_timestamp = await client.get_forecast_at_timestamp(
            dp.GetForecastAtTimestampRequest(
                energy_source=dp.EnergySource.SOLAR,
                timestamp_utc=first_target_time_utc,
                location_uuids=location_uuids,
                forecaster=forecasters_by_name[model_name],
            ),
        )
        assert len(forecast_at_first_timestamp.values) == NUM_GSPS + 1

        for forecaster_name in [model_name, model_adjust]:
            forecast_values = await _get_streamed_forecast_values(
                client=client,
                location_uuid=national_uuid,
                forecaster=forecasters_by_name[forecaster_name],
                test_t0=test_t0,
                num_horizons=expected_num_horizons,
            )

            assert sorted(forecast_values) == [
                30 * horizon for horizon in range(1, expected_num_horizons + 1)
            ]

            for value in forecast_values.values():
                assert "p10" in value.other_statistics_fractions
                assert "p90" in value.other_statistics_fractions


@pytest.mark.asyncio(loop_scope="session")
async def test_app(
    client,
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
        os.environ["FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        # Run prediction
        await run(t0=test_t0)

    model_configs = get_all_models(get_critical_only=False)
    await check_number_of_forecasts(client, model_configs, test_t0)


@pytest.mark.asyncio(loop_scope="session")
async def test_app_no_sat(
    client,
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
        os.environ["FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        await run(t0=test_t0)

    # Only the models which don't use satellite will be run in this case
    model_configs = get_all_models()
    model_configs = [model for model in model_configs if not model.uses_satellite_data]

    await check_number_of_forecasts(client, model_configs, test_t0)
