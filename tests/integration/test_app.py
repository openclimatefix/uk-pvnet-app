import datetime
import os
import tempfile
from unittest.mock import patch

import pytest
from ocf import dp

from pvnet_app.app import run
from pvnet_app.model_configs.pydantic_models import get_all_models
from pvnet_app.save import fetch_locations

NUM_GSPS = 334


async def check_number_of_forecasts(client, model_configs, test_t0):
    gsp_uuid_map = await fetch_locations(client=client)
    national_uuid = gsp_uuid_map[0].location_uuid
    location_uuids = [loc.location_uuid for loc in gsp_uuid_map.values()]
    init_time_utc = test_t0.to_pydatetime().replace(tzinfo=datetime.UTC)
    first_target_time_utc = init_time_utc + datetime.timedelta(minutes=30)

    list_response = await client.list_forecasters(dp.ListForecastersRequest())
    forecasters_by_name = {f.forecaster_name: f for f in list_response.forecasters}

    for model_config in model_configs:
        model_name = model_config.name.replace("-", "_")
        model_adjust = f"{model_name}_adjust"
        expected_num_horizons = 72 if model_config.is_day_ahead else 16
        end_timestamp_utc = init_time_utc + datetime.timedelta(
            minutes=30 * (expected_num_horizons + 1),
        )

        assert model_name in forecasters_by_name
        assert model_adjust in forecasters_by_name

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
            forecast_response = await client.get_forecast_as_timeseries(
                dp.GetForecastAsTimeseriesRequest(
                    energy_source=dp.EnergySource.SOLAR,
                    location_uuid=national_uuid,
                    forecaster=forecasters_by_name[forecaster_name],
                    time_window=dp.TimeWindow(
                        start_timestamp_utc=init_time_utc,
                        end_timestamp_utc=end_timestamp_utc,
                    ),
                    initialization_timestamp_utc=init_time_utc,
                ),
            )

            forecast_values = {
                int((value.target_timestamp_utc - init_time_utc).total_seconds() // 60): value
                for value in forecast_response.values
            }

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

        # Satellite data will be mocked
        os.environ["SATELLITE_ICECHUNK_PATH_5"] = "s3://fake/sat5"

        # Set environmental variables
        os.environ["RUN_CRITICAL_MODELS_ONLY"] = "False"
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"
        os.environ["FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        with patch(
            "pvnet_app.data.satellite.open_satellite_data",
            side_effect=[sat_5_data_zero_delay, None],
        ):
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

        # Satellite data will be mocked as unavailable
        os.environ["SATELLITE_ICECHUNK_PATH_5"] = "s3://fake/sat5"

        os.environ["RUN_CRITICAL_MODELS_ONLY"] = "False"
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"
        os.environ["FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        with patch("pvnet_app.data.satellite.open_satellite_data", return_value=None):
            await run(t0=test_t0)

    # Only the models which don't use satellite will be run in this case
    model_configs = get_all_models()
    model_configs = [model for model in model_configs if not model.uses_satellite_data]

    await check_number_of_forecasts(client, model_configs, test_t0)
