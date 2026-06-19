import os
import tempfile
from datetime import UTC, timedelta
from unittest.mock import patch

import pytest
from ocf import dp

from pvnet_app.app import run
from pvnet_app.model_configs.pydantic_models import get_all_models
from pvnet_app.save import fetch_locations
from pvnet_app.settings import AppSettings

NUM_GSPS = 334


async def check_number_of_forecasts(client, model_configs, test_t0):
    locations_dict = await fetch_locations(client=client)
    national_uuid = locations_dict[0].location_uuid
    location_uuids = [loc.location_uuid for loc in locations_dict.values()]
    init_time_utc = test_t0.to_pydatetime().replace(tzinfo=UTC)
    first_target_time_utc = init_time_utc + timedelta(minutes=30)

    forecasters = (await client.list_forecasters(dp.ListForecastersRequest())).forecasters
    forecasters_by_name = {f.forecaster_name: f for f in forecasters}

    for model_config in model_configs:
        model_name = model_config.name.replace("-", "_")
        model_adjust = f"{model_name}_adjust"
        expected_num_horizons = 72 if model_config.is_day_ahead else 16
        end_timestamp_utc = init_time_utc + timedelta(
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
    dp_client,
    dp_host_and_port,
    setup_dp_locations,  # noqa: ARG001
    test_t0,
    nwp_ukv_data,
    nwp_ecmwf_data,
    sat_5_data_zero_delay,
    cloudcasting_data,
):
    """Test the app running the intraday models"""

    host, port = dp_host_and_port

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables
        ukv_path = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(ukv_path)

        ecmwf_path = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(ecmwf_path)

        cloudcasting_path = "temp_cloudcasting.zarr"
        cloudcasting_data.to_zarr(cloudcasting_path)

        settings = AppSettings(
            nwp_ukv_zarr_path=ukv_path,
            nwp_ecmwf_zarr_path=ecmwf_path,
            cloudcasting_zarr_path=cloudcasting_path,
            # Satellite data will be mocked
            satellite_icechunk_path_5="s3://fake/sat5",
            run_critical_models_only=False,
            forecast_validate_zig_zag_error_threshold=100000,
            forecast_validate_sun_elevation_lower_limit=90,
            data_platform_host=host,
            data_platform_port=port,
        )

        with patch(
            "pvnet_app.data.satellite.open_satellite_data",
            side_effect=[sat_5_data_zero_delay, None],
        ):
            # Run prediction
            await run(settings=settings, t0=test_t0)

    model_configs = get_all_models(get_critical_only=False)
    await check_number_of_forecasts(dp_client, model_configs, test_t0)


@pytest.mark.asyncio(loop_scope="session")
async def test_app_no_sat(
    dp_client,
    dp_host_and_port,
    setup_dp_locations,  # noqa: ARG001
    test_t0,
    nwp_ukv_data,
    nwp_ecmwf_data,
):
    """Test the app for the case when no satellite data is available"""

    host, port = dp_host_and_port

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables
        ukv_path = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(ukv_path)

        ecmwf_path = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(ecmwf_path)

        settings = AppSettings(
            nwp_ukv_zarr_path=ukv_path,
            nwp_ecmwf_zarr_path=ecmwf_path,
            # Satellite data will be mocked as unavailable
            satellite_icechunk_path_5="s3://fake/sat5",
            run_critical_models_only=False,
            forecast_validate_zig_zag_error_threshold=100000,
            forecast_validate_sun_elevation_lower_limit=90,
            data_platform_host=host,
            data_platform_port=port,
        )


        with patch("pvnet_app.data.satellite.open_satellite_data", return_value=None):
            await run(settings=settings, t0=test_t0)

    # Only the models which don't use satellite will be run in this case
    model_configs = get_all_models()
    model_configs = [model for model in model_configs if not model.uses_satellite_data]

    await check_number_of_forecasts(dp_client, model_configs, test_t0)
