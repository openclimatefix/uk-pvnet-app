import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr
from ocf import dp
from pvnet_app.models.pydantic_models import ModelConfig, get_all_models

from pvnet_app.app import run
from pvnet_app.save import fetch_locations
from pvnet_app.settings import AppSettings


async def check_number_of_forecasts(
    client: dp.DataPlatformDataServiceStub,
    model_configs: list[ModelConfig],
    test_t0: pd.Timestamp,
    expected_gsp_ids: list[int],
) -> None:
    """Check that the expected number of forecasts have been saved to the Data Platform."""
    locations_dict = await fetch_locations(client=client)
    forecasters = (await client.list_forecasters(dp.ListForecastersRequest())).forecasters
    forecasters_by_name = {f.forecaster_name: f for f in forecasters}

    for model_config in model_configs:
        model_name = model_config.name.replace("-", "_")
        model_adjust = f"{model_name}_adjust"
        expected_valid_times = pd.date_range(
            test_t0 + pd.Timedelta("30min"),
            periods=(72 if model_config.is_day_ahead else 16),
            freq="30min",
            tz="UTC",
        )
        # Check the model and adjusted model were registered with the Data Platform
        assert model_name in forecasters_by_name
        assert model_adjust in forecasters_by_name

        # Check the model has the right number of locations for the first timestamp
        forecast_at_first_timestamp = (
            await client.get_forecast_at_timestamp(
                dp.GetForecastAtTimestampRequest(
                    energy_source=dp.EnergySource.SOLAR,
                    timestamp_utc=expected_valid_times[0].to_pydatetime(),
                    location_uuids=[loc.location_uuid for loc in locations_dict.values()],
                    forecaster=forecasters_by_name[model_name],
                ),
            )
        ).values

        assert len(forecast_at_first_timestamp) == len(expected_gsp_ids)

        for forecaster_name in [model_name, model_adjust]:
            # Check the model and adjusted model produce forecasts for all expected valid times
            national_forecast = (
                await client.get_forecast_as_timeseries(
                    dp.GetForecastAsTimeseriesRequest(
                        energy_source=dp.EnergySource.SOLAR,
                        location_uuid=locations_dict[0].location_uuid,
                        forecaster=forecasters_by_name[forecaster_name],
                        initialization_timestamp_utc=test_t0.tz_localize("UTC").to_pydatetime(),
                        # Set wide time window to make sure there aren't any values outside the
                        # expected valid times
                        time_window=dp.TimeWindow(
                            start_timestamp_utc=(
                                (expected_valid_times[0] - pd.Timedelta("1D")).to_pydatetime()
                            ),
                            end_timestamp_utc=(
                                (expected_valid_times[-1] + pd.Timedelta("1D")).to_pydatetime()
                            ),
                        ),
                    ),
                )
            ).values

            forecast_valid_times = pd.to_datetime(
                sorted([v.target_timestamp_utc for v in national_forecast]),
            )
            assert all(forecast_valid_times == expected_valid_times)

            # Check the extra statistics fractions are present
            for value in national_forecast:
                assert "p10" in value.other_statistics_fractions
                assert "p90" in value.other_statistics_fractions


@pytest.mark.asyncio(loop_scope="session")
async def test_app(
    dp_client_with_locations: dp.DataPlatformDataServiceStub,
    dp_host_and_port: tuple[str, int],
    test_t0: pd.Timestamp,
    nwp_ukv_data: xr.Dataset,
    nwp_ecmwf_data: xr.Dataset,
    sat_5_data_zero_delay: xr.Dataset,
    cloudcasting_data: xr.Dataset,
    gsp_ids: list[int],
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
    await check_number_of_forecasts(dp_client_with_locations, model_configs, test_t0, gsp_ids)


@pytest.mark.asyncio(loop_scope="session")
async def test_app_no_sat(
    dp_client_with_locations: dp.DataPlatformDataServiceStub,
    dp_host_and_port: tuple[str, int],
    test_t0: pd.Timestamp,
    nwp_ukv_data: xr.Dataset,
    nwp_ecmwf_data: xr.Dataset,
    gsp_ids: list[int],
):
    """Test the app for the case when no satellite data is available"""

    host, port = dp_host_and_port

    # Change the init time to be 30 minutes later than the test_t0, so that the forecasts are for a
    # different time than the previous test
    t0 = test_t0 - pd.Timedelta("30min")

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
            await run(settings=settings, t0=t0)

    # Only the models which don't use satellite will be run in this case
    model_configs = get_all_models()
    model_configs = [model for model in model_configs if not model.uses_satellite_data]

    await check_number_of_forecasts(dp_client_with_locations, model_configs, test_t0, gsp_ids)
