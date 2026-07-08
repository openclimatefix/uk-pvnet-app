import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp

from src.pvnet_app.save import (
    create_forecaster_if_not_exists,
    limit_adjuster,
    save_forecast_to_data_platform,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_save_to_generation_to_data_platform(
    client: dp.DataPlatformDataServiceStub,
    setup_dp_locations,  # noqa: ARG001 - ensures observer + locations exist before this test
):
    """
    Test saving data to the Data Platform.
    This test uses the `data_platform` fixture to ensure that the Data Platform service
    is running and can accept data.

    For gsp_id 0, we expect 2 forecasts, one normal and one with the adjusted values
    For gsp_id 1, we expect 1 forecast, only the normal one

    UK National (gsp_id 0) saves all 7 plevels (p2, p10, p25, p50, p75, p90, p98);
    every other GSP saves only p10, p50 and p90.

    In this test we
    1. select up locations gsp 0 and gsp 1
    2. add some fake generation data for gsp 0 on 2024-12-31
    3. add a fake forecast for gsp 0 on 2024-12-31
    4. Make forecast data for 2025-01-01 for both gsp 0 and gsp 1
    5. call the save_forecast_to_data_platform function
    6. check that the forecasts were saved correctly
    7. check that the forecast values are correctly
    8. check that the adjusted forecast values are limited correctly
    """
    # 1. setup: add location - gsp 0
    metadata = Struct(fields={"gsp_id": Value(number_value=0)})
    create_location_request = dp.CreateLocationRequest(
        location_name="test_save_gsp0",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(10 10)",
        location_type=dp.LocationType.NATION,
        effective_capacity_watts=1_000_000,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    create_location_response = await client.create_location(create_location_request)
    location_uuid_0 = create_location_response.location_uuid

    # setup: add location - gsp 1
    metadata = Struct(fields={"gsp_id": Value(number_value=1)})
    create_location_request = dp.CreateLocationRequest(
        location_name="test_save_gsp1",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(11 11)",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=1_000_000,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    create_location_response = await client.create_location(create_location_request)
    location_uuid_1 = create_location_response.location_uuid

    # 2. add fake generation data
    create_observation_request = dp.CreateObservationsRequest(
        location_uuid=location_uuid_0,
        energy_source=dp.EnergySource.SOLAR,
        observer_name="pvlive_day_after",
        values=[
            dp.CreateObservationsRequestValue(
                timestamp_utc=datetime.datetime(
                    2024,
                    12,
                    31,
                    tzinfo=datetime.UTC,
                )
                + datetime.timedelta(minutes=30 * i),
                value_watts=500_000 + 10_000 * i,  # go from 0.5MW to 0.98MW
            )
            for i in range(49)
        ],
    )
    _ = await client.create_observations(create_observation_request)

    # 3. add a fake forecast
    forecaster = await create_forecaster_if_not_exists(client, model_tag="test_model")
    create_forecast_request = dp.CreateForecastRequest(
        location_uuid=location_uuid_0,
        forecaster=forecaster,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=datetime.datetime(2024, 12, 31, tzinfo=datetime.UTC),
        values=[
            dp.CreateForecastRequestForecastValue(
                horizon_mins=30 * i,
                p50_fraction=0.5,
            )
            for i in range(48)
        ],
    )
    _ = await client.create_forecast(create_forecast_request)

    # setup: make forecast data
    forecast_data = pd.DataFrame(
        {
            "solar_generation_mw": [0.5] * 24,
            "target_datetime_utc": pd.Timestamp("2025-01-01")
            + pd.timedelta_range(
                start=0,
                periods=24,
                freq="30min",
            ),
        },
    )
    forecast_data["gsp_id"] = 0
    forecast_data["output_label"] = "forecast_fraction"

    # p10 and p90 are saved for every GSP (National and non-National)
    forecast_data_p10 = forecast_data.copy()
    forecast_data_p10["solar_generation_mw"] = [0.3] * 24
    forecast_data_p10["output_label"] = "forecast_fraction_plevel_10"

    forecast_data_p90 = forecast_data.copy()
    forecast_data_p90["solar_generation_mw"] = [0.7] * 24
    forecast_data_p90["output_label"] = "forecast_fraction_plevel_90"

    # non-National GSPs (gsp 1) only get p50, p10 and p90
    forecast_data_gsp1 = pd.concat(
        [forecast_data, forecast_data_p10, forecast_data_p90],
        ignore_index=True,
    )
    forecast_data_gsp1["gsp_id"] = 1

    # National (gsp 0) additionally gets p2, p25, p75 and p98
    extra_national_plevels = {
        "forecast_fraction_plevel_02": 0.1,
        "forecast_fraction_plevel_25": 0.4,
        "forecast_fraction_plevel_75": 0.6,
        "forecast_fraction_plevel_98": 0.9,
    }
    extra_national_data = []
    for label, fraction in extra_national_plevels.items():
        df = forecast_data.copy()
        df["solar_generation_mw"] = [fraction] * 24
        df["output_label"] = label
        extra_national_data.append(df)

    forecast_data = pd.concat(
        [forecast_data, forecast_data_p10, forecast_data_p90,
         *extra_national_data, forecast_data_gsp1],
        ignore_index=True,
    )

    forecast_data = forecast_data.set_index(["target_datetime_utc", "gsp_id", "output_label"])
    forecast_data_da = forecast_data.to_xarray().to_dataarray()

    # 5. Test the function
    _ = await save_forecast_to_data_platform(
        forecast_normed_da=forecast_data_da,
        locations_gsp_uuid_map={0: location_uuid_0, 1: location_uuid_1},
        client=client,
        model_tag="test_model",
        init_time_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
    )

    # 6. check: read from the data platform to check it was saved
    # Filter to only the forecasters created by this test (shared dp_client may have others)
    list_forecasters_response = await client.list_forecasters(
        dp.ListForecastersRequest(forecaster_names_filter=["test_model", "test_model_adjust"]),
    )
    assert len(list_forecasters_response.forecasters) == 2

    # check: There is a forecast object for gsp_id 1
    get_latest_forecasts_request = dp.GetLatestForecastsRequest(
        energy_source=dp.EnergySource.SOLAR,
        location_uuid=location_uuid_1,
    )
    get_latest_forecasts_response = await client.get_latest_forecasts(
        get_latest_forecasts_request,
    )
    assert len(get_latest_forecasts_response.forecasts) == 1
    forecast_gsp1 = get_latest_forecasts_response.forecasts[0]
    assert forecast_gsp1.forecaster.forecaster_name == "test_model"

    # check: There is a forecast object for gsp_id 0
    get_latest_forecasts_request = dp.GetLatestForecastsRequest(
        energy_source=dp.EnergySource.SOLAR,
        location_uuid=location_uuid_0,
    )
    get_latest_forecasts_response = await client.get_latest_forecasts(
        get_latest_forecasts_request,
    )
    assert len(get_latest_forecasts_response.forecasts) == 2
    forecast = get_latest_forecasts_response.forecasts[0]
    assert forecast.forecaster.forecaster_name == "test_model"
    assert forecast.metadata.fields["app_version"].string_value == version("pvnet-app")
    forecast_adjuster = get_latest_forecasts_response.forecasts[1]
    assert forecast_adjuster.forecaster.forecaster_name == "test_model_adjust"

    time_window = dp.TimeWindow(
        start_timestamp_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
        end_timestamp_utc=datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC),
    )

    # check: non-National GSP (gsp 1) only carries p10 and p90, not the extra plevels
    forecast_response_gsp1 = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=location_uuid_1,
            forecaster=forecast_gsp1.forecaster,
            time_window=time_window,
        ),
    )
    assert len(forecast_response_gsp1.values) == 24
    for value in forecast_response_gsp1.values:
        assert set(value.other_statistics_fractions.keys()) == {"p10", "p90"}
        assert np.isclose(value.other_statistics_fractions["p10"], 0.3)
        assert np.isclose(value.other_statistics_fractions["p90"], 0.7)

    # check: the number of forecast values for non-adjusted forecast
    forecast_response = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=location_uuid_0,
            forecaster=forecast.forecaster,
            time_window=time_window,
        ),
    )
    for value in forecast_response.values:
        # National forecast carries all 7 plevels with their raw values
        assert value.p50_value_fraction == 0.5
        assert np.isclose(value.other_statistics_fractions["p02"], 0.1)
        assert np.isclose(value.other_statistics_fractions["p10"], 0.3)
        assert np.isclose(value.other_statistics_fractions["p25"], 0.4)
        assert np.isclose(value.other_statistics_fractions["p75"], 0.6)
        assert np.isclose(value.other_statistics_fractions["p90"], 0.7)
        assert np.isclose(value.other_statistics_fractions["p98"], 0.9)
    assert len(forecast_response.values) == 24

    # 7. check: the number of forecast values, for adjuster forecast
    forecast_response = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=location_uuid_0,
            forecaster=forecast_adjuster.forecaster,
            time_window=time_window,
        ),
    )
    # 8. check the adjusted forecast p50 values
    # The previous days forecast was 0.5 and
    # the observed values are 0.5, 0.51, 0.52, ...
    # the deltas are 0, -0.01, -0.02, ...
    # limited to 10% of 0.5 = 0.05, so we should be limited to 0.5 +/- 0.05
    # Also there are checks for all 7 plevels
    for count, value in enumerate(forecast_response.values):
        # p50
        new_value = 0.5 + 0.01 * count
        if new_value > 0.55:
            new_value = 0.55
        assert np.isclose(value.p50_value_fraction, new_value, atol=1e-4)

        # p02
        new_value_p02 = 0.1 + 0.01 * count
        if new_value_p02 > 0.15:
            new_value_p02 = 0.15
        assert np.isclose(value.other_statistics_fractions["p02"], new_value_p02)

        # p10
        new_value_p10 = 0.3 + 0.01 * count
        if new_value_p10 > 0.35:
            new_value_p10 = 0.35
        assert np.isclose(value.other_statistics_fractions["p10"], new_value_p10)

        # p25
        new_value_p25 = 0.4 + 0.01 * count
        if new_value_p25 > 0.45:
            new_value_p25 = 0.45
        assert np.isclose(value.other_statistics_fractions["p25"], new_value_p25)

        # p75
        new_value_p75 = 0.6 + 0.01 * count
        if new_value_p75 > 0.65:
            new_value_p75 = 0.65
        assert np.isclose(value.other_statistics_fractions["p75"], new_value_p75)

        # p90
        new_value_p90 = 0.7 + 0.01 * count
        if new_value_p90 > 0.75:
            new_value_p90 = 0.75
        assert np.isclose(value.other_statistics_fractions["p90"], new_value_p90)

        # p98
        new_value_p98 = 0.9 + 0.01 * count
        if new_value_p98 > 0.95:
            new_value_p98 = 0.95
        assert np.isclose(value.other_statistics_fractions["p98"], new_value_p98)

    assert len(forecast_response.values) == 24


@pytest.mark.parametrize(
    "forecast_fraction,delta_fraction,capacity_mw,expected",
    [
        # no change
        (0.5, 0.01, 1, 0.01),
        # limit to 10% of 0.5 = 0.05
        (0.5, 0.1, 1, 0.05),
        # limit to 10% of 0.5 = 0.05
        (0.5, 0.2, 1, 0.05),
        # limit to 10% of 0.5 = 0.05
        (0.5, -0.2, 1, -0.05),
        # Delta 0.06 is 1.2 MW, and .06< 10% of 0.8, no change
        (0.8, 0.06, 20, 0.06),
        # Delta 0.06 is 1200 MW  -> 1000 MW -> 0.05 fraction
        (0.8, 0.06, 20_000, 0.05),
        # Delta 0.06 is 1200 MW  -> 1000 MW -> 0.05 fraction
        (0.8, -0.06, 20_000, -0.05),
    ],
)
def test_limit_adjuster(forecast_fraction, delta_fraction, capacity_mw, expected):
    """Test the limit_adjuster function."""
    assert limit_adjuster(delta_fraction, forecast_fraction, capacity_mw) == expected
