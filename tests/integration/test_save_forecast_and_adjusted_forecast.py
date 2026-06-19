import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp

from src.pvnet_app.save import build_multi_forecast_creation_request, fetch_or_create_forecaster


@pytest.mark.asyncio(loop_scope="session")
async def test_save_forecast_and_adjusted_forecast(
    client: dp.DataPlatformDataServiceStub,
    setup_dp_locations,  # noqa: ARG001 - ensures observer exists before this test
):
    """Test saving data to the data-platform and that the adjusted forecast is calculated correctly.

    In this test we
    - Create 2 new locations
    - Add some generation data for the national location (gsp 0) for the previous day
    - Add a forecast for gsp 0 on the previous day
    - Save a current forecast for gsp 0 and gsp 1 using `build_multi_forecast_creation_request()`
    - Check that the forecasts were saved
    - Check that the forecast values are correct
    - Check that the adjusted forecast values are calculated and saved correctly
    """

    # This will be the init time for the forecast we test on
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    t0_yesterday = t0 - timedelta(days=1)

    n_steps = 24 # number of forecast steps
    freq_mins = 30 # frequency of forecast steps in minutes
    capacity_watts = 1_000
    p10_frac = 0.3
    p50_frac = 0.5
    p90_frac = 0.7

    forecaster_name = "test_model"

    async def create_location(
        gsp_id: int,
        location_name: str,
        location_type: dp.LocationType,
        geometry_wkt: str,
    ):
        return await client.create_location(
                dp.CreateLocationRequest(
                location_name=location_name,
                energy_source=dp.EnergySource.SOLAR,
                geometry_wkt=geometry_wkt,
                location_type=location_type,
                effective_capacity_watts=capacity_watts,
                metadata=Struct(fields={"gsp_id": Value(number_value=gsp_id)}),
                valid_from_utc=t0_yesterday,
            ),
        )

    # Setup: Create locations
    locations = {}

    locations[0] = await create_location(
        gsp_id=0,
        location_name="test_save_gsp0",
        geometry_wkt="POINT(10 10)",
        location_type=dp.LocationType.NATION,
    )

    locations[1] = await create_location(
        gsp_id=1,
        location_name="test_save_gsp1",
        geometry_wkt="POINT(11 11)",
        location_type=dp.LocationType.GSP,
    )

    # Setup: Add forecast from same time "yesterday" so that the adjusted forecast can be calculated
    forecaster = await fetch_or_create_forecaster(client, model_tag=forecaster_name)

    _ = await client.create_forecast(
        dp.CreateForecastRequest(
            location_uuid=locations[0].location_uuid,
            forecaster=forecaster,
            energy_source=dp.EnergySource.SOLAR,
            init_time_utc=t0_yesterday,
            values=[
                dp.CreateForecastRequestForecastValue(
                    horizon_mins=i*freq_mins,
                    p50_fraction=p50_frac,
                )
                for i in range(n_steps)
            ],
        ),
    )

    # Setup: Add fake generation data so that the adjusted forecast can be calculated
    prev_valid_times = (
        pd.date_range(t0_yesterday, periods=n_steps, freq=f"{freq_mins}min", tz="UTC")
        .to_pydatetime()
    )
    prev_values = np.array([0.5 + 0.01 * i for i in range(n_steps)]) * capacity_watts

    _ = await client.create_observations(
        dp.CreateObservationsRequest(
            location_uuid=locations[0].location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            observer_name="pvlive_day_after",
            values=[
                # Go from 500W to 730W in 10W increments
                dp.CreateObservationsRequestValue(
                    timestamp_utc=dt,
                    value_watts=int(prev_values[i]),
                )
                for i, dt in enumerate(prev_valid_times)
            ],
        ),
    )

    forecast_normed_da = xr.DataArray(
        # 24 horizons, 2 GSPs, 3 outputs (values: p10=0.3, p50=0.5, p90=0.7)
        data=np.tile([p10_frac, p50_frac, p90_frac], (n_steps, len(locations), 1)),
        dims=["valid_times_utc", "gsp_id", "output_label"],
        coords={
            "valid_times_utc": pd.date_range(t0, periods=n_steps, freq=f"{freq_mins}min"),
            "gsp_id": list(locations.keys()),
            "output_label": ["p10", "p50", "p90"],
            "horizon_mins": ("valid_times_utc", np.arange(n_steps) * freq_mins),
        },
    )

    # Test: Run the function
    requests = await build_multi_forecast_creation_request(
        forecast_normed_da=forecast_normed_da,
        locations=locations,
        client=client,
        model_tag=forecaster_name,
        init_time_utc=pd.Timestamp(t0).tz_convert(None),
        metadata=Struct().from_pydict({}),
    )

    # Check: The requests are as expected
    assert len(requests) == len(locations) + 1  # 2 forecasts for gsp_id 0, 1 forecast for others
    assert isinstance(requests[0], dp.CreateForecastRequest)

    # Test: Save the forecasts to the data platform
    _ = await asyncio.gather(
        *(client.create_forecast(req) for req in requests),
        return_exceptions=True,
    )

    # Check: Read from the data platform to check it was saved
    forecasters_resp = await client.list_forecasters(
        dp.ListForecastersRequest(
            forecaster_names_filter=[forecaster_name, f"{forecaster_name}_adjust"],
        ),
    )

    forecasters_lookup = {f.forecaster_name: f for f in forecasters_resp.forecasters}

    # Check: The forecaster and adjuster forecaster were created
    expected_forecasters = {forecaster_name, f"{forecaster_name}_adjust"}
    assert set(forecasters_lookup) == expected_forecasters

    # Check: There is one forecast for GSP 1 with the non-adjusted forecaster
    latest_forecasts_resp = await client.get_latest_forecasts(
        dp.GetLatestForecastsRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=locations[1].location_uuid,
        ),
    )
    assert len(latest_forecasts_resp.forecasts) == 1
    assert latest_forecasts_resp.forecasts[0].forecaster.forecaster_name == forecaster_name

    # Check: There are two forecasts for GSP 0, one with the non-adjusted forecaster and one with
    # the adjusted forecaster
    latest_forecasts_resp = await client.get_latest_forecasts(
        dp.GetLatestForecastsRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=locations[0].location_uuid,
        ),
    )
    assert len(latest_forecasts_resp.forecasts) == 2
    assert (
        {f.forecaster.forecaster_name for f in latest_forecasts_resp.forecasts}
        == expected_forecasters
    )

    time_window = dp.TimeWindow(start_timestamp_utc=t0, end_timestamp_utc=t0 + timedelta(days=1))

    # Check: The forecast values for non-adjusted forecast are as expected
    forecast_response = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=locations[0].location_uuid,
            forecaster=forecasters_lookup[forecaster_name],
            time_window=time_window,
        ),
    )
    assert len(forecast_response.values) == n_steps
    assert all(v.p50_value_fraction == p50_frac for v in forecast_response.values)

    # Check: The number of forecast values for adjuster forecast
    forecast_response = await client.get_forecast_as_timeseries(
        dp.GetForecastAsTimeseriesRequest(
            energy_source=dp.EnergySource.SOLAR,
            location_uuid=locations[0].location_uuid,
            forecaster=forecasters_lookup[f"{forecaster_name}_adjust"],
            time_window=time_window,
        ),
    )

    assert len(forecast_response.values) == n_steps

    # Check: The adjusted forecast p50 values
    # - The previous day's forecast was 0.5 for all horizons
    # - The previous day's observed values are 0.5, 0.51, 0.52, ...
    # - The deltas are 0, -0.01, -0.02, ...
    # - Limited to 10% of forecasted value
    expected_adjustments = np.clip(
        p50_frac - prev_values/capacity_watts,
        -p50_frac* 0.1,
        p50_frac * 0.1,
    )

    for i, value in enumerate(forecast_response.values):
        expected_adjustment = expected_adjustments[i]
        assert np.isclose(value.p50_value_fraction, p50_frac - expected_adjustment, atol=1e-4)
        assert np.isclose(value.other_statistics_fractions["p10"],  p10_frac - expected_adjustment)
        assert np.isclose(value.other_statistics_fractions["p90"], p90_frac - expected_adjustment)

