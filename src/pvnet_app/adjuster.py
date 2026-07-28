"""Functions to create the adjusted forecast based on recent errors."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from ocf import dp

from pvnet_app.utils import convert_to_utc_datetime

logger = logging.getLogger(__name__)

# Limit adjuster so it doesn't change the forecast by more than this amount in watts
ADJUSTER_LIMIT_ABSOLUTE_WATTS: float = 1e9

# Limit adjuster so it doesn't change the forecast by more than this fraction of the forecast value
ADJUSTER_LIMIT_FORECAST_FRACTION: float = 0.1


async def fetch_adjuster_values(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    init_time_utc: pd.Timestamp,
    forecaster: dp.Forecaster,
) -> dict[int, float]:
    """Fetch the adjuster values for the given location and forecaster."""
    deltas = (
        await client.get_week_average_deltas(
            dp.GetWeekAverageDeltasRequest(
                location_uuid=location_uuid,
                energy_source=dp.EnergySource.SOLAR,
                pivot_timestamp_utc=convert_to_utc_datetime(init_time_utc),
                forecaster=forecaster,
                observer_name="pvlive_day_after",
            ),
        )
    ).deltas

    return {d.horizon_mins: d.delta_fraction for d in deltas}


def apply_adjuster_values(
    da_forecast: xr.DataArray,
    adjuster_values: dict[int, float],
    effective_capacity_watts: float,
    model_name: str,
) -> xr.DataArray:
    """Apply adjuster values to a forecast DataArray."""
    horizon_mins = da_forecast.horizon_mins.values.tolist()
    adjuster_values_array = np.zeros(len(horizon_mins), dtype=float)
    missing_horizon_mins: list[int] = []

    for i, h in enumerate(horizon_mins):
        if h in adjuster_values:
            adjuster_values_array[i] = adjuster_values[h]
        else:
            missing_horizon_mins.append(h)

    if missing_horizon_mins:
        if len(adjuster_values) == 0:
            logger.info(
                "%s: no adjuster history found; using 0.0 for all horizon_mins=%s",
                model_name,
                missing_horizon_mins,
            )
        else:
            logger.warning(
                "%s: no adjuster values found for horizon_mins=%s; using 0.0 as default",
                model_name,
                missing_horizon_mins,
            )

    # Apply absolute limit to the adjuster
    absolute_limit = ADJUSTER_LIMIT_ABSOLUTE_WATTS / effective_capacity_watts
    adjuster_values_array = np.clip(adjuster_values_array, -absolute_limit, absolute_limit)

    # Apply fraction limit to the adjuster
    fraction_limit = da_forecast.sel(output_label="p50").values * ADJUSTER_LIMIT_FORECAST_FRACTION
    adjuster_values_array = np.clip(adjuster_values_array, -fraction_limit, fraction_limit)

    da_adjuster_values = xr.DataArray(
        data=adjuster_values_array,
        dims=["valid_times_utc"],
        coords={"valid_times_utc": da_forecast.valid_times_utc},
    )

    # Adjuster values are the average of (forecast - observed) so we need to subtract
    # Also force the adjusted forecast to be positive by clipping at 0.0
    da_adjusted_forecast = (da_forecast - da_adjuster_values).clip(0, None)

    return da_adjusted_forecast


async def calculate_adjusted_forecast(
    client: dp.DataPlatformDataServiceStub,
    location: dp.ListLocationsResponseLocationSummary,
    init_time_utc: pd.Timestamp,
    da_forecast: xr.DataArray,
    forecaster: dp.Forecaster,
    model_name: str,
) -> xr.DataArray:
    """Make an adjusted forecast based on week average deltas."""
    adjuster_values = await fetch_adjuster_values(
        client=client,
        location_uuid=location.location_uuid,
        init_time_utc=init_time_utc,
        forecaster=forecaster,
    )

    da_adjusted_forecast = apply_adjuster_values(
        da_forecast=da_forecast,
        adjuster_values=adjuster_values,
        effective_capacity_watts=location.effective_capacity_watts,
        model_name=model_name,
    )

    return da_adjusted_forecast
