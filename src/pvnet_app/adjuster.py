"""Functions to create the adjusted forecast based on recent errors."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from ocf import dp

from pvnet_app.utils import convert_to_utc_datetime

logger = logging.getLogger(__name__)


async def fetch_adjuster_values(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    init_time_utc: pd.Timestamp,
    forecaster: dp.Forecaster,
) -> dict[int, float]:
    """Make a forecaster adjuster based on week average deltas."""
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
) -> xr.DataArray:
    """Apply adjuster values to a forecast DataArray."""
    adjuster_values_array = np.zeros(len(da_forecast.horizon_mins.values.tolist()), dtype=float)
    for i, h in enumerate(da_forecast.horizon_mins.values.tolist()):
        if h in adjuster_values:
            adjuster_values_array[i] = adjuster_values[h]
        else:
            logger.warning(f"No adjuster value found for horizon_mins={h}; using 0.0 as default")

    # Limit adjuster values to be no more than 1 GW
    frac_1gw = 1e9 / effective_capacity_watts
    adjuster_values_array = np.clip(adjuster_values_array, -frac_1gw, frac_1gw)

    # Limit adjuster values to be no more than 10% of the forecast value
    frac_10pc = da_forecast.sel(output_label="p50").values * 0.1
    adjuster_values_array = np.clip(adjuster_values_array, -frac_10pc, frac_10pc)

    da_adjuster_values = xr.DataArray(
        data=adjuster_values_array,
        dims=["valid_times_utc"],
        coords={"valid_times_utc": da_forecast.valid_times_utc},
    )

    # Adjuster values are the average of (forecast - observed) so we need to subtract
    # Also force the adjusted forecast to be positive by clipping at 0.0
    da_adjusted_forecast = (da_forecast - da_adjuster_values).clip(0, 1)

    return da_adjusted_forecast


async def calculate_adjusted_forecast(
    client: dp.DataPlatformDataServiceStub,
    location: dp.ListLocationsResponseLocationSummary,
    init_time_utc: pd.Timestamp,
    da_forecast: xr.DataArray,
    forecaster: dp.Forecaster,
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
    )

    return da_adjusted_forecast
