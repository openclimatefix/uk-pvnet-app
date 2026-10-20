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

# The per-horizon adjuster values are noisy, independent weekly means, so they are smoothed across
# horizon before use. Total width in minutes of the centered triangular smoothing window; the
# window in bins is derived from the forecast's horizon spacing (9 bins at 30-minute steps).
ADJUSTER_SMOOTHING_WINDOW_MINS: int = 270

# The week-average correction is a noisy estimate of the next day's error, so only this fraction
# of it is applied: full strength chases last week's noise as well as the systematic error.
ADJUSTER_GAIN: float = 0.8


def smoothing_window_bins(horizon_mins: list[int]) -> int:
    """Odd number of horizon bins spanning ADJUSTER_SMOOTHING_WINDOW_MINS, at least 1."""
    if len(horizon_mins) < 2:
        return 1
    spacing_mins = horizon_mins[1] - horizon_mins[0]
    half = round((ADJUSTER_SMOOTHING_WINDOW_MINS / spacing_mins - 1) / 2)
    return 2 * max(half, 0) + 1


def smooth_across_horizon(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a 1-D array with a centered triangular window of `window` bins, ignoring NaNs.

    At each position only the non-NaN entries inside the window are used, and the weights are
    renormalised by the weights actually used. Positions whose whole window is NaN stay NaN.
    """
    if len(values) == 0:
        return values.astype(float)

    half = window // 2
    weights = (half + 1 - np.abs(np.arange(-half, half + 1))).astype(float)
    valid = ~np.isnan(values)

    # Slide the weights along for a weighted sum per bin, sliced so short inputs also work
    centre = slice(half, half + len(values))
    weighted_sum = np.convolve(np.where(valid, values, 0.0), weights, mode="full")[centre]
    weight_total = np.convolve(valid.astype(float), weights, mode="full")[centre]

    smoothed = np.full(len(values), np.nan, dtype=float)
    np.divide(weighted_sum, weight_total, out=smoothed, where=weight_total > 0)
    return smoothed


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
    adjuster_values_array = np.full(len(horizon_mins), np.nan, dtype=float)
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
                "%s: no adjuster values found for horizon_mins=%s; "
                "filling from neighbouring horizons",
                model_name,
                missing_horizon_mins,
            )

    # Smooth across horizon, then use 0.0 where there are no values at all
    window = smoothing_window_bins(horizon_mins)
    adjuster_values_array = np.nan_to_num(
        smooth_across_horizon(adjuster_values_array, window),
        nan=0.0,
    )

    # Damp the correction
    adjuster_values_array = adjuster_values_array * ADJUSTER_GAIN

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
