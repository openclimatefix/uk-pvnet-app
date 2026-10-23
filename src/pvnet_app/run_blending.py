"""Blend each published forecast with the previous run at the same target times.

About half of each run's revision to a given target time is reversed by the run
after it, so averaging consecutive runs cancels revision noise: on a year-long
replay, run-to-run churn roughly halves at a blend weight of 0.5 while MAE
slightly improves, and short horizons are unaffected. The weight applies to the
fresh run; 1.0 disables blending and is the default until a live shadow run has
confirmed the replay result.

The blend is recursive by construction: the previous run fetched here is itself
already blended, matching how the replay evaluated it. A run with no prior run
within RUN_BLEND_MAX_AGE_MINS (first run after an outage) publishes unblended.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from ocf import dp

from pvnet_app.utils import convert_to_utc_datetime

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger(__name__)

# Do not blend with a previous run older than this: after a publishing gap the
# forecast restarts fresh instead of leaning on stale values.
RUN_BLEND_MAX_AGE_MINS: int = 90


def _naive_utc(ts: "pd.Timestamp | np.datetime64 | datetime") -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


async def fetch_previous_run_values(
    client: dp.DataPlatformDataServiceStub,
    forecaster: dp.Forecaster,
    location_uuid: str,
    init_time_utc: pd.Timestamp,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Fetch the most recent published run of `forecaster` before `init_time_utc`.

    Returns {target_time: {output_label: fraction}} for the latest prior init
    within RUN_BLEND_MAX_AGE_MINS, or {} when there is none or the fetch fails —
    the caller then publishes unblended, so blending degrades gracefully.
    """
    window_start = init_time_utc - pd.Timedelta(minutes=RUN_BLEND_MAX_AGE_MINS)
    request = dp.StreamForecastDataRequest(
        location_uuids=[location_uuid],
        energy_source=dp.EnergySource.SOLAR,
        time_window=dp.TimeWindow(
            start_timestamp_utc=convert_to_utc_datetime(window_start),
            end_timestamp_utc=convert_to_utc_datetime(init_time_utc),
        ),
        forecasters=[forecaster],
    )

    values = []
    try:
        async for chunk in client.stream_forecast_data(request):
            values.extend(chunk.values)
    except Exception:
        logger.warning(
            "Run blending: could not fetch previous run; publishing unblended",
            exc_info=True,
        )
        return {}

    prior = [v for v in values if _naive_utc(v.init_timestamp) < _naive_utc(init_time_utc)]
    if not prior:
        return {}

    latest_init = max(_naive_utc(v.init_timestamp) for v in prior)
    out: dict[pd.Timestamp, dict[str, float]] = {}
    for v in prior:
        if _naive_utc(v.init_timestamp) != latest_init:
            continue
        labels = {"p50": v.p50_fraction}
        labels.update(dict(v.other_statistics_fractions))
        out[_naive_utc(v.target_timestamp_utc)] = labels
    return out


def blend_with_previous_run(
    da_forecast: xr.DataArray,
    previous_values: dict[pd.Timestamp, dict[str, float]],
    alpha: float,
    model_name: str,
) -> xr.DataArray:
    """Blend the forecast with the previous run: alpha*fresh + (1-alpha)*previous.

    Target times the previous run did not cover stay fresh, so the newest
    horizon bin and any post-gap run are published as forecast.
    """
    if alpha >= 1.0 or not previous_values:
        return da_forecast

    da = da_forecast.copy(deep=True)
    valid_times = pd.DatetimeIndex([_naive_utc(t) for t in da.valid_times_utc.values])
    n_blended = 0

    for label in [str(lab) for lab in da.output_label.values]:
        prev = pd.Series(
            {t: vals[label] for t, vals in previous_values.items() if label in vals},
            dtype=float,
        )
        if prev.empty:
            continue
        prev_aligned = prev.reindex(valid_times)
        mask = prev_aligned.notna().to_numpy()
        if not mask.any():
            continue
        fresh = da.sel(output_label=label).values
        da.loc[{"output_label": label}] = np.where(
            mask,
            alpha * fresh + (1.0 - alpha) * prev_aligned.to_numpy(),
            fresh,
        )
        n_blended = max(n_blended, int(mask.sum()))

    logger.info(
        "%s: blended %d/%d target times with previous run (alpha=%.2f)",
        model_name,
        n_blended,
        len(valid_times),
        alpha,
    )
    return da
