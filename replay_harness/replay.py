"""Replay of the production adjuster (week-average deltas, smoothing, gain, caps).

Mirrors src/pvnet_app/adjuster.py without importing it. Capacity is the final (latest) effective
capacity, in watts, for the absolute cap.
"""

import numpy as np
import pandas as pd

ABSOLUTE_LIMIT_WATTS = 1e9
FRACTION_LIMIT = 0.1
SMOOTHING_WINDOW_MINS = 270
OBS_FLOOR_HOURS = 36  # day-after PVLive for day D is usable from D+1 12:00 UTC


def apply_floor(obs: pd.DataFrame, hours: int = OBS_FLOOR_HOURS) -> pd.DataFrame:
    """created_time = max(stored created time, obs day midnight + `hours`); missing -> floor."""
    floor = pd.Series(obs.index.normalize() + pd.Timedelta(hours=hours), index=obs.index)
    out = obs.copy()
    created = out["created_time_utc"]
    out["created_time_utc"] = created.where(created > floor, floor)
    return out


def _naive_ns(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """UTC datetimes (Series or Index) as a naive datetime64[ns] array; NaT stays NaT."""
    t = pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None)
    return t.to_numpy(dtype="datetime64[ns]")


def replay_deltas(
    mat: pd.DataFrame,
    obs: pd.DataFrame,
    inits: pd.DatetimeIndex,
    lookback_days: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-init, per-horizon week-average deltas as production computes them.

    For init t0, the deltas average (forecast - observation) over the same forecaster's runs at
    t0 - 1d ... t0 - lookback_days (same time of day), using only observations whose created time
    is <= t0. Returns (deltas, counts), both indexed by `inits`, columns = horizons. Horizons
    with no usable sample are NaN with count 0.
    """
    value, created = obs["value_fraction"], obs["created_time_utc"]
    err = np.empty(mat.shape)
    avail = np.empty(mat.shape, dtype="datetime64[ns]")  # when each (init, horizon) obs exists
    for j, h in enumerate(mat.columns):
        target = mat.index + pd.Timedelta(minutes=int(h))
        err[:, j] = mat[h].to_numpy() - value.reindex(target).to_numpy()
        avail[:, j] = _naive_ns(created.reindex(target))
    err_df = pd.DataFrame(err, index=mat.index, columns=mat.columns)
    avail_df = pd.DataFrame(avail, index=mat.index, columns=mat.columns)

    t0 = _naive_ns(inits)[:, None]
    total = np.zeros((len(inits), mat.shape[1]))
    count = np.zeros_like(total)
    for lag in range(1, lookback_days + 1):
        past = inits - pd.Timedelta(days=lag)
        e = err_df.reindex(past).to_numpy()
        usable = np.isfinite(e) & (avail_df.reindex(past).to_numpy() <= t0)  # NaT compares False
        total += np.where(usable, e, 0.0)
        count += usable
    with np.errstate(invalid="ignore", divide="ignore"):
        deltas = np.where(count > 0, total / count, np.nan)
    return (
        pd.DataFrame(deltas, index=inits, columns=mat.columns),
        pd.DataFrame(count.astype(int), index=inits, columns=mat.columns),
    )


def window_bins(horizons: pd.Index, minutes: int = SMOOTHING_WINDOW_MINS) -> int:
    """Odd number of horizon bins spanning `minutes` (9 at 30-minute spacing), at least 1."""
    half = round((minutes / (horizons[1] - horizons[0]) - 1) / 2)
    return 2 * max(half, 0) + 1


def smooth_across_horizon(deltas: pd.DataFrame, window: int) -> pd.DataFrame:
    """Centred triangular smoothing along the horizon axis, ignoring NaNs.

    Weights are 1, 2, ..., half+1, ..., 2, 1 for window = 2*half+1. Each position uses only the
    non-NaN entries inside the window, renormalised by the weights actually used; positions with
    an all-NaN window stay NaN.
    """
    half = window // 2
    values = deltas.to_numpy(dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    n = values.shape[1]
    num, den = np.zeros_like(filled), np.zeros_like(filled)
    for k in range(-half, half + 1):
        w = half + 1 - abs(k)
        lo, hi = max(0, -k), min(n, n - k)  # output columns whose source column j+k exists
        if lo >= hi:
            continue
        num[:, lo:hi] += w * filled[:, lo + k : hi + k]
        den[:, lo:hi] += w * valid[:, lo + k : hi + k]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(out, index=deltas.index, columns=deltas.columns)


def apply_adjuster(mat: pd.DataFrame, deltas: pd.DataFrame, capacity_watts: float) -> pd.DataFrame:
    """Subtract deltas from the forecast as the app does.

    Order: NaN -> 0, absolute cap (1e9 W as a fraction of capacity), +-10% of the forecast p50,
    subtract, clip at 0. Any smoothing or gain belongs in `deltas` before this call.
    """
    f = mat.loc[deltas.index]
    absolute = ABSOLUTE_LIMIT_WATTS / capacity_watts
    d = deltas.fillna(0.0).clip(-absolute, absolute)
    limit = f * FRACTION_LIMIT
    d = pd.DataFrame(
        np.clip(d.to_numpy(), -limit.to_numpy(), limit.to_numpy()), index=d.index, columns=d.columns
    )
    return (f - d).clip(lower=0.0)
