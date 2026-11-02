"""Candidates: named transforms of the replayed deltas or of the adjusted forecast.

A "deltas" candidate maps the week-average deltas to new deltas (applied with the production caps
afterwards). A "forecast" candidate maps the adjusted forecast matrix to a new one. Names can be
stacked with "+", left to right, deltas steps first: "w9_gain08+run_blend".
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from replay_harness.metrics import init_step, run_shift
from replay_harness.replay import apply_adjuster, smooth_across_horizon, window_bins

REGISTRY: dict[str, tuple[str, Callable]] = {}


def candidate(kind: str) -> Callable[[Callable], Callable]:
    """Register a function under its own name; kind is "deltas" or "forecast"."""

    def register(fn: Callable) -> Callable:
        REGISTRY[fn.__name__] = (kind, fn)
        return fn

    return register


@candidate("deltas")
def current(deltas: pd.DataFrame) -> pd.DataFrame:
    """The production mechanism as replayed: raw week-average deltas."""
    return deltas


@candidate("deltas")
def w9(deltas: pd.DataFrame) -> pd.DataFrame:
    """Triangular smoothing across horizon (9 bins at 30-minute spacing)."""
    return smooth_across_horizon(deltas, window_bins(deltas.columns))


@candidate("deltas")
def w9_gain08(deltas: pd.DataFrame) -> pd.DataFrame:
    """Smoothing, then apply only 80% of the correction (before the caps, as in production)."""
    return w9(deltas) * 0.8


@candidate("deltas")
def longer_memory(deltas: pd.DataFrame) -> pd.DataFrame:
    """0.4 * latest delta + 0.6 * mean of the previous 15 deltas at the same time of day."""
    time_of_day = deltas.index.hour * 60 + deltas.index.minute
    trail = deltas.groupby(time_of_day).transform(
        lambda s: s.shift(1).rolling(15, min_periods=7).mean(),
    )
    return 0.4 * deltas + 0.6 * trail.fillna(deltas)


@candidate("forecast")
def run_blend(forecast: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """g_t = alpha * f_t + (1 - alpha) * g_{t-1} on shared target times (consecutive inits)."""
    index, values = forecast.index, forecast.to_numpy(dtype=float)
    step, shift = init_step(index), run_shift(index, forecast.columns)
    out = values.copy()
    for i in range(1, len(index)):
        if index[i] - index[i - 1] != step:
            continue
        previous = np.full(values.shape[1], np.nan)
        previous[:-shift] = out[i - 1][shift:]  # g_prev at valid time v is its column h + step
        blended = alpha * values[i] + (1 - alpha) * previous
        out[i] = np.where(np.isfinite(previous), blended, values[i])
    return pd.DataFrame(out, index=index, columns=forecast.columns)


def check(name: str) -> None:
    """Raise ValueError for an unknown step name or a deltas step after a forecast step."""
    seen_forecast = False
    for step in name.split("+"):
        if step not in REGISTRY:
            raise ValueError(f"unknown candidate {step!r}; known: {sorted(REGISTRY)}")
        if REGISTRY[step][0] == "deltas" and seen_forecast:
            raise ValueError(f"{name!r}: deltas step {step!r} after a forecast step")
        seen_forecast |= REGISTRY[step][0] == "forecast"


def build(
    name: str,
    base: pd.DataFrame,
    deltas: pd.DataFrame,
    capacity_watts: float,
) -> pd.DataFrame:
    """Forecast matrix for a (possibly stacked) candidate name."""
    check(name)
    forecast = None
    for step in name.split("+"):
        kind, fn = REGISTRY[step]
        if kind == "deltas":
            deltas = fn(deltas)
        else:
            if forecast is None:
                forecast = apply_adjuster(base, deltas, capacity_watts)
            forecast = fn(forecast)
    return forecast if forecast is not None else apply_adjuster(base, deltas, capacity_watts)
