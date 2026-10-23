import numpy as np
import pandas as pd
import xarray as xr

from pvnet_app.adjuster import (
    ADJUSTER_GAIN,
    apply_adjuster_values,
    smooth_across_horizon,
    smoothing_window_bins,
)

WINDOW = 9
HALF = WINDOW // 2


def test_smooth_constant_unchanged():
    values = np.full(30, 0.03)
    np.testing.assert_allclose(smooth_across_horizon(values, WINDOW), values)


def test_smooth_linear_ramp_unchanged_in_interior():
    values = np.linspace(-0.2, 0.3, 30)
    smoothed = smooth_across_horizon(values, WINDOW)

    np.testing.assert_allclose(smoothed[HALF:-HALF], values[HALF:-HALF])
    # Edges are renormalised one-sided averages, so they are pulled towards the interior
    assert smoothed[0] > values[0]
    assert smoothed[-1] < values[-1]


def test_smooth_single_spike_is_spread_and_reduced():
    values = np.zeros(21)
    values[10] = 1.0
    smoothed = smooth_across_horizon(values, WINDOW)

    weights = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]) / 25
    np.testing.assert_allclose(smoothed[10 - HALF : 10 + HALF + 1], weights)
    assert smoothed.max() < values.max()
    assert smoothed[10 - 1] > 0
    assert smoothed[10 - HALF - 1] == 0
    assert np.isclose(smoothed.sum(), 1.0)


def test_smooth_ignores_isolated_nan():
    values = np.full(21, 0.02)
    values[10] = np.nan
    smoothed = smooth_across_horizon(values, WINDOW)

    assert not np.isnan(smoothed).any()
    np.testing.assert_allclose(smoothed, 0.02)


def test_smooth_all_nan_window_is_nan_then_zero_after_fill():
    values = np.full(30, np.nan)
    values[:3] = 0.1
    smoothed = smooth_across_horizon(values, WINDOW)

    # Positions further than HALF bins from any value have an all-NaN window
    assert np.isnan(smoothed[3 + HALF :]).all()
    np.testing.assert_allclose(smoothed[: 3 + HALF], 0.1)
    assert (np.nan_to_num(smoothed, nan=0.0)[3 + HALF :] == 0.0).all()


def test_smooth_array_shorter_than_window():
    np.testing.assert_allclose(smooth_across_horizon(np.array([0.5]), WINDOW), [0.5])
    np.testing.assert_allclose(smooth_across_horizon(np.full(3, 0.2), WINDOW), 0.2)

    smoothed = smooth_across_horizon(np.array([0.0, 1.0, 2.0]), WINDOW)
    # Position 0 uses weights 5, 4, 3 on 0, 1, 2
    assert np.isclose(smoothed[0], (4 * 1.0 + 3 * 2.0) / 12)
    assert np.isclose(smoothed[1], 1.0)
    assert smoothed.shape == (3,)

    assert smooth_across_horizon(np.array([]), WINDOW).shape == (0,)


def _make_forecast(horizon_mins: list[int], p50: float) -> xr.DataArray:
    n = len(horizon_mins)
    return xr.DataArray(
        data=np.full((n, 1), p50),
        dims=["valid_times_utc", "output_label"],
        coords={
            "valid_times_utc": pd.date_range("2025-01-01", periods=n, freq="30min"),
            "output_label": ["p50"],
            "horizon_mins": ("valid_times_utc", horizon_mins),
        },
    )


def test_apply_adjuster_values_constant_deltas():
    horizon_mins = [i * 30 for i in range(20)]
    da_forecast = _make_forecast(horizon_mins, p50=0.5)

    result = apply_adjuster_values(
        da_forecast=da_forecast,
        adjuster_values=dict.fromkeys(horizon_mins, 0.02),
        effective_capacity_watts=1_000,
        model_name="test",
    )

    # 0.5 - ADJUSTER_GAIN * 0.02
    np.testing.assert_allclose(result.sel(output_label="p50").values, 0.484)


def test_apply_adjuster_values_no_values_leaves_forecast_unchanged():
    horizon_mins = [i * 30 for i in range(20)]
    da_forecast = _make_forecast(horizon_mins, p50=0.5)

    result = apply_adjuster_values(
        da_forecast=da_forecast,
        adjuster_values={},
        effective_capacity_watts=1_000,
        model_name="test",
    )

    np.testing.assert_allclose(result.values, da_forecast.values)


def test_apply_adjuster_values_missing_horizons_are_filled_by_smoothing():
    horizon_mins = [i * 30 for i in range(20)]
    da_forecast = _make_forecast(horizon_mins, p50=0.5)
    adjuster_values = dict.fromkeys(horizon_mins, 0.02)
    del adjuster_values[300]

    result = apply_adjuster_values(
        da_forecast=da_forecast,
        adjuster_values=adjuster_values,
        effective_capacity_watts=1_000,
        model_name="test",
    )

    np.testing.assert_allclose(result.sel(output_label="p50").values, 0.484)


def test_apply_adjuster_values_damped_by_gain():
    horizon_mins = [i * 30 for i in range(20)]
    da_forecast = _make_forecast(horizon_mins, p50=0.5)

    result = apply_adjuster_values(
        da_forecast=da_forecast,
        adjuster_values=dict.fromkeys(horizon_mins, 0.01),
        effective_capacity_watts=1_000,
        model_name="test",
    )

    applied = 0.5 - result.sel(output_label="p50").values
    np.testing.assert_allclose(applied, ADJUSTER_GAIN * 0.01)


def test_smoothing_window_bins():
    assert smoothing_window_bins([i * 30 for i in range(10)]) == 9
    assert smoothing_window_bins([i * 60 for i in range(10)]) == 5
    assert smoothing_window_bins([i * 15 for i in range(40)]) == 17
    assert smoothing_window_bins([30]) == 1
    assert smoothing_window_bins([]) == 1
    assert smoothing_window_bins([i * 1440 for i in range(5)]) == 1
