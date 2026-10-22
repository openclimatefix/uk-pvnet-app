"""The replay must mirror the production adjuster exactly.

replay.py deliberately reimplements src/pvnet_app/adjuster.py without importing it,
so the harness stays portable. This test is the enforcement: it runs both
implementations on randomised inputs and requires bit-level agreement, so a change
to the production mechanism that is not mirrored in replay.py fails CI here.

It skips until the production adjuster has the smoothing functions (they arrive
with the smooth-adjuster-deltas PR); before that there is no mirror to check.
"""

import numpy as np
import pandas as pd
import pytest

from replay_harness import replay

pvnet_adjuster = pytest.importorskip("pvnet_app.adjuster")

if not hasattr(pvnet_adjuster, "smooth_across_horizon"):
    pytest.skip(
        "production adjuster has no smoothing yet; this check activates once the "
        "smooth-adjuster-deltas PR is merged",
        allow_module_level=True,
    )

xr = pytest.importorskip("xarray")

RNG = np.random.default_rng(42)


def _random_values(n: int) -> np.ndarray:
    values = RNG.normal(0.0, 0.05, n)
    values[RNG.random(n) < 0.15] = np.nan
    return values


@pytest.mark.parametrize("n", [1, 3, 9, 75])
def test_smoothing_matches_production(n: int):
    values = _random_values(n)
    window = replay.window_bins(pd.Index(range(0, 30 * n, 30))) if n > 1 else 1
    ours = replay.smooth_across_horizon(
        pd.DataFrame([values], columns=range(0, 30 * n, 30)), window
    ).to_numpy()[0]
    theirs = pvnet_adjuster.smooth_across_horizon(values.copy(), window)
    # the two implementations sum identical weighted terms in different orders,
    # so agreement is to float summation order (~1e-17), not bit-for-bit
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-15)


@pytest.mark.parametrize("spacing", [15, 30, 60, 1440])
def test_window_bins_match_production(spacing: int):
    horizons = [i * spacing for i in range(10)]
    assert replay.window_bins(pd.Index(horizons)) == pvnet_adjuster.smoothing_window_bins(horizons)


def test_full_mechanism_matches_production():
    """End to end per run: fill, smooth, gain, caps, clip, on randomised inputs."""
    horizons = list(range(0, 2190, 30))
    capacity_watts = 22.5e9
    gain = getattr(pvnet_adjuster, "ADJUSTER_GAIN", 1.0)
    for _ in range(20):
        forecast = np.abs(RNG.normal(0.3, 0.2, len(horizons)))
        deltas = _random_values(len(horizons))
        # occasionally huge deltas so both caps and the clip engage
        deltas[RNG.random(len(horizons)) < 0.05] = RNG.choice([-1.0, 1.0]) * 0.5

        da = xr.DataArray(
            data=forecast[:, None],
            dims=["valid_times_utc", "output_label"],
            coords={
                "valid_times_utc": pd.date_range("2026-01-01", periods=len(horizons), freq="30min"),
                "output_label": ["p50"],
                "horizon_mins": ("valid_times_utc", horizons),
            },
        )
        values = dict(zip(horizons, deltas, strict=True))
        theirs = (
            pvnet_adjuster.apply_adjuster_values(
                da_forecast=da,
                adjuster_values={h: d for h, d in values.items() if np.isfinite(d)},
                effective_capacity_watts=capacity_watts,
                model_name="mirror-test",
            )
            .sel(output_label="p50")
            .to_numpy()
        )

        mat = pd.DataFrame([forecast], columns=horizons)
        d = pd.DataFrame([deltas], columns=horizons)
        smoothed = replay.smooth_across_horizon(d, replay.window_bins(pd.Index(horizons)))
        ours = replay.apply_adjuster(mat, smoothed * gain, capacity_watts).to_numpy()[0]

        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-15)
