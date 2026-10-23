import numpy as np
import pandas as pd
import xarray as xr

from pvnet_app.run_blending import blend_with_previous_run

TIMES = pd.date_range("2026-01-01 08:00", periods=4, freq="30min")


def _make_forecast(values_by_label: dict[str, list[float]]) -> xr.DataArray:
    labels = list(values_by_label)
    data = np.array([values_by_label[lab] for lab in labels], dtype=float).T
    return xr.DataArray(
        data=data,
        dims=["valid_times_utc", "output_label"],
        coords={"valid_times_utc": TIMES, "output_label": labels},
    )


def test_alpha_one_is_passthrough():
    da = _make_forecast({"p50": [0.1, 0.2, 0.3, 0.4]})
    prev = {TIMES[0]: {"p50": 0.9}}
    result = blend_with_previous_run(da, prev, alpha=1.0, model_name="test")
    xr.testing.assert_identical(result, da)


def test_empty_previous_is_passthrough():
    da = _make_forecast({"p50": [0.1, 0.2, 0.3, 0.4]})
    result = blend_with_previous_run(da, {}, alpha=0.5, model_name="test")
    xr.testing.assert_identical(result, da)


def test_blends_matching_targets_only():
    da = _make_forecast({"p50": [0.1, 0.2, 0.3, 0.4]})
    prev = {TIMES[1]: {"p50": 0.4}, TIMES[2]: {"p50": 0.1}}
    result = blend_with_previous_run(da, prev, alpha=0.5, model_name="test")
    np.testing.assert_allclose(
        result.sel(output_label="p50").values,
        [0.1, 0.3, 0.2, 0.4],
    )


def test_blends_each_output_label_independently():
    da = _make_forecast({"p50": [0.2, 0.2, 0.2, 0.2], "p90": [0.4, 0.4, 0.4, 0.4]})
    prev = {t: {"p50": 0.0, "p90": 0.8} for t in TIMES}
    result = blend_with_previous_run(da, prev, alpha=0.5, model_name="test")
    np.testing.assert_allclose(result.sel(output_label="p50").values, 0.1)
    np.testing.assert_allclose(result.sel(output_label="p90").values, 0.6)


def test_label_missing_from_previous_stays_fresh():
    da = _make_forecast({"p50": [0.2, 0.2, 0.2, 0.2], "p90": [0.4, 0.4, 0.4, 0.4]})
    prev = {t: {"p50": 0.0} for t in TIMES}
    result = blend_with_previous_run(da, prev, alpha=0.5, model_name="test")
    np.testing.assert_allclose(result.sel(output_label="p50").values, 0.1)
    np.testing.assert_allclose(result.sel(output_label="p90").values, 0.4)
