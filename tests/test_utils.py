from unittest.mock import patch

import pandas as pd
import pytest

from pvnet_app.models.registry import HuggingFaceCommit, ModelSpec
from pvnet_app.utils import check_model_runs_finished, resolve_t0


def test_check_model_runs_finished():
    null_commit = HuggingFaceCommit(repo="dummy", commit="dummy")
    kwargs = {"pvnet": null_commit, "summation": null_commit, "log_level": "INFO"}
    model_specs = [
        ModelSpec(name="pvnet_v2", is_critical=True, **kwargs),
        ModelSpec(name="pvnet_test", is_critical=False, **kwargs),
    ]

    # 1. In this scenario, the critical model has been run but the non-critical model has not
    completed_forecasts = ["pvnet_v2"]

    # This should not raise an exception since we only check critical models
    check_model_runs_finished(completed_forecasts, model_specs, raise_if_missing="critical")

    # This should raise an exception since we are checking all models
    with pytest.raises(ValueError):
        check_model_runs_finished(completed_forecasts, model_specs, raise_if_missing="any")

    # 2. In this scenario, the critical model has failed but the non-critical model has run
    completed_forecasts = ["pvnet_test"]

    # This should raise an exception since the critical model has not been run
    with pytest.raises(ValueError):
        check_model_runs_finished(completed_forecasts, model_specs, raise_if_missing="critical")

    # This should raise an exception since a model has not been run
    with pytest.raises(ValueError):
        check_model_runs_finished(completed_forecasts, model_specs, raise_if_missing="any")

    # 3. In this scenario, both models have been run
    completed_forecasts = ["pvnet_v2", "pvnet_test"]

    # These should not raise an exception since all models have been run
    check_model_runs_finished(completed_forecasts, model_specs, raise_if_missing="critical")
    check_model_runs_finished(completed_forecasts, model_specs, raise_if_missing="any")


def test_resolve_t0_floors_naive_timestamp() -> None:
    result = resolve_t0(pd.Timestamp("2025-01-01 10:44:59"))

    assert result == pd.Timestamp("2025-01-01 10:30:00")
    assert result.tzinfo is None


def test_resolve_t0_converts_aware_timestamp_to_naive_utc() -> None:
    result = resolve_t0(pd.Timestamp("2025-01-01 10:44:59+01:00"))

    assert result == pd.Timestamp("2025-01-01 09:30:00")
    assert result.tzinfo is None


def test_resolve_t0_accepts_string_input() -> None:
    result = resolve_t0("2025-01-01T10:44:59Z")

    assert result == pd.Timestamp("2025-01-01 10:30:00")
    assert result.tzinfo is None


def test_resolve_t0_uses_current_time_when_none() -> None:
    fake_now = pd.Timestamp("2025-01-01 10:44:59", tz="UTC")

    with patch("pvnet_app.utils.pd.Timestamp.now", return_value=fake_now):
        result = resolve_t0(None)

    assert result == pd.Timestamp("2025-01-01 10:30:00")
    assert result.tzinfo is None
