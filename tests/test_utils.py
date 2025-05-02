import pytest

from pvnet_app.model_configs.pydantic_models import HuggingFaceCommit, ModelConfig
from pvnet_app.utils import check_model_runs_finished


def test_check_model_runs_finished():

    null_commit = HuggingFaceCommit(repo="dummy", commit="dummy")

    model_configs = [
        ModelConfig(
            name="pvnet_v2",
            is_critical=True,
            pvnet=null_commit,
            summation=null_commit,
        ),
        ModelConfig(
            name="pvnet_test",
            is_critical=False,
            pvnet=null_commit,
            summation=null_commit,
        ),
    ]

    # 1. In this scenario, the critical model has been run but the non-critical model has not
    completed_forecasts = ["pvnet_v2"]

    # This should not raise an exception since we only check critical models
    check_model_runs_finished(
        completed_forecasts,
        model_configs,
        raise_if_missing="critical",
    )

    # This should raise an exception since we are checking all models
    with pytest.raises(Exception):
        check_model_runs_finished(
            completed_forecasts,
            model_configs,
            raise_if_missing="any",
        )

    # 2. In this scenario, the critical model has failed but the non-critical model has run
    completed_forecasts = ["pvnet_test"]

    # This should raise an exception since the critical model has not been run
    with pytest.raises(Exception):
        check_model_runs_finished(
            completed_forecasts,
            model_configs,
            raise_if_missing="critical",
        )

    # This should raise an exception since a model has not been run
    with pytest.raises(Exception):
        check_model_runs_finished(
            completed_forecasts,
            model_configs,
            raise_if_missing="any",
        )

    # 3. In this scenario, both models have been run
    completed_forecasts = ["pvnet_v2", "pvnet_test"]

    # These should not raise an exception since all models have been run
    check_model_runs_finished(
        completed_forecasts,
        model_configs,
        raise_if_missing="critical",
    )
    check_model_runs_finished(
        completed_forecasts,
        model_configs,
        raise_if_missing="any",
    )
