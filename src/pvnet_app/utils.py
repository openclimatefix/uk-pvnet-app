"""General utility functions."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

import fsspec
import pandas as pd
import torch
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch

from pvnet_app.models.registry import ModelSpec

logger = logging.getLogger(__name__)


@contextmanager
def log_duration(
    logger: logging.Logger, activity: str, level: int = logging.INFO
) -> Iterator[None]:
    """Log an activity start and completion time."""
    start = perf_counter()
    logger.log(level, "%s", activity)
    try:
        yield
    finally:
        logger.log(level, "%s completed in %.2fs", activity, perf_counter() - start)


def save_batch_to_s3(
    batch: NumpyBatch,
    model_name: str,
    s3_directory: str,
    scratch_dir: str,
) -> None:
    """Saves a batch to a local file and uploads it to S3.

    Args:
        batch: The data batch to save
        model_name: The name of the model
        s3_directory: The S3 directory to save the batch to
        scratch_dir: The local directory to save the batch to before uploading
    """
    filename = f"{model_name}_latest_batch.pt"
    local_path = f"{scratch_dir}/{filename}"
    torch.save(batch, local_path)

    try:
        fs = fsspec.open(s3_directory).fs
        fs.put(local_path, f"{s3_directory}/{filename}")
    except Exception as e:
        logger.error(f"Failed to save batch to {s3_directory}/{filename} with error {e}")


def check_model_runs_finished(
    completed_forecasts: list[str],
    model_specs: list[ModelSpec],
    raise_if_missing: str,
) -> None:
    """Check if the required models have been run and raise an exception if not.

    Args:
        completed_forecasts: List of forecast names which have been completed
        model_specs: List of model specifications
        raise_if_missing: If set to "any", any missing model will raise an exception.
            If set to "critical", only missing critical models will raise an exception.
    """
    if raise_if_missing == "any":
        required_forecasts = {model_spec.name for model_spec in model_specs}
        failed_forecasts = required_forecasts - set(completed_forecasts)
        message = "The following models failed to run"

    elif raise_if_missing == "critical":
        required_forecasts = {
            model_spec.name for model_spec in model_specs if model_spec.is_critical
        }
        failed_forecasts = required_forecasts - set(completed_forecasts)
        message = "The following critical models failed to run"

    else:
        raise ValueError(
            f"Invalid value for raise_if_missing: {raise_if_missing}. "
            "Should be 'any' or 'critical'",
        )

    if len(failed_forecasts) > 0:
        raise ValueError(
            f"{message}: {failed_forecasts}. Completed forecasts: {completed_forecasts}",
        )


def resolve_t0(t0: str | datetime | pd.Timestamp | None) -> pd.Timestamp:
    """Resolve any accepted t0 input to a timezone-naive UTC timestamp floored to 30 minutes.

    Naive inputs are assumed to already be in UTC; timezone-aware inputs are converted to UTC before
    the zone is dropped.

    Args:
        t0: The input timestamp. If None, the current time is used.
    """
    t0 = pd.Timestamp.now(tz="UTC") if t0 is None else pd.Timestamp(t0)

    # If the input is timezone-aware, convert to UTC and then make it naive
    if t0.tzinfo is not None:
        t0 = t0.tz_convert("UTC").tz_localize(None)
    return t0.floor("30min")


def convert_to_utc_datetime(ts: pd.Timestamp) -> datetime:
    """Converts internal naive UTC timestamp to aware UTC datetime."""
    return ts.tz_localize("UTC").to_pydatetime()
