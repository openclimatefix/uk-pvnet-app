"""General utility functions for pvnet_app."""
import logging
import os

import fsspec
import torch
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch

from pvnet_app.models.pydantic_models import ModelConfig

logger = logging.getLogger()



def save_batch_to_s3(batch: NumpyBatch, model_name: str, s3_directory: str) -> None:
    """Saves a batch to a local file and uploads it to S3.

    Args:
        batch: The data batch to save
        model_name: The name of the model
        s3_directory: The S3 directory to save the batch to
    """
    save_path = f"{model_name}_latest_batch.pt"
    torch.save(batch, save_path)

    try:
        fs = fsspec.open(s3_directory).fs
        fs.put(save_path, f"{s3_directory}/{save_path}")
        logger.info(f"Saved first batch for model {model_name} to {s3_directory}/{save_path}")
    except Exception as e:
        logger.error(f"Failed to save batch to {s3_directory}/{save_path} with error {e}")
    finally:
        os.remove(save_path)


def check_model_runs_finished(
    completed_forecasts: list[str],
    model_configs: list[ModelConfig],
    raise_if_missing: str,
) -> None:
    """Check if the required models have been run and raise an exception if not.

    Args:
        completed_forecasts: List of forecast names which have been completed
        model_configs: List of model configurations
        raise_if_missing: If set to "any", any missing model will raise an exception.
            If set to "critical", only missing critical models will raise an exception.
    """
    if raise_if_missing == "any":
        required_forecasts = {model_config.name for model_config in model_configs}
        failed_forecasts = required_forecasts - set(completed_forecasts)
        message = "The following models failed to run"

    elif raise_if_missing == "critical":
        required_forecasts = {
            model_config.name for model_config in model_configs if model_config.is_critical
        }
        failed_forecasts = required_forecasts - set(completed_forecasts)
        message = "The following critical models failed to run"

    else:
        raise ValueError(
            f"Invalid value for raise_if_missing: {raise_if_missing}. "
            "Should be 'any' or 'critical'",
        )

    if len(failed_forecasts) > 0:
        raise Exception(
            f"{message}: {failed_forecasts}. Completed forecasts: {completed_forecasts}",
        )
