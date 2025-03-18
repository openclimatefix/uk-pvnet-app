import os
import fsspec
import torch
import logging

from ocf_datapipes.batch import NumpyBatch

from pvnet_app.model_configs.pydantic_models import ModelConfig

logger = logging.getLogger()


def get_boolean_env_var(env_var: str, default: bool) -> bool:
    """Get a boolean environment variable.

    Args:
        env_var: The name of the environment variable.
        default: The default value to use if the environment variable is not set.

    Returns:
        The boolean value of the environment variable.
    """
    if env_var in os.environ:
        env_var_value = os.getenv(env_var).lower()
        assert env_var_value in ["true", "false"]
        return env_var_value == "true"
    else:
        return default
    

def save_batch_to_s3(batch: NumpyBatch, model_name: str, s3_directory: str):
    """Saves a batch to a local file and uploads it to S3.

    Args:
        batch: The data batch to save
        model_name: The name of the model
        s3_directory: The S3 directory to save the batch to
    """
    save_batch = f"{model_name}_latest_batch.pt"
    torch.save(batch, save_batch)

    try:
        fs = fsspec.open(s3_directory).fs
        fs.put(save_batch, f"{s3_directory}/{save_batch}")
        logger.info(f"Saved first batch for model {model_name} to {s3_directory}/{save_batch}")
        os.remove(save_batch)
    except Exception as e:
        logger.error(f"Failed to save batch to {s3_directory}/{save_batch} with error {e}")


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
    
    if raise_if_missing=="any":
        required_forecasts = set([model_config.name for model_config in model_configs])
        failed_forecasts = required_forecasts - set(completed_forecasts)
        message = "The following models failed to run"
    
    elif raise_if_missing=="critical":
        required_forecasts = set(
            [model_config.name for model_config in model_configs if model_config.is_critical]
        )
        failed_forecasts = required_forecasts - set(completed_forecasts)
        message = "The following critical models failed to run"
    
    else:
        raise ValueError(f"Invalid value for raise_if_missing: {raise_if_missing}")
    
    if len(failed_forecasts)>0:
        raise Exception(f"{message}: {failed_forecasts}")