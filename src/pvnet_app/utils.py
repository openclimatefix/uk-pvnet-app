import os
import fsspec
import torch
import logging

from ocf_datapipes.batch import NumpyBatch


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