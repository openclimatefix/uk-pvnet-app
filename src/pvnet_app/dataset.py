from pathlib import Path
import pandas as pd

from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKConcurrentDataset


from pvnet_app.config import modify_data_config_for_production


def get_dataset(
    config_filename: str,
    gsp_ids: list[int],
) -> PVNetUKConcurrentDataset:
    """Construct the dataset for the given configuration
    
    Args:
        config_filename: The path to the configuration file
        gsp_ids: The GSP IDs to forecast for
    """

    # Populate the data config with production data paths
    modified_data_config_filename = Path(config_filename).parent / "data_config.yaml"

    modify_data_config_for_production(
        input_path=config_filename,
        output_path=modified_data_config_filename,
    )

    return PVNetUKConcurrentDataset(
        config_filename=modified_data_config_filename,
        gsp_ids=gsp_ids,
    )
