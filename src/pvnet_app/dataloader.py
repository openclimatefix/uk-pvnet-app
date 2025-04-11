from pathlib import Path
import pandas as pd

from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKRegionalDataset


from torch.utils.data import DataLoader

from pvnet_app.config import modify_data_config_for_production


def get_dataloader(
    config_filename: str,
    t0: pd.Timestamp,
    gsp_ids: list[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Construct the dataloader for the given configuration
    
    Args:
        config_filename: The path to the configuration file
        t0: The init-time of the forecast
        gsp_ids: The GSP IDs to forecast for
        batch_size: The batch size to use
        num_workers: The number of workers to use
    """

    # Populate the data config with production data paths
    modified_data_config_filename = Path(config_filename).parent / "data_config.yaml"

    modify_data_config_for_production(
        input_path=config_filename,
        output_path=modified_data_config_filename,
    )

    dataset = PVNetUKRegionalDataset(
        config_filename=modified_data_config_filename,
        start_time=t0,
        end_time=t0,
        gsp_ids=gsp_ids,
    )

    return DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=stack_np_samples_into_batch,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=False,
    )
