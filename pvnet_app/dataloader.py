from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader
from ocf_datapipes.batch import stack_np_examples_into_batch
from ocf_data_sampler.torch_datasets.pvnet_uk_regional import PVNetUKRegionalDataset

from pvnet_app.config import modify_data_config_for_production

# Legacy imports - only used for legacy dataloader
import os
from ocf_datapipes.load import OpenGSPFromDatabase
from torch.utils.data.datapipes.iter import IterableWrapper
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.batch import BatchKey
from pvnet.utils import GSPLocationLookup


def get_dataloader(
    config_filename: str,
    t0: pd.Timestamp,
    gsp_ids: list[int],
    batch_size: int,
    num_workers: int,
):

    # Populate the data config with production data paths
    modified_data_config_filename = Path(config_filename).parent / "data_config.yaml"

    # TODO pass in schema_version to populated_data_config_filename using partial
    modify_data_config_for_production(config_filename, modified_data_config_filename)

    dataset = PVNetUKRegionalDataset(
        config_filename=modified_data_config_filename,
        start_time=t0,
        end_time=t0,
        gsp_ids=gsp_ids,
    )

    # Set up dataloader for parallel loading
    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=batch_size,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=stack_np_examples_into_batch,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=False,
    )

    return DataLoader(dataset, **dataloader_kwargs)


def legacy_squeeze(batch):
    batch[BatchKey.gsp_id] = batch[BatchKey.gsp_id].squeeze(1)
    return batch


def get_legacy_dataloader(
    config_filename: str,
    t0: pd.Timestamp,
    gsp_ids: list[int],
    batch_size: int,
    num_workers: int,
):

    # Populate the data config with production data paths
    populated_data_config_filename = Path(config_filename).parent / "data_config.yaml"

    modify_data_config_for_production(
        config_filename,
        populated_data_config_filename,
        gsp_path=os.environ["DB_URL"],
    )

    # Set up ID location query object
    ds_gsp = next(iter(OpenGSPFromDatabase()))
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)

    # Location and time datapipes
    location_pipe = IterableWrapper([gsp_id_to_loc(gsp_id) for gsp_id in gsp_ids])
    t0_datapipe = IterableWrapper([t0]).repeat(len(location_pipe))

    location_pipe = location_pipe.sharding_filter()
    t0_datapipe = t0_datapipe.sharding_filter()

    # Batch datapipe
    batch_datapipe = (
        construct_sliced_data_pipeline(
            config_filename=populated_data_config_filename,
            location_pipe=location_pipe,
            t0_datapipe=t0_datapipe,
            production=True,
        )
        .batch(batch_size)
        .map(stack_np_examples_into_batch)
        .map(legacy_squeeze)
    )

    # Set up dataloader for parallel loading
    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,  # batched in datapipe step
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=False,
    )

    return DataLoader(batch_datapipe, **dataloader_kwargs)
