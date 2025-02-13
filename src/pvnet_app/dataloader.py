# Legacy imports - only used for legacy dataloader
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKRegionalDataset
from ocf_datapipes.batch import BatchKey
from ocf_datapipes.batch import stack_np_examples_into_batch as legacy_stack_np_examples_into_batch
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.utils import Location
from ocf_datapipes.utils.eso import get_gsp_shape_from_eso
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.iter import IterableWrapper

from pvnet_app.config import modify_data_config_for_production


def get_dataloader(
    config_filename: str,
    t0: pd.Timestamp,
    gsp_ids: list[int],
    batch_size: int,
    num_workers: int,
):

    # Populate the data config with production data paths
    modified_data_config_filename = Path(config_filename).parent / "data_config.yaml"

    modify_data_config_for_production(input_path=config_filename,
                                      output_path=modified_data_config_filename,
                                      reformat_config=True)

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
        collate_fn=stack_np_samples_into_batch,
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

    # Get gsp shape file, go get the osgb coorindates
    # This now gets data from the NG data portal.
    # We could change this so the x osgb and y osgb values are saved
    gsp_id_to_shape = get_gsp_shape_from_eso(return_filename=False)

    # Ensure the centroids have the same GSP ID index as the GSP PV power:
    gsp_id_to_shape = gsp_id_to_shape.loc[gsp_ids]
    x_osgb = gsp_id_to_shape.geometry.centroid.x.astype(np.float32)
    y_osgb = gsp_id_to_shape.geometry.centroid.y.astype(np.float32)
    locations = []
    for gsp_id in gsp_ids:
        location = Location(x=x_osgb.loc[gsp_id], y=y_osgb.loc[gsp_id], id=gsp_id)
        locations.append(location)

    # Location and time datapipes, the locations objects have x_osgb and y_osgb
    location_pipe = IterableWrapper(locations)
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
        .map(legacy_stack_np_examples_into_batch)
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
