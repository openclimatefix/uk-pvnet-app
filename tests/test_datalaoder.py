from datetime import timedelta
import tempfile, os

import pandas as pd
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.dataloader import get_dataloader, get_legacy_dataloader
from pvnet_app.model_configs.pydantic_models import get_all_models


# def test_dataloader(nwp_ukv_data, nwp_ecmwf_data,):
#
#     models = get_all_models(run_extra_models=True, use_ocf_data_sampler=True)
#     model_config = models[0]
#
#     # get config from huggingface
#     data_config_path = PVNetBaseModel.get_data_config(
#         model_config.pvnet.repo,
#         revision=model_config.pvnet.version,
#     )
#
#     t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor(timedelta(minutes=30))
#
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         os.chdir(tmpdirname)
#
#         temp_nwp_path = "temp_nwp_ukv.zarr"
#         os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path
#         nwp_ukv_data.to_zarr(temp_nwp_path)
#
#         temp_nwp_path = "temp_nwp_ecmwf.zarr"
#         os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
#         nwp_ecmwf_data.to_zarr(temp_nwp_path)
#
#         _ = get_dataloader(
#             config_filename=data_config_path,
#             t0=t0,
#             gsp_ids=list(range(0, 10)),
#             batch_size=2,
#             num_workers=1,
#         )


def test_dataloader_legacy(db_session):

    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=False)
    model_config = models[0]

    # get config from huggingface
    data_config_path = PVNetBaseModel.get_data_config(
        model_config.pvnet.repo,
        revision=model_config.pvnet.version,
    )

    t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor(timedelta(minutes=30))

    _ = get_legacy_dataloader(
        config_filename=data_config_path,
        t0=t0,
        gsp_ids=list(range(1, 10)),
        batch_size=2,
        num_workers=1,
    )
