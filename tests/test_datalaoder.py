from datetime import timedelta
import tempfile, os

import pandas as pd
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.dataloader import get_legacy_dataloader
from pvnet_app.model_configs.pydantic_models import get_all_models
from ocf_data_sampler.config import load_yaml_configuration
from pvnet_app.config import modify_data_config_for_production


def test_data_config():

    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=True)

    with tempfile.TemporaryDirectory() as tmpdirname:

        for model in models:
            # get config from huggingface
            data_config_path = PVNetBaseModel.get_data_config(
                model.pvnet.repo,
                revision=model.pvnet.version,
            )

            # make a temporary file ending in yaml
            temp_data_config_path = os.path.join(tmpdirname, "data_config.yaml")

            modify_data_config_for_production(
                input_path=data_config_path,
                output_path=temp_data_config_path,
                drop_input_data_forecast_and_history=True,
            )

            _ = load_yaml_configuration(temp_data_config_path)


def test_dataloader_legacy(db_session):

    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=False)
    for model_config in models:

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
