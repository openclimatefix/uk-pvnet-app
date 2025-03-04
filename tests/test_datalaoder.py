import tempfile

from ocf_data_sampler.config import load_yaml_configuration
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.config import modify_data_config_for_production
from pvnet_app.dataloader import get_datapipes_dataloader
from pvnet_app.model_configs.pydantic_models import get_all_models


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
            temp_data_config_path = f"{tmpdirname}/data_config.yaml"

            modify_data_config_for_production(
                input_path=data_config_path,
                output_path=temp_data_config_path,
                reformat_config=True,
            )

            _ = load_yaml_configuration(temp_data_config_path)


def test_get_datapipes_dataloader(db_url, test_t0):

    models = get_all_models(run_extra_models=True, use_ocf_data_sampler=False)
    for model_config in models:

        # get config from huggingface
        data_config_path = PVNetBaseModel.get_data_config(
            model_config.pvnet.repo,
            revision=model_config.pvnet.version,
        )

        _ = get_datapipes_dataloader(
            config_filename=data_config_path,
            t0=test_t0,
            gsp_ids=list(range(1, 10)),
            batch_size=2,
            num_workers=1,
            db_url=db_url,
        )
