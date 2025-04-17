import tempfile

from ocf_data_sampler.config import load_yaml_configuration
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.config import modify_data_config_for_production
from pvnet_app.model_configs.pydantic_models import get_all_models


def test_data_config():

    models = get_all_models(get_critical_only=False)

    with tempfile.TemporaryDirectory() as tmpdirname:

        for model in models:
            # get config from huggingface

            data_config_path = PVNetBaseModel.get_data_config(
                model.pvnet.repo,
                revision=model.pvnet.commit,
            )

            # make a temporary file ending in yaml
            temp_data_config_path = f"{tmpdirname}/data_config.yaml"

            modify_data_config_for_production(
                input_path=data_config_path,
                output_path=temp_data_config_path,
            )

            _ = load_yaml_configuration(temp_data_config_path)
