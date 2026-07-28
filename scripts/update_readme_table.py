"""This script updates the model summary table in the package README to the current models.

It adds all models in src/pvnet_app/models/catalogue.py
"""

import os
from pathlib import Path

from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.model_input_config import get_required_nwp_providers, load_yaml_config
from pvnet_app.models.registry import HuggingFaceCommit, get_model_specs


def make_huggingface_link(model_commit: HuggingFaceCommit) -> str:
    """Make a link to the model on huggingface.

    Args:
        model_commit: The model commit
    """
    return f"https://huggingface.co/{model_commit.repo}/tree/{model_commit.commit}"


def generate_table() -> str:
    """Make a new summary table for the models described in the model configs."""
    model_specs = get_model_specs()
    columns = [
        "Model Name",
        "Uses satellite",
        "Uses UKV",
        "Uses ECMWF",
        "Uses cloudcasting",
        "PVNet Hugging Face Link",
        "PVNet Summation Hugging Face Link",
    ]
    header = " | ".join(columns)
    separator = "|".join(["----" for _ in columns])
    rows = [header, separator]

    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)


    for model_spec in model_specs:
        pvnet_link = make_huggingface_link(model_spec.pvnet)
        summation_link = make_huggingface_link(model_spec.summation)

        data_config_path = PVNetBaseModel.get_data_config(
            model_spec.pvnet.repo,
            revision=model_spec.pvnet.commit,
            token=hf_token,
        )
        data_config = load_yaml_config(data_config_path)

        providers = get_required_nwp_providers([data_config])

        def yes_or_blank(value: bool) -> str:
            return "yes" if value else "-"

        row = " | ".join(
            [
                model_spec.name,
                yes_or_blank("satellite" in data_config["input_data"]),
                yes_or_blank("ukv" in providers),
                yes_or_blank("ecmwf" in providers),
                yes_or_blank("cloudcasting" in providers),
                f"[HF Link]({pvnet_link})",
                f"[Summation HF Link]({summation_link})",
            ],
        )
        rows.append(row)

    table = "".join([f"| {row} |\n" for row in rows])

    return table


def update_readme() -> None:
    """Update the model summary table in the package README with the models in the model config."""
    readme = Path("README.md").read_text()
    start, end = "<!-- START model-config-table -->", "<!-- END model-config-table -->"
    new_table = generate_table()
    updated = readme.split(start)[0] + start + "\n" + new_table + "\n" + end + readme.split(end)[1]
    Path("README.md").write_text(updated)


if __name__ == "__main__":
    update_readme()
