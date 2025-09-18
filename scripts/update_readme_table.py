"""This script updates the model summary table in the package README to the current models.

It adds all models in src/pvnet_app/model_configs/all_models.py
"""

from pathlib import Path

from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.config import load_yaml_config
from pvnet_app.model_configs.pydantic_models import HuggingFaceCommit, get_all_models


def make_huffingface_link(model_commit: HuggingFaceCommit) -> str:
    """Make a link to the model on huggingface.

    Args:
        repo_commit: The model commit
    """
    return f"https://huggingface.co/{model_commit.repo}/tree/{model_commit.commit}"


def generate_table() -> str:
    """Make a new summary table for the models descriobed in the model configs."""
    model_configs = get_all_models()
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

    for model_config in model_configs:
        pvnet_link = make_huffingface_link(model_config.pvnet)
        summation_link = make_huffingface_link(model_config.summation)

        data_config_path = PVNetBaseModel.get_data_config(
            model_config.pvnet.repo,
            revision=model_config.pvnet.commit,
        )
        data_config = load_yaml_config(data_config_path)

        providers = set()
        if "nwp" in data_config["input_data"]:
            for source in data_config["input_data"]["nwp"].values():
                providers.add(source["provider"])

        uses_ecmwf = "ecmwf" in providers
        uses_ukv = "ukv" in providers
        uses_cloud = "cloudcasting" in providers
        uses_sat = "satellite" in data_config["input_data"]

        row = " | ".join(
            [
                model_config.name,
                "yes" if uses_sat else "-",
                "yes" if uses_ukv else "-",
                "yes" if uses_ecmwf else "-",
                "yes" if uses_cloud else "-",
                f"[HF Link]({pvnet_link})",
                f"[Summation HF Link]({summation_link})",
            ],
        )
        rows.append(row)

    table = ""
    for row in rows:
        table += f"| {row} |\n"

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
