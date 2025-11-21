"""Functions to run the forecaster."""
import logging
import tempfile

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from dateutil.tz import UTC
from dp_sdk.ocf import dp
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch
from ocf_data_sampler.torch_datasets.pvnet_dataset import PVNetConcurrentDataset
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import (
    batch_to_tensor,
    copy_batch_to_device,
)
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.data.datamodule import construct_sample as construct_sum_sample
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from sqlalchemy.orm import Session

from pvnet_app.config import modify_data_config_for_production
from pvnet_app.model_configs.pydantic_models import ModelConfig
from pvnet_app.save import save_forecast, save_forecast_to_data_platform

# If the solar elevation (in degrees) is less than this the predictions are set to zero
MIN_DAY_ELEVATION = 0


_model_mismatch_msg = (
    "The PVNet commit running in this app is {}/{}. The summation model running in this app was "
    "trained on outputs from PVNet commit {}/{}. Combining these models may lead to an error if "
    "the shape of PVNet output doesn't match the expected shape of the summation model. Combining "
    "may lead to unreliable results even if the shapes match."
)


class Forecaster:
    """Class for making and compiling solar forecasts from for all GB GSPs and national total."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config_path: str,
        t0: pd.Timestamp,
        gsp_ids: list[int],
        device: torch.device,
        gsp_capacities: xr.DataArray,
        national_capacity: float,
    ) -> None:
        """Class for making and compiling solar forecasts from for all GB GSPs and national total.

        Args:
            model_config: The configuration for the model
            data_config_path: The path to the model data config
            t0: The forecast init-time
            gsp_ids: List of gsp_ids to make predictions for
            device: Device to run the model on
            gsp_capacities: DataArray of the solar capacities for all regional GSPs at t0
            national_capacity: The national solar capacity at t0
        """
        self.logger = logging.getLogger(model_config.name)
        self.logger.setLevel(getattr(logging, model_config.log_level))
        self.logger.info(f"Loading model: {model_config.pvnet.repo}")

        # Store settings
        self.model_tag = model_config.name
        self.data_config_path = data_config_path
        self.t0 = t0
        self.gsp_ids = gsp_ids
        self.device = device
        self.gsp_capacities = gsp_capacities
        self.national_capacity = national_capacity
        self.apply_adjuster = model_config.use_adjuster
        self.save_gsp_sum = model_config.save_gsp_sum
        self.save_gsp_to_recent = model_config.save_gsp_to_recent

        # Load the GSP and summation models
        self.model, self.summation_model = self.load_model(
            model_config.pvnet.repo,
            model_config.pvnet.commit,
            model_config.summation.repo,
            model_config.summation.commit,
            device,
        )

        # Values
        self.da_abs_all: xr.DataArray
        self.da_normed_all: xr.DataArray

        # These are the valid times this forecast will predict for
        self.valid_times = t0 + pd.timedelta_range(
            start="30min",
            freq="30min",
            periods=self.model.forecast_len,
        )

    def load_model(
        self,
        pvnet_repo: str,
        pvnet_commit: str,
        summation_repo: str | None,
        summation_commit: str | None,
        device: torch.device,
    ) -> tuple[PVNetBaseModel, SummationBaseModel | None]:
        """Load the GSP and summation models.

        Args:
            pvnet_repo: The huggingface repo of the GSP model
            pvnet_commit: The commit hash of the GSP repo to load
            summation_repo: The huggingface repo of the summation model
            summation_commit: The commit hash of the summation model to load
            device: The device the models will be run on
        """
        # Load the GSP level model
        model = PVNetBaseModel.from_pretrained(
            model_id=pvnet_repo,
            revision=pvnet_commit,
        ).to(device)

        # Load the summation model
        if summation_repo is None:
            sum_model = None
        else:
            sum_model = SummationBaseModel.from_pretrained(
                model_id=summation_repo,
                revision=summation_commit,
            ).to(device)

            # Compare the current GSP model with the one the summation model was trained on
            datamodule_path = SummationBaseModel.get_datamodule_config(
                model_id=summation_repo,
                revision=summation_commit,
            )
            with open(datamodule_path) as cfg:
                sum_pvnet_cfg = yaml.safe_load(cfg)["pvnet_model"]

            sum_expected_gsp_model = (sum_pvnet_cfg["model_id"], sum_pvnet_cfg["revision"])
            this_gsp_model = (pvnet_repo, pvnet_commit)

            if sum_expected_gsp_model != this_gsp_model:
                self.logger.warning(
                    _model_mismatch_msg.format(*this_gsp_model, *sum_expected_gsp_model),
                )

        return model, sum_model

    def make_batch(self) -> NumpyBatch:
        """Create the batch required to run this model."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as tmp:
            temp_path = tmp.name

            modify_data_config_for_production(
                input_path=self.data_config_path,
                output_path=temp_path,
            )

            dataset = PVNetConcurrentDataset(config_filename=temp_path)

        return dataset.get_sample(self.t0)

    @torch.inference_mode()
    def predict(self, batch: NumpyBatch) -> None:
        """Make predictions for the batch and store results internally."""
        self.logger.debug(f"Predicting for model: {self.model_tag}")

        gsp_ids = batch["location_id"]
        self.logger.debug(f"GSPs: {gsp_ids}")

        batch = copy_batch_to_device(batch_to_tensor(batch), self.device)

        # Run batch through model
        normed_preds = self.model(batch).detach().cpu().numpy()

        # Convert GSP results to xarray DataArray
        da_normed = self.preds_to_dataarray(
            normed_preds,
            self.model.output_quantiles,
            gsp_ids,
        )

        self.logger.debug("Clipping negatives, applying sundown mask")
        da_normed = da_normed.clip(0, None)

        # Calculate and apply sundown mask from solar elevation
        # - In the batch the solar elevation angle is scaled to the range [0, 1]
        elevation = (batch["solar_elevation"] - 0.5) * 180
        # - We only need elevation mask for forecasted values, not history
        elevation = elevation[:, -normed_preds.shape[1]:]
        da_sundown_mask = xr.DataArray(
            data=elevation < MIN_DAY_ELEVATION,
            dims=["gsp_id", "target_datetime_utc"],
            coords={
                "gsp_id": gsp_ids,
                "target_datetime_utc": self.valid_times,
            },
        )
        da_normed = da_normed.where(~da_sundown_mask).fillna(0.0)

        self.logger.debug("Converting to absolute MW")
        da_abs = da_normed * self.gsp_capacities.values[:, None, None]

        max_preds = da_abs.sel(output_label="forecast_mw").max(dim="target_datetime_utc")
        self.logger.debug(f"Maximum predictions: {max_preds}")

        if self.summation_model is None:
            self.logger.debug("Summing across GSPs to produce national forecast")
            da_abs_national = (
                da_abs.sum(dim="gsp_id").expand_dims(dim="gsp_id", axis=0).assign_coords(gsp_id=[0])
            )
        else:
            self.logger.debug("Using summation model to produce national forecast")

            # Make national predictions using summation model
            inputs = construct_sum_sample(
                pvnet_inputs=None,
                valid_times=self.valid_times,
                relative_capacities=self.gsp_capacities.values / self.national_capacity,
                target=None,
            )
            inputs["pvnet_outputs"] = normed_preds
            del inputs["pvnet_inputs"]

            # Expand for batch dimension and convert to tensors
            inputs = {k: torch.from_numpy(v[None, ...]).to(self.device) for k, v in inputs.items()}

            normed_national = self.summation_model(inputs).detach().squeeze().cpu().numpy()

            # Convert national predictions to DataArray
            da_normed_national = self.preds_to_dataarray(
                normed_national[np.newaxis],
                self.summation_model.output_quantiles,
                gsp_ids=[0],
            )

            # Clip negatives, apply sundown mask
            # All GSPs must be masked to mask national
            da_normed_national = (
                da_normed_national
                .clip(0, None)
                .where(~da_sundown_mask.all(dim="gsp_id"))
                .fillna(0.0)
            )

            # Convert to absolute MW
            da_abs_national = da_normed_national * self.national_capacity

        self.logger.debug(
            f"National forecast is {da_abs_national.sel(output_label='forecast_mw').values}",
        )

        # Rename the labels in the normalized dataset
        ds_normed_all = xr.concat(
            [da_normed_national, da_normed],
            dim="gsp_id",
        ).to_dataset(dim="output_label")
        for var in ds_normed_all.data_vars:
            ds_normed_all = ds_normed_all.rename({var: var.replace("_mw", "_fraction")})

        # Store the compiled predictions internally
        self.da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")
        self.da_normed_all = ds_normed_all.to_array(dim="output_label")

    def preds_to_dataarray(
        self,
        preds: np.ndarray,
        output_quantiles: list[float] | None,
        gsp_ids: list[int],
    ) -> xr.DataArray:
        """Put numpy array of predictions into a dataarray."""
        if output_quantiles is not None:
            output_labels = [f"forecast_mw_plevel_{int(q * 100):02}" for q in output_quantiles]
            output_labels[output_labels.index("forecast_mw_plevel_50")] = "forecast_mw"
        else:
            output_labels = ["forecast_mw"]
            preds = preds[..., np.newaxis]

        da = xr.DataArray(
            data=preds,
            dims=["gsp_id", "target_datetime_utc", "output_label"],
            coords={
                "gsp_id": gsp_ids,
                "target_datetime_utc": self.valid_times,
                "output_label": output_labels,
            },
        )
        return da

    def log_forecast_to_database(self, session: Session) -> None:
        """Log the compiled forecast to the database."""
        self.logger.debug("Saving ForecastSQL to database")

        # save using nowcasting_datamodel
        save_forecast(
            session=session,
            forecast_da=self.da_abs_all,
            model_tag=self.model_tag,
            save_gsp_to_recent=self.save_gsp_to_recent,
            apply_adjuster=self.apply_adjuster,
            save_gsp_sum=self.save_gsp_sum,
        )

    async def save_forecast_to_dataplatform(
        self,
        client: dp.DataPlatformDataServiceStub,
        locations_gsp_uuid_map: dict[int, str],
    ) -> None:
        """Save the compiled forecast to the data platform."""
        self.logger.debug("Saving forecast to data platform")
        await save_forecast_to_data_platform(
            forecast_normed_da=self.da_normed_all,
            locations_gsp_uuid_map=locations_gsp_uuid_map,
            model_tag=self.model_tag,
            init_time_utc=self.t0.to_pydatetime().replace(tzinfo=UTC),
            client=client,
    )
