"""Class and helpers to run PVNet model forecasts."""

import logging

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, TensorBatch
from ocf_data_sampler.torch_datasets.pvnet_dataset import PVNetConcurrentDataset
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import (
    batch_to_tensor,
    copy_batch_to_device,
)
from pvlib.solarposition import get_solarposition
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.data.datamodule import construct_sample as construct_sum_sample
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel

from pvnet_app.data.gsp import get_gsp_locations
from pvnet_app.model_input_config import modify_data_config_for_production
from pvnet_app.models.registry import ModelSpec

# If the solar elevation (in degrees) is less than this the predictions are set to zero
MIN_DAY_ELEVATION_DEGREES = 0


_model_mismatch_msg = (
    "The PVNet commit running in this app is {}/{}. The summation model running in this app was "
    "trained on outputs from PVNet commit {}/{}. Combining these models may lead to an error if "
    "the shape of PVNet output doesn't match the expected shape of the summation model. Combining "
    "may lead to unreliable results even if the shapes match."
)


def preds_to_dataarray(
    preds: np.ndarray,
    location_ids: list[int],
    output_quantiles: list[float] | None,
    valid_times_utc: pd.DatetimeIndex,
    horizon_mins: np.ndarray,
) -> xr.DataArray:
    """Put numpy array of predictions into a dataarray."""
    if output_quantiles is not None:
        output_labels = [f"p{int(q * 100):02}" for q in output_quantiles]
    else:
        output_labels = ["p50"]
        preds = preds[..., np.newaxis]

    return xr.DataArray(
        data=preds,
        dims=["location_id", "valid_times_utc", "output_label"],
        coords={
            "location_id": location_ids,
            "output_label": output_labels,
            "valid_times_utc": valid_times_utc,
            "horizon_mins": ("valid_times_utc", horizon_mins),
        },
    )


class PVNetForecaster:
    """Class for making and compiling solar forecasts from for all GB GSPs and national total."""

    def __init__(
        self,
        model_spec: ModelSpec,
        data_config_path: str,
        run_data_dir: str,
        t0: pd.Timestamp,
        device: torch.device,
        capacities: dict[int, float],
        hf_token: bool | str | None = None,
    ) -> None:
        """Class for making and compiling solar forecasts from for all GB GSPs and national total.

        Args:
            model_spec: The configuration for the model
            data_config_path: The path to the model data config
            run_data_dir: The directory where the downloaded input data is stored
            t0: The forecast init-time
            device: Device to run the model on
            capacities: Dictionary of the solar capacities for all locations at t0
            hf_token: HF authentication token. If True, the token is read from the HF config folder.
                If string, it is used as the authentication token.
        """
        self.logger = logging.getLogger(model_spec.name)
        self.logger.setLevel(getattr(logging, model_spec.log_level))
        self.logger.info(f"Loading model: {model_spec.pvnet.repo}")

        # Store settings
        self.model_tag = model_spec.name
        self.data_config_path = data_config_path
        self.run_data_dir = run_data_dir
        self.t0 = t0
        self.device = device
        self.capacities = capacities

        # Load the regional and summation models
        self.model, self.summation_model = self.load_model(
            model_spec.pvnet.repo,
            model_spec.pvnet.commit,
            model_spec.summation.repo,
            model_spec.summation.commit,
            device,
            hf_token,
        )

        # Load the coordinates of all locations
        self.location_coords = get_gsp_locations()

        # These are the valid times this forecast will predict for
        self.horizon_mins = np.arange(1, self.model.forecast_len + 1) * 30
        self.valid_times = self.t0 + pd.to_timedelta(self.horizon_mins, unit="m")

    def load_model(
        self,
        pvnet_repo: str,
        pvnet_commit: str,
        summation_repo: str | None,
        summation_commit: str | None,
        device: torch.device,
        hf_token: bool | str | None = None,
    ) -> tuple[PVNetBaseModel, SummationBaseModel | None]:
        """Load the GSP and summation models.

        Args:
            pvnet_repo: The huggingface repo of the GSP model
            pvnet_commit: The commit hash of the GSP repo to load
            summation_repo: The huggingface repo of the summation model
            summation_commit: The commit hash of the summation model to load
            device: The device the models will be run on
            hf_token: HF authentication token.
        """
        # Load the GSP level model
        model = PVNetBaseModel.from_pretrained(
            model_id=pvnet_repo,
            revision=pvnet_commit,
            token=hf_token,
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
        runtime_data_config_filepath = f"{self.run_data_dir}/{self.model_tag}_data_config.yaml"

        modify_data_config_for_production(
            input_path=self.data_config_path,
            output_path=runtime_data_config_filepath,
            run_data_dir=self.run_data_dir,
        )

        dataset = PVNetConcurrentDataset(config_filename=runtime_data_config_filepath)

        return dataset.get_sample(self.t0)

    @torch.inference_mode()
    def predict(self, batch: NumpyBatch) -> xr.DataArray:
        """Make predictions for the batch."""
        self.logger.debug(f"Predicting for model: {self.model_tag}")

        location_ids = batch["location_id"]
        self.logger.debug(f"GSPs: {location_ids}")

        relative_capacities = self.get_relative_capacities(location_ids.tolist())

        tensor_batch = copy_batch_to_device(batch_to_tensor(batch), self.device)

        if self.summation_model is None:
            self.logger.debug("Summing across GSPs to produce national forecast")
            da_preds = self.predict_with_regional_sum(tensor_batch, relative_capacities)

        else:
            self.logger.debug("Using summation model to produce national forecast")
            da_preds = self.predict_with_summation_model(tensor_batch, relative_capacities)

        return da_preds

    def predict_with_summation_model(
        self,
        batch: TensorBatch,
        relative_capacities: np.ndarray,
    ) -> xr.DataArray:
        """Make predictions for the batch using regional and summation models."""
        # Make regional predictions using the GSP model
        regional_preds = self.model(batch).detach().cpu().numpy()

        # Make national predictions using summation model
        summation_batch = construct_sum_sample(
            pvnet_inputs=None,
            valid_times=self.valid_times,
            relative_capacities=relative_capacities,
            # The summation model expects the central coords of the entire UK area
            longitude=self.location_coords.loc[0].longitude.item(),
            latitude=self.location_coords.loc[0].latitude.item(),
            target=None,
        )
        summation_batch["pvnet_outputs"] = regional_preds
        del summation_batch["pvnet_inputs"]

        # Expand for batch dimension and convert to tensors
        summation_batch = {
            k: torch.from_numpy(v[None, ...]).to(self.device) for k, v in summation_batch.items()
        }

        normed_national = (
            self.summation_model(summation_batch)
            .detach()
            .cpu()
            .numpy()
            .squeeze(axis=0)  # Remove the batch dimension
        )

        location_ids = batch["location_id"].cpu().numpy().tolist()

        # Convert regional and national predictions to DataArrays separately since they may have
        # different output quantiles. We will concatenate them later after all the processing is
        # done. Else we might accidentally infill missing quantiles with zeros when concatenating.
        da_regional_preds = preds_to_dataarray(
            preds=regional_preds,
            output_quantiles=self.model.output_quantiles,
            location_ids=location_ids,
            valid_times_utc=self.valid_times,
            horizon_mins=self.horizon_mins,
        )

        da_national_preds = preds_to_dataarray(
            preds=normed_national[np.newaxis],
            output_quantiles=self.summation_model.output_quantiles,
            location_ids=[0],
            valid_times_utc=self.valid_times,
            horizon_mins=self.horizon_mins,
        )

        # Make sundown masks so we can set predictions to zero when the sun is down
        da_regional_sundown_mask, da_national_sundown_mask = self._make_sundown_masks(location_ids)

        da_regional_preds = (
            # Set regional predictions to zero when sun is down
            da_regional_preds.where(~da_regional_sundown_mask, other=0)
            # Also clip negative predictions to zero
            .clip(0, None)
        )

        da_national_preds = (
            # Set national predictions to zero when sun is down
            da_national_preds.where(~da_national_sundown_mask, other=0)
            # Also clip negative predictions to zero
            .clip(0, None)
        )

        return xr.concat([da_national_preds, da_regional_preds], dim="location_id")

    def predict_with_regional_sum(
        self,
        batch: TensorBatch,
        relative_capacities: np.ndarray,
    ) -> xr.DataArray:
        """Make predictions for the batch using regional model and regional sum."""
        # Make regional predictions using the GSP model
        regional_preds = self.model(batch).detach().cpu().numpy()

        location_ids = batch["location_id"].cpu().numpy().tolist()

        da_regional_preds = preds_to_dataarray(
            preds=regional_preds,
            output_quantiles=self.model.output_quantiles,
            location_ids=location_ids,
            valid_times_utc=self.valid_times,
            horizon_mins=self.horizon_mins,
        )

        # Make sundown masks so we can set predictions to zero when the sun is down
        da_regional_sundown_mask, _ = self._make_sundown_masks(location_ids)

        # Postprocess the regional predictions
        da_regional_preds = (
            # Set regional predictions to zero when sun is down
            da_regional_preds.where(~da_regional_sundown_mask, other=0)
            # Also clip negative predictions to zero
            .clip(0, None)
        )

        # Make national predictions by summing the regional predictions weighted by the capacities
        # The national predictions inherit the sundown masking and clipping from the regional
        # predictions since they are derived from them
        da_national_preds = (
            (da_regional_preds * relative_capacities[:, None, None])
            .sum(dim="location_id")
            .expand_dims(dim="location_id", axis=0)
            .assign_coords(location_id=[0])
        )

        return xr.concat([da_national_preds, da_regional_preds], dim="location_id")

    def _make_sundown_masks(self, location_ids: list[int]) -> tuple[xr.DataArray, xr.DataArray]:
        """Make a sundown mask for all locations and valid times."""
        elevations = []
        for loc_id in location_ids:
            elevation = get_solarposition(
                time=self.valid_times,
                longitude=self.location_coords.loc[loc_id].longitude.item(),
                latitude=self.location_coords.loc[loc_id].latitude.item(),
                method="nrel_numpy",
            )["elevation"].values

            elevations.append(elevation)

        regional_sundown_mask = np.array(elevations) < MIN_DAY_ELEVATION_DEGREES

        da_regional_sundown_mask = xr.DataArray(
            data=regional_sundown_mask,
            dims=["location_id", "valid_times_utc"],
            coords={
                "location_id": location_ids,
                "valid_times_utc": self.valid_times,
            },
        )

        # All GSPs must be masked to mask national
        da_national_sundown_mask = da_regional_sundown_mask.all(dim="location_id")

        return da_regional_sundown_mask, da_national_sundown_mask

    def get_relative_capacities(self, location_ids: list[int]) -> np.ndarray:
        """Get the relative capacities for the given location IDs."""
        return (
            np.array(
                [self.capacities[location_id] for location_id in location_ids],
                dtype=np.float32,
            )
            / self.capacities[0]
        )
