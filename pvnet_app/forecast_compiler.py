import warnings
import logging
from datetime import timedelta
import torch

import numpy as np
import pandas as pd
import xarray as xr

from ocf_datapipes.batch import BatchKey
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD

import pvnet
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from pvnet_app.utils import preds_to_dataarray


logger = logging.getLogger(__name__)

# If the solar elevation is less than this the predictions are set to zero
MIN_DAY_ELEVATION = 0


_summation_mismatch_msg = (
    "The PVNet version running in this app is {}/{}. The summation model running in this app was "
    "trained on outputs from PVNet version {}/{}. Combining these models may lead to an error if "
    "the shape of PVNet output doesn't match the expected shape of the summation model. Combining "
    "may lead to unreliable results even if the shapes match."
)


class ForecastCompiler:
    """Class for making and compiling solar forecasts from for all GB GSPsn and national total"""
    def __init__(
        self, 
        model_name: str, 
        model_version: str, 
        summation_name: str | None, 
        summation_version: str | None, 
        device: torch.device, 
        t0: pd.Timestamp, 
        gsp_capacities: xr.DataArray, 
        national_capacity: float, 
        verbose: bool = False
    ):
        """Class for making and compiling solar forecasts from for all GB GSPsn and national total
        
        Args:
            model_name: Name of the huggingface repo where the PVNet model is stored
            model_version: Version of the PVNet model to run within the huggingface repo
            summation_name: Name of the huggingface repo where the summation model is stored
            summation_version: Version of the summation model to run within the huggingface repo
            device: Device to run the model on
            t0: The t0 time used to compile the results to numpy array
            gsp_capacities: DataArray of the solar capacities for all regional GSPs at t0
            national_capacity: The national solar capacity at t0
            verbose: Whether to log all messages throughout prediction and compilation
        """
        self.model_name = model_name
        self.model_version = model_version
        self.device = device
        self.t0 = t0
        self.gsp_capacities = gsp_capacities
        self.national_capacity = national_capacity
        self.verbose = verbose
        self.normed_preds = []
        self.gsp_ids_each_batch = []
        self.sun_down_masks = []
        
        
        logger.info(f"Loading model: {model_name} - {model_version}")
        
        self.model = PVNetBaseModel.from_pretrained(
            model_id=model_name,
            revision=model_version,
        ).to(device)

        if summation_name is None:
            self.summation_model = None
        else:
            self.summation_model = SummationBaseModel.from_pretrained(
                model_id=summation_name,
                revision=summation_version,
            ).to(device)

            if (
                (self.summation_model.pvnet_model_name, self.summation_model.pvnet_model_version) != 
                (model_name, model_version)
            ):
                warnings.warn(
                    _summation_mismatch_msg.format(
                        model_name, 
                        model_version, 
                        self.summation_model.pvnet_model_name, 
                        self.summation_model.pvnet_model_version,
                    )
                )
    
    
    def log_info(self, message):
        """Maybe log message depending on verbosity"""
        if self.verbose:
            logger.info(message)
    
    
    def predict_batch(self, batch):
        """Make predictions for a batch and store results internally"""
        
        self.log_info(f"Predicting for model: {self.model_name}-{self.model_version}")
        # Store GSP IDs for this batch for reordering later
        these_gsp_ids = batch[BatchKey.gsp_id].cpu().numpy()
        self.gsp_ids_each_batch += [these_gsp_ids]

        # Run batch through model
        preds = self.model(batch).detach().cpu().numpy()

        # Calculate unnormalised elevation and sun-dowm mask
        self.log_info("Zeroing predictions after sundown")
        elevation = (
            batch[BatchKey.gsp_solar_elevation].cpu().numpy() * ELEVATION_STD + ELEVATION_MEAN
        )
        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, -preds.shape[1] :]
        sun_down_mask = elevation < MIN_DAY_ELEVATION

        # Store predictions internally
        self.normed_preds += [preds]
        self.sun_down_masks += [sun_down_mask]

        # Log max prediction
        self.log_info(f"GSP IDs: {these_gsp_ids}")
        self.log_info(f"Max prediction: {np.max(preds, axis=1)}")
        
    
    def compile_forecasts(self):
        """Compile all forecasts internally
        
        Compiles all the regional GSP-level forecasts, makes national forecast, and compiles all
        into a Dataset
        """
        
        # Complie results from all batches
        normed_preds = np.concatenate(self.normed_preds)
        sun_down_masks = np.concatenate(self.sun_down_masks)
        gsp_ids_all_batches = np.concatenate(self.gsp_ids_each_batch).squeeze()

        n_times = normed_preds.shape[1]

        valid_times = pd.to_datetime([self.t0 + timedelta(minutes=30 * (i + 1)) for i in range(n_times)])

        # Reorder GSPs which can end up shuffled if multiprocessing is used
        inds = gsp_ids_all_batches.argsort()

        normed_preds = normed_preds[inds]
        sun_down_masks = sun_down_masks[inds]
        gsp_ids_all_batches = gsp_ids_all_batches[inds]
        
        # Merge batch results to xarray DataArray
        da_normed = preds_to_dataarray(normed_preds, self.model, valid_times, gsp_ids_all_batches)
        
        da_sundown_mask = xr.DataArray(
            data=sun_down_masks,
            dims=["gsp_id", "target_datetime_utc"],
            coords=dict(
                gsp_id=gsp_ids_all_batches,
                target_datetime_utc=valid_times,
            ),
        )

        # Multiply normalised forecasts by capacities and clip negatives
        self.log_info(f"Converting to absolute MW using {self.gsp_capacities}")
        da_abs = da_normed.clip(0, None) * self.gsp_capacities.values[:, None, None]
        max_preds = da_abs.sel(output_label="forecast_mw").max(dim="target_datetime_utc")
        self.log_info(f"Maximum predictions: {max_preds}")

        # Apply sundown mask
        da_abs = da_abs.where(~da_sundown_mask).fillna(0.0)
        
        if self.summation_model is None:
            self.log_info("Summing across GSPs to produce national forecast")
            da_abs_national = (
                da_abs.sum(dim="gsp_id").expand_dims(dim="gsp_id", axis=0).assign_coords(gsp_id=[0])
            )
        else:
            self.log_info("Using summation model to produce national forecast")

            # Make national predictions using summation model
            inputs = {
                "pvnet_outputs": torch.Tensor(normed_preds[np.newaxis]).to(self.device),
                "effective_capacity": (
                    torch.Tensor(self.gsp_capacities.values / self.national_capacity)
                    .to(self.device)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                ),
            }
            normed_national = self.summation_model(inputs).detach().squeeze().cpu().numpy()

            # Convert national predictions to DataArray
            da_normed_national = preds_to_dataarray(
                normed_national[np.newaxis], 
                self.summation_model, 
                valid_times, 
                gsp_ids=[0]
            )

            # Multiply normalised forecasts by capacities and clip negatives
            da_abs_national = da_normed_national.clip(0, None) * self.national_capacity

            # Apply sundown mask - All GSPs must be masked to mask national
            da_abs_national = da_abs_national.where(~da_sundown_mask.all(dim="gsp_id")).fillna(0.0)

        # Store the compiled predictions internally
        self.da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")

        self.log_info(
            f"National forecast is {self.da_abs_all.sel(gsp_id=0, output_label='forecast_mw').values}"
        )