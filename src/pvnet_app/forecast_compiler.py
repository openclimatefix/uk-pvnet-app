import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xarray as xr
import torch
from nowcasting_datamodel.models import ForecastSQL, ForecastValue
from nowcasting_datamodel.read.read import get_latest_input_data_last_updated, get_location
from nowcasting_datamodel.read.read_models import get_model
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from ocf_datapipes.batch import (
    BatchKey, NumpyBatch, NWPBatchKey, batch_to_tensor, copy_batch_to_device
)
from ocf_datapipes.utils.consts import ELEVATION_MEAN, ELEVATION_STD
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet_summation.models.base_model import BaseModel as SummationBaseModel
from sqlalchemy.orm import Session


from pvnet_app.model_configs.pydantic_models import ModelConfig
from pvnet_app.consts import __version__

logger = logging.getLogger(__name__)


# If the solar elevation (in degrees) is less than this the predictions are set to zero
MIN_DAY_ELEVATION = 0


_model_mismatch_msg = (
    "The PVNet version running in this app is {}/{}. The summation model running in this app was "
    "trained on outputs from PVNet version {}/{}. Combining these models may lead to an error if "
    "the shape of PVNet output doesn't match the expected shape of the summation model. Combining "
    "may lead to unreliable results even if the shapes match."
)


class ForecastCompiler:
    """Class for making and compiling solar forecasts from for all GB GSPs and national total"""

    def __init__(
        self,
        model_config: ModelConfig,
        device: torch.device,
        t0: pd.Timestamp,
        gsp_capacities: xr.DataArray,
        national_capacity: float,
        use_legacy: bool = False,
    ):
        """Class for making and compiling solar forecasts from for all GB GSPs and national total

        Args:
            model_config: The configuration for the model
            device: Device to run the model on
            t0: The t0 time used to compile the results to numpy array
            gsp_capacities: DataArray of the solar capacities for all regional GSPs at t0
            national_capacity: The national solar capacity at t0
            use_legacy: Whether to run legacy dataloader
        """
        model_name = model_config.pvnet.repo
        model_version = model_config.pvnet.version

        logger.info(f"Loading model: {model_name} - {model_version}")

        # Store settings
        self.model_tag = model_config.name
        self.model_name = model_name
        self.model_version = model_version
        self.device = device
        self.gsp_capacities = gsp_capacities
        self.national_capacity = national_capacity
        self.apply_adjuster = model_config.use_adjuster
        self.save_gsp_sum = model_config.save_gsp_sum
        self.save_gsp_to_recent = model_config.save_gsp_to_recent
        self.verbose = model_config.verbose
        self.use_legacy = use_legacy            

        # Create stores for the predictions
        self.normed_preds = []
        self.gsp_ids_each_batch = []
        self.sun_down_masks = []

        # Load the GSP and summation models
        self.model, self.summation_model = self.load_model(
            model_name,
            model_version,
            model_config.summation.repo,
            model_config.summation.version,
            device,
        )

        # These are the valid times this forecast will predict for
        self.valid_times = t0 + pd.timedelta_range(
            start="30min", freq="30min", periods=self.model.forecast_len,
        )


    @staticmethod
    def load_model(
        model_name: str,
        model_version: str,
        summation_name: str | None,
        summation_version: str | None,
        device: torch.device,
    ):
        """Load the GSP and summation models
        
        Args:
            model_name: The huggingface repo of the GSP model
            model_version: The commit hash of the GSP repo to load
            summation_name: The huggingface repo of the summation model
            summation_version: The commit hash of the summation model to load
            device: The device the models will be run on
        """
        
        # Load the GSP level model
        model = PVNetBaseModel.from_pretrained(
            model_id=model_name,
            revision=model_version,
        ).to(device)

        # Load the summation model
        if summation_name is None:
            sum_model = None
        else:
            sum_model = SummationBaseModel.from_pretrained(
                model_id=summation_name,
                revision=summation_version,
            ).to(device)

            # Compare the current GSP model with the one the summation model was trained on
            this_gsp_model = (model_name, model_version)
            sum_expected_gsp_model = (sum_model.pvnet_model_name, sum_model.pvnet_model_version)

            if sum_expected_gsp_model != this_gsp_model:
                logger.warning(_model_mismatch_msg.format(*this_gsp_model, *sum_expected_gsp_model))

        return model, sum_model


    def log_info(self, message: str) -> None:
        """Maybe log message depending on verbosity"""
        if self.verbose:
            logger.info(message)
    

    @torch.no_grad
    def predict_batch(self, batch: NumpyBatch) -> None:
        """Make predictions for a batch and store results internally"""
        self.log_info(f"Predicting for model: {self.model_name}-{self.model_version}")

        batch = copy_batch_to_device(batch_to_tensor(batch), self.device)

        if not self.use_legacy:
            change_keys_to_ocf_datapipes_keys(batch)

        # Store GSP IDs for this batch for reordering later
        these_gsp_ids = batch[BatchKey.gsp_id].cpu().numpy()

        self.gsp_ids_each_batch += [these_gsp_ids]

        self.log_info(f"{batch[BatchKey.gsp_id]=}")

        # TODO: This maintains compatibility between data-sampler and the old PVNet. This will go 
        # after upgrading to PVNet 4.0 and removing support for legacy models run with datapipes
        batch[BatchKey.gsp_id] = batch[BatchKey.gsp_id].unsqueeze(1)

        # Run batch through model
        preds = self.model(batch).detach().cpu().numpy()

        # Calculate unnormalised elevation and sun-dowm mask
        self.log_info("Computing sundown mask")
        if self.use_legacy:
            # The old dataloader standardises the data
            elevation = (
                batch[BatchKey.gsp_solar_elevation].cpu().numpy() *
                ELEVATION_STD + ELEVATION_MEAN
            )
        else:
            # The new dataloader normalises the data to [0, 1]
            elevation = (
                batch[BatchKey.gsp_solar_elevation].cpu().numpy() - 0.5) * 180

        # We only need elevation mask for forecasted values, not history
        elevation = elevation[:, -preds.shape[1]:]
        sun_down_mask = elevation < MIN_DAY_ELEVATION

        # Store predictions internally
        self.normed_preds += [preds]
        self.sun_down_masks += [sun_down_mask]

        # Log max prediction
        self.log_info(f"GSP IDs: {these_gsp_ids}")
        self.log_info(f"Max prediction: {np.max(preds, axis=1)}")


    @torch.no_grad
    def compile_forecasts(self) -> None:
        """Compile all forecasts internally in a single DataArray

        Steps:
        - Compile all the GSP level forecasts
        - Make national forecast
        - Compile all forecasts into a DataArray stored inside the object as `da_abs_all`
        """
        # Compile results from all batches
        normed_preds = np.concatenate(self.normed_preds)
        sun_down_masks = np.concatenate(self.sun_down_masks)
        # TODO: With the datapipes, the GSP IDs are in a 2D array. The squeeze below can be removed 
        # after removing support for legacy models run with datapipes
        gsp_ids = np.concatenate(self.gsp_ids_each_batch).squeeze()

        # Reorder GSPs which can end up shuffled if multiprocessing is used
        inds = gsp_ids.argsort()

        normed_preds = normed_preds[inds]
        sun_down_masks = sun_down_masks[inds]
        gsp_ids = gsp_ids[inds]

        # Merge batch results to xarray DataArray
        da_normed = self.preds_to_dataarray(normed_preds, self.model.output_quantiles, gsp_ids)

        da_sundown_mask = xr.DataArray(
            data=sun_down_masks,
            dims=["gsp_id", "target_datetime_utc"],
            coords=dict(
                gsp_id=gsp_ids,
                target_datetime_utc=self.valid_times,
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
                da_abs.sum(dim="gsp_id")
                .expand_dims(dim="gsp_id", axis=0)
                .assign_coords(gsp_id=[0])
            )
        else:
            self.log_info("Using summation model to produce national forecast")

            # Make national predictions using summation model
            inputs = {
                "pvnet_outputs": torch.Tensor(normed_preds[np.newaxis]).to(self.device),
                "effective_capacity": (
                    torch.Tensor(self.gsp_capacities.values/self.national_capacity)
                    .to(self.device)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                ),
            }
            normed_national = self.summation_model(inputs).detach().squeeze().cpu().numpy()

            # Convert national predictions to DataArray
            da_normed_national = self.preds_to_dataarray(
                normed_national[np.newaxis],
                self.summation_model.output_quantiles,
                gsp_ids=[0],
            )

            # Multiply normalised forecasts by capacity, clip negatives
            da_abs_national = da_normed_national.clip(0, None) * self.national_capacity

            # Apply sundown mask - All GSPs must be masked to mask national
            da_abs_national = da_abs_national.where(
                ~da_sundown_mask.all(dim="gsp_id")).fillna(0.0)

        self.log_info(
            f"National forecast is {da_abs_national.sel(output_label='forecast_mw').values}",
        )

        # Store the compiled predictions internally
        self.da_abs_all = xr.concat([da_abs_national, da_abs], dim="gsp_id")


    def preds_to_dataarray(
        self,
        preds: np.ndarray,
        output_quantiles: list[float] | None,
        gsp_ids: list[int],
    ) -> xr.DataArray:
        """Put numpy array of predictions into a dataarray"""
        if output_quantiles is not None:
            output_labels = [
                f"forecast_mw_plevel_{int(q*100):02}" for q in output_quantiles]
            output_labels[output_labels.index(
                "forecast_mw_plevel_50")] = "forecast_mw"
        else:
            output_labels = ["forecast_mw"]
            preds = preds[..., np.newaxis]

        da = xr.DataArray(
            data=preds,
            dims=["gsp_id", "target_datetime_utc", "output_label"],
            coords=dict(
                gsp_id=gsp_ids,
                target_datetime_utc=self.valid_times,
                output_label=output_labels,
            ),
        )
        return da


    def log_forecast_to_database(self, session: Session) -> None:
        """Log the compiled forecast to the database"""
        self.log_info("Converting DataArray to list of ForecastSQL")

        sql_forecasts = self.convert_dataarray_to_forecasts(
            self.da_abs_all,
            session,
            model_tag=self.model_tag,
            version=__version__,
        )

        self.log_info("Saving ForecastSQL to database")

        if self.save_gsp_to_recent:

            # Save all forecasts and save to last_seven_days table
            save_sql_forecasts(
                forecasts=sql_forecasts,
                session=session,
                update_national=True,
                update_gsp=True,
                apply_adjuster=self.apply_adjuster,
                save_to_last_seven_days=True,
            )
        else:
            # Save national and save to last_seven_days table
            save_sql_forecasts(
                forecasts=sql_forecasts[0:1],
                session=session,
                update_national=True,
                update_gsp=False,
                apply_adjuster=self.apply_adjuster,
                save_to_last_seven_days=True,
            )

            # Save GSP results but not to last_seven_dats table
            save_sql_forecasts(
                forecasts=sql_forecasts[1:],
                session=session,
                update_national=False,
                update_gsp=True,
                apply_adjuster=self.apply_adjuster,
                save_to_last_seven_days=False,
            )

        if self.save_gsp_sum:
            # Compute the sum if we are logging the sum of GSPs independently
            da_abs_sum_gsps = (
                self.da_abs_all.sel(gsp_id=slice(1, 317))
                .sum(dim="gsp_id")
                # Only select the central forecast for the GSP sum. The sums of different p-levels
                # are not a meaningful qauntities
                .sel(output_label=["forecast_mw"])
                .expand_dims(dim="gsp_id", axis=0)
                .assign_coords(gsp_id=[0])
            )

            # Save the sum of GSPs independently - mainly for summation model monitoring
            gsp_sum_sql_forecasts = self.convert_dataarray_to_forecasts(
                da_abs_sum_gsps,
                session,
                model_tag=f"{self.model_tag}_gsp_sum",
                version=__version__,
            )

            save_sql_forecasts(
                forecasts=gsp_sum_sql_forecasts,
                session=session,
                update_national=True,
                update_gsp=False,
                apply_adjuster=False,
                save_to_last_seven_days=True,
            )


    @staticmethod
    def convert_dataarray_to_forecasts(
        da_preds: xr.DataArray, 
        session: Session, 
        model_tag: str, 
        version: str,
    ) -> list[ForecastSQL]:
        """Make a ForecastSQL object from a DataArray.

        Args:
            da_preds: DataArray of forecasted values
            session: Database session
            model_key: the name of the model to saved to the database
            version: The version of the model
        Return:
            List of ForecastSQL objects
        """

        # Get time when the input data was last updated
        # TODO: This time will probably be wrong. It can take 15 mins to run the app, so the 
        # forecast will have downloaded older data than is reflected here
        input_data_last_updated = get_latest_input_data_last_updated(session=session)

        model = get_model(name=model_tag, version=version, session=session)

        forecasts = []

        for gsp_id in da_preds.gsp_id.values:

            da_gsp = da_preds.sel(gsp_id=gsp_id)

            forecast_values = []

            for target_time in pd.to_datetime(da_gsp.target_datetime_utc.values):

                da_gsp_time = da_gsp.sel(target_datetime_utc=target_time)

                forecast_value_sql = ForecastValue(
                    target_time=target_time.replace(tzinfo=UTC),
                    expected_power_generation_megawatts=(
                        da_gsp_time.sel(output_label="forecast_mw").item()
                    ),
                ).to_orm()

                properties = {}

                for p_level in ["10", "90"]:

                    if f"forecast_mw_plevel_{p_level}" in da_gsp_time.output_label:
                        p_val = da_gsp_time.sel(output_label=f"forecast_mw_plevel_{p_level}").item()
                        # `p[10, 90]` can be NaN if PVNet has probabilistic outputs and 
                        # PVNet_summation doesn't, or vice versa. Do not log the value if NaN
                        if not np.isnan(p_val):
                            properties[p_level] = p_val

                if len(properties) > 0:
                    forecast_value_sql.properties = properties

                forecast_values.append(forecast_value_sql)

            location = get_location(session=session, gsp_id=int(gsp_id))

            forecast = ForecastSQL(
                model=model,
                # TODO: Should this time reflect when the forecast is saved, or the forecast 
                # init-time?
                forecast_creation_time=datetime.now(tz=UTC),
                location=location,
                input_data_last_updated=input_data_last_updated,
                forecast_values=forecast_values,
                historic=False,
            )

            forecasts.append(forecast)

        return forecasts


def change_keys_to_ocf_datapipes_keys(batch):
    """Change string keys from ocf-data-sampler to BatchKey from ocf-datapipes

    The key change is done in-place

    Until PVNet is merged from dev-data-sampler, we need to do this.
    After this, we might need to change the other way around, for the legacy models.
    """
    keys_to_rename = [
        BatchKey.satellite_actual,
        BatchKey.nwp,
        BatchKey.gsp_solar_elevation,
        BatchKey.gsp_solar_azimuth,
        BatchKey.gsp_id
    ]

    for key in keys_to_rename:
        if key.name in batch:
            batch[key] = batch[key.name]
            del batch[key.name]

    if BatchKey.nwp in batch.keys():
        nwp_batch = batch[BatchKey.nwp]
        for nwp_source in nwp_batch.keys():
            nwp_batch[nwp_source][NWPBatchKey.nwp] = nwp_batch[nwp_source]["nwp"]
            del nwp_batch[nwp_source]["nwp"]
