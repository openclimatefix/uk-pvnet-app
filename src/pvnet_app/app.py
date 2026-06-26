"""Application to run forecasts with a collection of PVNet models."""

import asyncio
import contextlib
import logging
import os
import tempfile
from datetime import datetime
from importlib.metadata import version
from typing import TYPE_CHECKING

import pandas as pd
import sentry_sdk
import torch
from grpclib.client import Channel
from ocf import dp
from pvnet.utils import validate_batch_against_config

from pvnet_app.consts import (
    generation_path,
    nwp_cloudcasting_path,
    nwp_ecmwf_path,
    nwp_ukv_path,
    sat_path,
)
from pvnet_app.data.batch_validation import check_batch
from pvnet_app.data.gsp import create_null_generation_data
from pvnet_app.data.nwp import CloudcastingDownloader, ECMWFDownloader, UKVDownloader
from pvnet_app.data.satellite import SatelliteDownloader
from pvnet_app.forecaster import PVNetForecaster
from pvnet_app.model_input_config import (
    fetch_model_data_config_paths,
    get_required_nwp_providers,
    load_yaml_config,
)
from pvnet_app.models.registry import get_model_specs
from pvnet_app.data_platform import (
    extract_location_capacities_mwp,
    fetch_locations,
    write_forecasts_to_data_platform,
)
from pvnet_app.settings import AppSettings
from pvnet_app.utils import check_model_runs_finished, resolve_t0, save_batch_to_s3
from pvnet_app.validate_forecast import validate_forecast

if TYPE_CHECKING:
    import xarray as xr

__version__ = version("pvnet-app")

# ---------------------------------------------------------------------------
# LOGGING AND SENTRY

# Create a logger
logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Turn off logs from aiobotocore
logging.getLogger("aiobotocore").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# APP MAIN

async def run_app(
    settings: AppSettings,
    t0: str | datetime | pd.Timestamp | None = None,
    write_predictions: bool = True,
) -> None | dict:
    """Set up the app environment and run a forecast.

    Handles Sentry, init-time resolution, and the scratch directory, then
    delegates to the forecast pipeline.

    Args:
        settings: The application settings
        t0: The forecast init-time. If None, the current time is used. Input will be floored to the
            previous 30-minutes.
        write_predictions: If True, write forecasts to the data platform. If
            False, skip the write and return the forecasters for local testing
    """
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        traces_sample_rate=1,
    )
    sentry_sdk.set_tag("app_name", "pvnet_app")
    sentry_sdk.set_tag("version", __version__)

    t0 = resolve_t0(t0)

    logger.info(f"Using `pvnet` library version: {version('pvnet')}")
    logger.info(f"Using `pvnet_app` library version: {__version__}")
    logger.info(f"Making forecast for init time: {t0}")
    logger.info(f"Running critical models only: {settings.run_critical_models_only}")

    # If a scratch directory is configured, use it. Otherwise, create a temporary directory which
    # will be deleted after the forecast is complete.
    if settings.scratch_dir is not None:
        # Create the scratch directory if it doesn't exist. If it does exist, raise an error to
        # avoid clashing with data from a previous run.
        os.makedirs(settings.scratch_dir, exist_ok=False)
        dir_context = contextlib.nullcontext(settings.scratch_dir)
    else:
        dir_context = tempfile.TemporaryDirectory(prefix="pvnet-app-")

    with dir_context as scratch_dir:
        logger.info(f"Using scratch directory: {scratch_dir}")
        return await _run_forecast_pipeline(settings, t0, scratch_dir, write_predictions)


async def _run_forecast_pipeline(
    settings: AppSettings,
    t0: pd.Timestamp,
    scratch_dir: str,
    write_predictions: bool,
) -> dict | None:
    """Run the forecast pipeline for all configured models.

    Prepares inputs, runs each model whose data is available, validates the
    forecasts, and writes them to the data platform.

    Args:
        settings: The application settings
        t0: The forecast init-time. Must be in naive-UTC and floored to 30 minutes
        scratch_dir: Directory for downloaded inputs and temporary files
        write_predictions: If True, write forecasts to the data platform. If
            False, skip the write and return the forecasters for local testing
    """
    # ---------------------------------------------------------------------------
    # Basic set up

    # --- Get the model configurations
    model_specs = get_model_specs(get_critical_only=settings.run_critical_models_only)

    if len(model_specs) == 0:
        raise Exception("No models found after filtering")

    # Fetch the models' data configs paths and load the configs
    data_config_paths = fetch_model_data_config_paths(model_specs, settings.huggingface_token)
    data_configs = [load_yaml_config(path) for path in data_config_paths.values()]

    # ---------------------------------------------------------------------------
    # Prepare data sources

    # --- Get locations metadata from the data platform
    logger.info("Loading locations")

    async with Channel(settings.data_platform_host, settings.data_platform_port) as dp_channel:
        locations = await fetch_locations(client=dp.DataPlatformDataServiceStub(dp_channel))

    capacities_mwp = extract_location_capacities_mwp(locations)

    # Save out a dummy generation dataset to the scratch directory
    # The models don't actually use this data, but the current ocf-data-sampler version requires it
    ds_gen = create_null_generation_data(t0=t0, capacities_mwp=capacities_mwp)
    ds_gen.to_zarr(f"{scratch_dir}/{generation_path}")

    data_downloaders = []

    # --- Try to download satellite data if any models require it
    if any("satellite" in conf["input_data"] for conf in data_configs):
        logger.info("Downloading satellite data")

        sat_downloader = SatelliteDownloader(
            t0=t0,
            source_path_5=settings.satellite_icechunk_path_5,
            source_path_15=settings.satellite_icechunk_path_15,
            s3_region=settings.satellite_s3_region,
            destination_path=f"{scratch_dir}/{sat_path}",
        )
        sat_downloader.run()

        data_downloaders.append(sat_downloader)

    # --- Try to download NWP data if any models require it
    if any("nwp" in conf["input_data"] for conf in data_configs):
        logger.info("Downloading NWP data")

        # Only download the NWP sources which are required by the models
        required_providers = get_required_nwp_providers(data_configs)

        if "ukv" in required_providers:
            ukv_downloader = UKVDownloader(
                source_path=settings.nwp_ukv_zarr_path,
                destination_path=f"{scratch_dir}/{nwp_ukv_path}",
            )
            ukv_downloader.run()

            data_downloaders.append(ukv_downloader)

        if "ecmwf" in required_providers:
            ecmwf_downloader = ECMWFDownloader(
                source_path=settings.nwp_ecmwf_zarr_path,
                destination_path=f"{scratch_dir}/{nwp_ecmwf_path}",
            )
            ecmwf_downloader.run()

            data_downloaders.append(ecmwf_downloader)

        if "cloudcasting" in required_providers:
            cloudcasting_downloader = CloudcastingDownloader(
                source_path=settings.cloudcasting_zarr_path,
                destination_path=f"{scratch_dir}/{nwp_cloudcasting_path}",
            )
            cloudcasting_downloader.run()

            data_downloaders.append(cloudcasting_downloader)

    # ---------------------------------------------------------------------------
    # Set up models

    # Prepare all the models which can be run
    model_forecasters = {}
    for model_spec in model_specs:

        # Check if the data available will allow the model to run
        data_config_path = data_config_paths[model_spec.name]

        logger.info(f"Checking that the input data for model '{model_spec.name}' exists")

        model_can_run = all(
            downloader.check_model_inputs_available(data_config_path, t0)
            for downloader in data_downloaders
        )

        if model_can_run:
            logger.info(f"The input data for model '{model_spec.name}' is available")

            model_forecasters[model_spec.name] = PVNetForecaster(
                model_spec=model_spec,
                data_config_path=data_config_path,
                run_data_dir=scratch_dir,
                t0=t0,
                device=device,
                capacities=capacities_mwp,
                hf_token=settings.huggingface_token,
            )

        else:
            logger.warning(f"The model {model_spec.name} cannot be run with input data available")

    if len(model_forecasters) == 0:
        raise Exception("No models were compatible with the available input data.")

    # ---------------------------------------------------------------------------
    # Make predictions

    logger.info("Making predictions")

    forecasts: dict[str, xr.DataArray] = {}

    for i, (model_name, forecaster) in enumerate(model_forecasters.items()):
        batch = forecaster.make_batch()

        # Do basic validation of the batch: Will raise error if the batch fails the checks
        try:
            check_batch(batch)
            validate_batch_against_config(batch=batch, model=forecaster.model)
        except Exception as e:
            logger.error(f"Batch validation failed for model {model_name}: {e}")
            continue

        forecasts[model_name] = forecaster.predict(batch)

        # Save the batch for the first model
        if (settings.save_batches_dir is not None) and i == 0:
            save_batch_to_s3(batch, model_name, settings.save_batches_dir, scratch_dir=scratch_dir)


    # ---------------------------------------------------------------------------
    # Run validation checks on the forecast values

    logger.info("Validating forecasts")
    for model_name, da_forecast in list(forecasts.items()):

        forecast_okay = validate_forecast(
            da_forecast=da_forecast,
            national_capacity_mw=capacities_mwp[0],
            zig_zag_warning_threshold_mw=settings.forecast_validate_zig_zag_warning_threshold,
            zig_zag_error_threshold_mw=settings.forecast_validate_zig_zag_error_threshold,
            sun_elevation_lower_limit=settings.forecast_validate_sun_elevation_lower_limit,
            model_name=model_name,
        )

        if not forecast_okay:
            logger.warning(f"Forecast for model {model_name} failed validation")
            if settings.filter_bad_forecasts:
                # This forecast will not be saved
                del forecasts[model_name]

    if len(forecasts) == 0:
        raise Exception("No models passed the forecast validation checks")

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return forecasts

    # ---------------------------------------------------------------------------
    # Write predictions to data-platform

    logger.info("Writing to data platform")

    async with Channel(settings.data_platform_host, settings.data_platform_port) as dp_channel:
        dp_client = dp.DataPlatformDataServiceStub(dp_channel)

        await write_forecasts_to_data_platform(
            client=dp_client,
            forecasts=forecasts,
            locations=locations,
            t0=t0,
            input_s3_paths={
                "nwp_ecmwf": settings.nwp_ecmwf_zarr_path,
                "nwp_ukv": settings.nwp_ukv_zarr_path,
                "satellite": settings.satellite_icechunk_path_5,
                "satellite_15": settings.satellite_icechunk_path_15,
            },
            app_version=__version__,
        )

    logger.info("Finished forecast")

    if settings.raise_model_failure in ["any", "critical"]:
        check_model_runs_finished(
            completed_forecasts=list(forecasts.keys()),
            model_specs=model_specs,
            raise_if_missing=settings.raise_model_failure,
        )

def main() -> None:
    """Main entrypoint to the inference app."""
    settings = AppSettings()
    asyncio.run(run_app(settings=settings))
