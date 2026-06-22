"""Application to run inference for PVNet multiple models."""

import asyncio
import logging
import os
from importlib.metadata import version

import pandas as pd
import sentry_sdk
import torch
from grpclib.client import Channel
from ocf import dp
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.config import load_yaml_config
from pvnet_app.consts import generation_path
from pvnet_app.data.batch_validation import check_batch
from pvnet_app.data.gsp import create_null_generation_data
from pvnet_app.data.nwp import CloudcastingDownloader, ECMWFDownloader, UKVDownloader
from pvnet_app.data.satellite import SatelliteDownloader
from pvnet_app.forecaster import Forecaster
from pvnet_app.model_configs.pydantic_models import get_all_models
from pvnet_app.save import build_input_metadata, extract_location_capacities_mwp, fetch_locations
from pvnet_app.settings import AppSettings
from pvnet_app.utils import check_model_runs_finished, save_batch_to_s3
from pvnet_app.validate_forecast import validate_forecast

__version__ = version("pvnet-app")

# ---------------------------------------------------------------------------
# LOGGING AND SENTRY

# Create a logger
logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Turn off logs from aiobotocore
logging.getLogger("aiobotocore").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# APP MAIN

async def run(
    settings: AppSettings,
    t0: str | None = None,
    write_predictions: bool = True,
) -> None:
    """Inference function to run PVNet.

    Args:
        settings: The application settings
        t0 (str): Datetime at which forecast is made
        write_predictions (bool): Whether to write prediction to the database. Else returns as
            DataArray for local testing.
    """
    # ---------------------------------------------------------------------------
    # Basic set up

    # -- Initialize Sentry
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        traces_sample_rate=1,
    )
    sentry_sdk.set_tag("app_name", "pvnet_app")
    sentry_sdk.set_tag("version", __version__)

    # If inference datetime is None, set to now
    t0 = pd.Timestamp.now(tz="UTC") if t0 is None else pd.Timestamp(t0).tz_localize("UTC")
    # Round down to last 30 minutes
    t0 = t0.replace(tzinfo=None).floor("30min")

    # --- Log version and variables
    pvnet_version = version("pvnet")
    logger.info(f"Using `pvnet` library version: {pvnet_version}")
    logger.info(f"Using `pvnet_app` library version: {__version__}")
    logger.info(f"Making forecast for init time: {t0}")
    logger.info(f"Running critical models only: {settings.run_critical_models_only}")

    # --- Get the model configurations
    model_configs = get_all_models(get_critical_only=settings.run_critical_models_only)

    if len(model_configs) == 0:
        raise Exception("No models found after filtering")

    # Get Model configs
    data_config_paths: dict[str, str] = {}
    data_configs: list[dict] = []
    for model_config in model_configs:
        # First load the data config
        data_config_path = PVNetBaseModel.get_data_config(
            model_config.pvnet.repo,
            revision=model_config.pvnet.commit,
            token=settings.huggingface_token,
        )
        data_config_paths[model_config.name] = data_config_path
        data_configs.append(load_yaml_config(data_config_path))

    # ---------------------------------------------------------------------------
    #  Prepare data sources

    # --- Get locations metadata from the data platform
    logger.info("Loading locations")

    async with Channel(settings.data_platform_host, settings.data_platform_port) as dp_channel:
        locations = await fetch_locations(client=dp.DataPlatformDataServiceStub(dp_channel))

    ds_gen = create_null_generation_data(
        t0=t0,
        capacities_mwp=extract_location_capacities_mwp(locations),
    )

    ds_gen.to_zarr(generation_path)

    national_capacity = ds_gen.sel(time_utc=t0, location_id=0).capacity_mwp.item()
    gsp_capacities = ds_gen.sel(time_utc=t0, location_id=slice(1, None)).capacity_mwp.values
    gsp_ids = ds_gen.location_id.values

    data_downloaders = []

    # --- Try to download satellite data if any models require it
    if any("satellite" in conf["input_data"] for conf in data_configs):
        logger.info("Downloading satellite data")

        sat_downloader = SatelliteDownloader(
            t0=t0,
            source_path_5=settings.satellite_icechunk_path_5,
            source_path_15=settings.satellite_icechunk_path_15,
            s3_region=settings.satellite_s3_region,
        )
        sat_downloader.run()

        data_downloaders.append(sat_downloader)

    # --- Try to download NWP data if any models require it
    if any("nwp" in conf["input_data"] for conf in data_configs):
        logger.info("Downloading NWP data")

        # Find the NWP sources required by the models
        required_providers = set()
        for conf in data_configs:
            if "nwp" in conf["input_data"]:
                for source in conf["input_data"]["nwp"].values():
                    required_providers.add(source["provider"])

        if "ukv" in required_providers:
            ukv_downloader = UKVDownloader(source_path=settings.nwp_ukv_zarr_path)
            ukv_downloader.run()

            data_downloaders.append(ukv_downloader)

        if "ecmwf" in required_providers:
            ecmwf_downloader = ECMWFDownloader(source_path=settings.nwp_ecmwf_zarr_path)
            ecmwf_downloader.run()

            data_downloaders.append(ecmwf_downloader)

        if "cloudcasting" in required_providers:
            cloudcasting_downloader = CloudcastingDownloader(
                source_path=settings.cloudcasting_zarr_path,
            )
            cloudcasting_downloader.run()

            data_downloaders.append(cloudcasting_downloader)

    # ---------------------------------------------------------------------------
    # Set up models

    # Prepare all the models which can be run
    forecasters = {}
    for model_config in model_configs:
        # First load the data config
        data_config_path = data_config_paths[model_config.name]

        # Check if the data available will allow the model to run
        logger.info(f"Checking that the input data for model '{model_config.name}' exists")
        model_can_run = all(
            downloader.check_model_inputs_available(data_config_path, t0)
            for downloader in data_downloaders
        )

        if model_can_run:
            logger.info(f"The input data for model '{model_config.name}' is available")
            # Set up a forecast compiler for the model
            forecasters[model_config.name] = Forecaster(
                model_config=model_config,
                data_config_path=data_config_path,
                t0=t0,
                gsp_ids=gsp_ids,
                device=device,
                gsp_capacities=gsp_capacities,
                national_capacity=national_capacity,
                hf_token=settings.huggingface_token,
            )

        else:
            logger.warning(f"The model {model_config.name} cannot be run with input data available")

    if len(forecasters) == 0:
        raise Exception("No models were compatible with the available input data.")

    # ---------------------------------------------------------------------------
    # Make predictions

    logger.info("Making predictions")

    for i, (model_name, forecaster) in enumerate(forecasters.items()):
        batch = forecaster.make_batch()

        # Do basic validation of the batch: Will raise error if the batch fails the checks
        check_batch(batch)

        if (settings.save_batches_dir is not None) and i == 0:
            # Save the batch under the name of the first model
            save_batch_to_s3(batch, model_name, settings.save_batches_dir)

        forecaster.predict(batch)

    # Delete the downloaded data
    for downloader in data_downloaders:
        downloader.clean_up()

    # ---------------------------------------------------------------------------
    # Run validation checks on the forecast values

    logger.info("Validating forecasts")
    for model_name in list(forecasters.keys()):
        national_forecast = (
            forecasters[model_name].da_abs_all.sel(gsp_id=0, output_label="p50")
        ).to_series()

        forecast_okay = validate_forecast(
            national_forecast=national_forecast,
            national_capacity=national_capacity,
            zig_zag_warning_threshold=settings.forecast_validate_zig_zag_warning_threshold,
            zig_zag_error_threshold=settings.forecast_validate_zig_zag_error_threshold,
            sun_elevation_lower_limit=settings.forecast_validate_sun_elevation_lower_limit,
            model_name=model_name,
        )

        if not forecast_okay:
            logger.warning(f"Forecast for model {model_name} failed validation")
            if settings.filter_bad_forecasts:
                # This forecast will not be saved
                del forecasters[model_name]

    if len(forecasters) == 0:
        raise Exception("No models passed the forecast validation checks")

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return forecasters

    # ---------------------------------------------------------------------------
    # Write predictions to data-platform

    logger.info("Writing to data platform")

    async with Channel(settings.data_platform_host, settings.data_platform_port) as dp_channel:
        dp_client = dp.DataPlatformDataServiceStub(dp_channel)

        input_metadata = await build_input_metadata(
            client=dp_client,
            location_uuid=locations[0].location_uuid,
            input_s3_paths = {
                "nwp_ecmwf": settings.nwp_ecmwf_zarr_path,
                "nwp_ukv": settings.nwp_ukv_zarr_path,
                "satellite": settings.satellite_icechunk_path_5,
                "satellite_15": settings.satellite_icechunk_path_15,
            },
            app_version=__version__,
        )

        all_requests: list[list[dp.CreateForecastRequest]] = await asyncio.gather(
            *(
                forecaster.create_write_requests(
                    client=dp_client,
                    locations=locations,
                    metadata=input_metadata,
                )
                for forecaster in forecasters.values()
            ),
        )

        write_results = await asyncio.gather(
            *(dp_client.create_forecast(req) for reqs in all_requests for req in reqs),
            return_exceptions=True,
        )

    for exc in filter(lambda x: isinstance(x, Exception), write_results):
        raise exc

    logger.info("Finished forecast")

    if settings.raise_model_failure in ["any", "critical"]:
        check_model_runs_finished(
            completed_forecasts=list(forecasters.keys()),
            model_configs=model_configs,
            raise_if_missing=settings.raise_model_failure,
        )

def main() -> None:
    """Main entrypoint to the inference app."""
    settings = AppSettings()
    asyncio.run(run(settings=settings))
