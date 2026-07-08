"""Application to run forecasts with a collection of PVNet models."""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import warnings
from datetime import datetime
from importlib.metadata import version

import pandas as pd
import sentry_sdk
import torch
import xarray as xr
from grpclib.client import Channel
from huggingface_hub.utils import disable_progress_bars
from ocf import dp

from pvnet_app.consts import (
    generation_path,
    nwp_cloudcasting_path,
    nwp_ecmwf_path,
    nwp_ukv_path,
    sat_path,
)
from pvnet_app.data.batch_validation import get_batch_validation_error
from pvnet_app.data.gsp import create_null_generation_data
from pvnet_app.data.nwp import CloudcastingDownloader, ECMWFDownloader, NWPDownloader, UKVDownloader
from pvnet_app.data.satellite import SatelliteDownloader
from pvnet_app.data_platform import (
    extract_location_capacities_mwp,
    fetch_locations,
    write_forecasts_to_data_platform,
)
from pvnet_app.forecaster import PVNetForecaster
from pvnet_app.model_input_config import (
    fetch_model_data_config_paths,
    get_maximum_nwp_spatial_window_sizes,
    get_maximum_satellite_spatial_window_size,
    get_required_nwp_providers,
    get_required_satellite_interval,
    load_yaml_config,
)
from pvnet_app.models.registry import get_model_specs
from pvnet_app.settings import AppSettings
from pvnet_app.utils import check_model_runs_finished, log_duration, resolve_t0, save_batch_to_s3
from pvnet_app.validate_forecast import validate_forecast

__version__ = version("pvnet-app")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# APP MAIN


async def run_app(
    settings: AppSettings,
    t0: str | datetime | pd.Timestamp | None = None,
    write_predictions: bool = True,
) -> None | dict[str, xr.DataArray]:
    """Set up the app environment and run a forecast.

    Handles Sentry, init-time resolution, and the scratch directory, then
    delegates to the forecast pipeline.

    Args:
        settings: The application settings
        t0: The forecast init-time. If None, the current time is used. Input will be floored to the
            previous 30-minutes.
        write_predictions: If True, write forecasts to the data platform. If
            False, skip the write and return the forecasts for local testing
    """
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s",
        force=True,
    )

    # Filter some logs and warnings
    logging.getLogger("aiobotocore").setLevel(logging.ERROR)
    logging.getLogger("ocf_data_sampler.load.load_dataset").setLevel(logging.WARNING)
    logging.getLogger("pvnet.utils").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=("The data type \\(FixedLengthUTF32.* does not have a Zarr V3 specification.*"),
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            "Consolidated metadata is currently not part in the Zarr format 3 specification.*"
        ),
    )
    disable_progress_bars()

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )
    sentry_sdk.set_tag("app_name", "pvnet_app")
    sentry_sdk.set_tag("version", __version__)

    t0 = resolve_t0(t0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using `pvnet` library version: {version('pvnet')}")
    logger.info(f"Using `pvnet_app` library version: {__version__}")
    logger.info(f"Using device: {device}")
    logger.info(f"Making forecast for init time: {t0}")
    resolved_settings = settings.model_dump(exclude={"huggingface_token", "sentry_dsn"})
    logger.info(
        "Resolved app settings:\n%s",
        json.dumps(resolved_settings, indent=2, sort_keys=True),
    )

    # If a scratch directory is configured, treat it as the parent directory for persistent
    # per-run workspaces. Otherwise, create a temporary directory which will be deleted after the
    # forecast is complete.
    if settings.scratch_dir is not None:
        os.makedirs(settings.scratch_dir, exist_ok=True)
        t0_dirname = t0.strftime("%Y%m%dT%H%M%SZ")
        run_scratch_dir = tempfile.mkdtemp(
            prefix=f"pvnet-app-{t0_dirname}-",
            dir=settings.scratch_dir,
        )
        dir_context = contextlib.nullcontext(run_scratch_dir)
    else:
        dir_context = tempfile.TemporaryDirectory(prefix="pvnet-app-")

    with dir_context as scratch_dir:
        logger.info(f"Using scratch directory: {scratch_dir}")
        return await _run_forecast_pipeline(settings, t0, scratch_dir, write_predictions, device)


async def _run_forecast_pipeline(
    settings: AppSettings,
    t0: pd.Timestamp,
    scratch_dir: str,
    write_predictions: bool,
    device: torch.device,
) -> dict | None:
    """Run the forecast pipeline for all configured models.

    Prepares inputs, runs each model whose data is available, validates the
    forecasts, and writes them to the data platform.

    Args:
        settings: The application settings
        t0: The forecast init-time. Must be in naive-UTC and floored to 30 minutes
        scratch_dir: Directory for downloaded inputs and temporary files
        write_predictions: If True, write forecasts to the data platform. If
            False, skip the write and return the forecasts for local testing
        device: Device to run the models on
    """
    # ---------------------------------------------------------------------------
    # Basic set up

    # --- Get the model configurations
    model_specs = get_model_specs(get_critical_only=settings.run_critical_models_only)

    if len(model_specs) == 0:
        raise ValueError("No models found after filtering")

    # Fetch the models' data configs paths and load the configs
    with log_duration(logger, "Fetching model data configs"):
        data_config_paths = fetch_model_data_config_paths(model_specs, settings.huggingface_token)
        data_configs = [load_yaml_config(path) for path in data_config_paths.values()]

    # ---------------------------------------------------------------------------
    # Prepare data sources

    # --- Get locations metadata from the data platform
    with log_duration(logger, "Loading locations"):
        async with Channel(settings.data_platform_host, settings.data_platform_port) as dp_channel:
            locations = await fetch_locations(client=dp.DataPlatformDataServiceStub(dp_channel))

    capacities_mwp = extract_location_capacities_mwp(locations)

    # Save out a dummy generation dataset to the scratch directory
    # The models don't actually use this data, but the current ocf-data-sampler version requires it
    ds_gen = create_null_generation_data(t0=t0, capacities_mwp=capacities_mwp)
    ds_gen.to_zarr(f"{scratch_dir}/{generation_path}")

    named_downloaders: list[tuple[str, SatelliteDownloader | NWPDownloader]] = []

    # --- Try to download satellite data if any models require it
    if any("satellite" in conf["input_data"] for conf in data_configs):
        # Only download and check the satellite data which is required by the models
        interval_start_minutes, interval_end_minutes = get_required_satellite_interval(data_configs)
        window_size = get_maximum_satellite_spatial_window_size(data_configs)

        sat_downloader = SatelliteDownloader(
            t0=t0,
            source_path_5=settings.satellite_icechunk_path_5,
            source_path_15=settings.satellite_icechunk_path_15,
            s3_region=settings.satellite_s3_region,
            destination_path=f"{scratch_dir}/{sat_path}",
            interval_start_minutes=interval_start_minutes,
            interval_end_minutes=interval_end_minutes,
            window_size_pixels=window_size,
        )

        named_downloaders.append(("satellite", sat_downloader))

    # --- Try to download NWP data if any models require it
    if any("nwp" in conf["input_data"] for conf in data_configs):
        # Only download the NWP sources which are required by the models
        required_providers = get_required_nwp_providers(data_configs)
        nwp_window_sizes = get_maximum_nwp_spatial_window_sizes(data_configs)

        if "ukv" in required_providers:
            ukv_downloader = UKVDownloader(
                source_path=settings.nwp_ukv_zarr_path,
                destination_path=f"{scratch_dir}/{nwp_ukv_path}",
                window_size_pixels=nwp_window_sizes["ukv"],
            )

            named_downloaders.append(("UKV", ukv_downloader))

        if "ecmwf" in required_providers:
            with log_duration(logger, "Downloading ECMWF data"):
                ecmwf_downloader = ECMWFDownloader(
                    source_path=settings.nwp_ecmwf_zarr_path,
                    destination_path=f"{scratch_dir}/{nwp_ecmwf_path}",
                    window_size_pixels=nwp_window_sizes["ecmwf"],
                )

            named_downloaders.append(("ECMWF", ecmwf_downloader))

        if "cloudcasting" in required_providers:
            with log_duration(logger, "Downloading cloudcasting data"):
                cloudcasting_downloader = CloudcastingDownloader(
                    source_path=settings.cloudcasting_zarr_path,
                    destination_path=f"{scratch_dir}/{nwp_cloudcasting_path}",
                    window_size_pixels=nwp_window_sizes["cloudcasting"],
                )

            named_downloaders.append(("cloudcasting", cloudcasting_downloader))

    # --- Run all downloads concurrently in threads
    with log_duration(logger, "Downloading all input data"):
        await _run_downloaders_concurrently(named_downloaders)

    data_downloaders = [d for _, d in named_downloaders]

    # ---------------------------------------------------------------------------
    # Set up models

    # Prepare all the models which can be run
    model_forecasters = {}
    skipped_models = []
    with log_duration(logger, "Preparing runnable models"):
        for model_spec in model_specs:
            data_config_path = data_config_paths[model_spec.name]

            model_can_run = all(
                downloader.check_model_inputs_available(data_config_path, t0)
                for downloader in data_downloaders
            )

            if model_can_run:
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
                logger.warning(
                    f"The model {model_spec.name} cannot be run with input data available",
                )
                skipped_models.append(model_spec.name)

    logger.info(
        "Prepared runnable models: runnable=%d skipped=%d runnable_models=%s skipped_models=%s",
        len(model_forecasters),
        len(skipped_models),
        sorted(model_forecasters),
        skipped_models,
    )

    if len(model_forecasters) == 0:
        raise ValueError("No models were compatible with the available input data.")

    # ---------------------------------------------------------------------------
    # Make predictions

    forecasts: dict[str, xr.DataArray] = {}

    with log_duration(logger, "Making all forecasts"):
        for i, (model_name, forecaster) in enumerate(model_forecasters.items()):
            batch = forecaster.make_batch()

            # Check the batch for any validation errors before running the model
            batch_val_error = get_batch_validation_error(batch, model=forecaster.model)
            if batch_val_error is None:
                forecasts[model_name] = forecaster.predict(batch)
            else:
                logger.error(f"Batch validation failed for model {model_name}: {batch_val_error}")

            # Save the batch for the first model
            if (settings.save_batches_dir is not None) and i == 0:
                save_batch_to_s3(
                    batch,
                    model_name,
                    settings.save_batches_dir,
                    scratch_dir=scratch_dir,
                )

    # ---------------------------------------------------------------------------
    # Run validation checks on the forecast values

    with log_duration(logger, "Validating forecasts"):
        for model_name, da_forecast in list(forecasts.items()):
            forecast_okay = validate_forecast(
                da_forecast=da_forecast,
                national_capacity_mw=capacities_mwp[0],
                zig_zag_warning_threshold_mw=settings.forecast_validate_zig_zag_warning_threshold,
                zig_zag_error_threshold_mw=settings.forecast_validate_zig_zag_error_threshold,
                national_max_forecast_mw=settings.forecast_validate_national_max_mw,
                sun_elevation_lower_limit=settings.forecast_validate_sun_elevation_lower_limit,
                model_name=model_name,
            )

            if not forecast_okay:
                logger.warning(f"Forecast for model {model_name} failed validation")
                if settings.filter_bad_forecasts:
                    # This forecast will not be saved
                    del forecasts[model_name]

    if len(forecasts) == 0:
        raise ValueError("No models passed the forecast validation checks")

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return forecasts

    # ---------------------------------------------------------------------------
    # Write predictions to data-platform

    with log_duration(logger, "Writing to data platform"):
        async with Channel(settings.data_platform_host, settings.data_platform_port) as dp_channel:
            dp_client = dp.DataPlatformDataServiceStub(dp_channel)

            await write_forecasts_to_data_platform(
                client=dp_client,
                forecasts=forecasts,
                locations=locations,
                t0=t0,
                input_paths={
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


def _run_downloader_with_timing(name: str, downloader: SatelliteDownloader | NWPDownloader) -> None:
    """Run a data downloader, logging its duration. Used as a thread target."""
    with log_duration(logger, f"Downloading {name} data"):
        downloader.run()


async def _run_downloaders_concurrently(
    named_downloaders: list[tuple[str, SatelliteDownloader | NWPDownloader]],
) -> None:
    """Run all data downloaders concurrently in threads.

    Fails fast if any downloader raises, but only after all have finished, so no
    thread is left writing to the scratch directory during teardown.

    Raises:
        ExceptionGroup: If one or more downloaders fail
    """
    results = await asyncio.gather(
        *(asyncio.to_thread(_run_downloader_with_timing, name, d) for name, d in named_downloaders),
        return_exceptions=True,
    )

    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise ExceptionGroup("Failed downloading input data", errors)


def main() -> None:
    """Main entrypoint to the inference app."""
    settings = AppSettings()
    asyncio.run(run_app(settings=settings))
