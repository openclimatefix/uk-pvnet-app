"""Application to run inference for PVNet multiple models."""

import logging
import os
from importlib.metadata import version

import pandas as pd
import sentry_sdk
import torch
import typer
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from ocf_data_sampler.load.gsp import get_gsp_boundaries
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.config import load_yaml_config
from pvnet_app.data.batch_validation import check_batch
from pvnet_app.data.gsp import get_gsp_and_national_capacities
from pvnet_app.data.nwp import CloudcastingDownloader, ECMWFDownloader, UKVDownloader
from pvnet_app.data.satellite import SatelliteDownloader, get_satellite_source_paths
from pvnet_app.forecaster import Forecaster
from pvnet_app.model_configs.pydantic_models import get_all_models
from pvnet_app.utils import check_model_runs_finished, get_boolean_env_var, save_batch_to_s3
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

# Get rid of the verbose sqlalchemy logs
logging.getLogger("sqlalchemy").setLevel(logging.ERROR)
# Turn off logs from aiobotocore
logging.getLogger("aiobotocore").setLevel(logging.ERROR)

# Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("ENVIRONMENT", "local"),
    traces_sample_rate=1,
)

sentry_sdk.set_tag("app_name", "pvnet_app")
sentry_sdk.set_tag("version", __version__)

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# APP MAIN


def app(
    t0: str | None = None,
    gsp_ids: list[int] | None = None,
    write_predictions: bool = True,
) -> None:
    """Inference function to run PVNet.

    Args:
        t0 (str): Datetime at which forecast is made
        gsp_ids (array_like): List of gsp_ids to make predictions for. This list of GSPs are summed
            to national.
        write_predictions (bool): Whether to write prediction to the database. Else returns as
            DataArray for local testing.

    This app requires these environmental variables to be available:
        - DB_URL
    These variables are optional depending on the models being run:
        - NWP_UKV_ZARR_PATH
        - NWP_ECMWF_ZARR_PATH
        - CLOUDCASTING_ZARR_PATH
        - SATELLITE_ZARR_PATH
    The following are optional:
        - SENTRY_DSN, optional link to sentry
        - ENVIRONMENT, the environment this is running in, defaults to local
        - ALLOW_ADJUSTER: Option to allow the adjuster to be used. If false this overwrites the
          adjuster option in the model configs so it is not used. Defaults to true.
        - ALLOW_SAVE_GSP_SUM: Option to allow model to save the GSP sum. If false this overwrites
          the model configs so saving of the GSP sum is not used. Defaults to false.
        - RUN_CRITICAL_MODELS_ONLY, option to run critical models only, defaults to false
        - FORECAST_VALIDATE_ZIG_ZAG_WARNING, threshold for forecast zig-zag warning,
          defaults to 250 MW.
        - FORECAST_VALIDATE_ZIG_ZAG_ERROR, threshold for forecast zig-zag error on,
          defaults to 500 MW.
        - FORECAST_VALIDATION_SUN_ELEVATION_LOWER_LIMIT, when the solar elevation is above this,
          we expect positive forecast values. Defaults to 10 degrees.
        - FILTER_BAD_FORECASTS, option to filter out bad forecasts. If set to true and the forecast
          fails the validation checks, it will not be saved. Defaults to false, where all forecasts
          are saved even if they fail the checks.
        - RAISE_MODEL_FAILURE: Option to raise an exception if a model fails to run. If set to
          "any" it will raise an exception if any model fails. If set to "critical" it will raise
          an exception if any critical model fails. If not set, it will not raise an exception.
    """
    # ---------------------------------------------------------------------------
    # 0. Basic set up

    # If inference datetime is None, round down to last 30 minutes
    if t0 is None:
        t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor("30min")
    else:
        t0 = pd.Timestamp(t0).floor("30min")

    if gsp_ids is not None and len(gsp_ids) == 0:
            raise ValueError("No GSP IDs provided")

    # --- Unpack the environment variables
    run_critical_models_only = get_boolean_env_var("RUN_CRITICAL_MODELS_ONLY", default=False)
    allow_adjuster = get_boolean_env_var("ALLOW_ADJUSTER", default=True)
    allow_save_gsp_sum = get_boolean_env_var("ALLOW_SAVE_GSP_SUM", default=False)
    filter_bad_forecasts = get_boolean_env_var("FILTER_BAD_FORECASTS", default=False)
    raise_model_failure = os.getenv("RAISE_MODEL_FAILURE", None)

    zig_zag_warning_threshold = float(os.getenv("FORECAST_VALIDATE_ZIG_ZAG_WARNING", 250))
    zig_zag_error_threshold = float(os.getenv("FORECAST_VALIDATE_ZIG_ZAG_ERROR", 500))
    sun_elevation_lower_limit = float(os.getenv("FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT", 10))

    db_url = os.environ["DB_URL"]  # Will raise KeyError if not set
    s3_batch_save_dir = os.getenv("SAVE_BATCHES_DIR", None)
    ecmwf_source_path = os.getenv("NWP_ECMWF_ZARR_PATH", None)
    ukv_source_path = os.getenv("NWP_UKV_ZARR_PATH", None)
    cloudcasting_source_path = os.getenv("CLOUDCASTING_ZARR_PATH", None)
    sat_source_path_5, sat_source_path_15 = get_satellite_source_paths()

    # --- Log version and variables
    pvnet_version = version("pvnet")
    logger.info(f"Using `pvnet` library version: {pvnet_version}")
    logger.info(f"Using `pvnet_app` library version: {__version__}")
    logger.info(f"Making forecast for init time: {t0}")
    logger.info(f"Making forecast for GSP IDs: {gsp_ids}")
    logger.info(f"Running critical models only: {run_critical_models_only}")
    logger.info(f"Allow adjuster: {allow_adjuster}")
    logger.info(f"Allow saving GSP sum: {allow_save_gsp_sum}")

    # --- Get the model configurations
    model_configs = get_all_models(
        allow_adjuster=allow_adjuster,
        allow_save_gsp_sum=allow_save_gsp_sum,
        get_critical_only=run_critical_models_only,
    )

    if len(model_configs) == 0:
        raise Exception("No models found after filtering")

    # Open connection to the database - used for pulling GSP capacitites and writing forecasts
    db_connection = DatabaseConnection(url=db_url, base=Base_Forecast, echo=False)

    # 0. Get Model configs
    data_config_paths: dict[str, str] = {}
    data_configs: list[dict] = []
    for model_config in model_configs:
        # First load the data config
        data_config_path = PVNetBaseModel.get_data_config(
            model_config.pvnet.repo,
            revision=model_config.pvnet.commit,
        )
        data_config_paths[model_config.name] = data_config_path
        data_configs.append(load_yaml_config(data_config_path))

    # ---------------------------------------------------------------------------
    # 1. Prepare data sources

    if gsp_ids is None:
        gsp_ids = get_gsp_boundaries(version="20250109").iloc[1:].index.tolist()

    # --- Get capacities from the database
    logger.info("Loading capacities from the database")
    gsp_capacities, national_capacity = get_gsp_and_national_capacities(
        db_connection=db_connection,
        gsp_ids=gsp_ids,
        t0=t0,
    )

    data_downloaders = []

    # --- Try to download satellite data if any models require it
    if any("satellite" in conf["input_data"] for conf in data_configs):
        logger.info("Downloading satellite data")

        sat_downloader = SatelliteDownloader(
            t0=t0,
            source_path_5=sat_source_path_5,
            source_path_15=sat_source_path_15,
            gsp_ids=gsp_ids,
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
            ukv_downloader = UKVDownloader(source_path=ukv_source_path)
            ukv_downloader.run()

            data_downloaders.append(ukv_downloader)

        if "ecmwf" in required_providers:
            ecmwf_downloader = ECMWFDownloader(source_path=ecmwf_source_path)
            ecmwf_downloader.run()

            data_downloaders.append(ecmwf_downloader)

        if "cloudcasting" in required_providers:
            cloudcasting_downloader = CloudcastingDownloader(source_path=cloudcasting_source_path)
            cloudcasting_downloader.run()

            data_downloaders.append(cloudcasting_downloader)

    # ---------------------------------------------------------------------------
    # 2. Set up models

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
            )

        else:
            logger.warning(f"The model {model_config.name} cannot be run with input data available")

    if len(forecasters) == 0:
        raise Exception("No models were compatible with the available input data.")

    # ---------------------------------------------------------------------------
    # Make predictions
    logger.info("Making predictions")

    first_model_name = next(iter(forecasters.keys()))
    for model_name, forecaster in forecasters.items():
        batch = forecaster.make_batch()

        # Do basic validation of the batch: Will raise error if the batch fails the checks
        check_batch(batch)

        if (s3_batch_save_dir is not None) and model_name == first_model_name:
            # Save the batch under the name of the first model
            save_batch_to_s3(batch, model_name, s3_batch_save_dir)

        forecaster.predict(batch)

    # Delete the downloaded data
    for downloader in data_downloaders:
        downloader.clean_up()

    # ---------------------------------------------------------------------------
    # Run validation checks on the forecast values
    logger.info("Validating forecasts")
    for model_name in list(forecasters.keys()):
        national_forecast = (
            forecasters[model_name].da_abs_all.sel(gsp_id=0, output_label="forecast_mw")
        ).to_series()

        forecast_okay = validate_forecast(
            national_forecast=national_forecast,
            national_capacity=national_capacity,
            zip_zag_warning_threshold=zig_zag_warning_threshold,
            zig_zag_error_threshold=zig_zag_error_threshold,
            sun_elevation_lower_limit=sun_elevation_lower_limit,
            model_name=model_name,
        )

        if not forecast_okay:
            logger.warning(f"Forecast for model {model_name} failed validation")
            if filter_bad_forecasts:
                # This forecast will not be saved
                del forecasters[model_name]

    if len(forecasters) == 0:
        raise Exception("No models passed the forecast validation checks")

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return next(iter(forecasters.values())).da_abs_all

    # ---------------------------------------------------------------------------
    # Write predictions to database
    logger.info("Writing to database")

    with db_connection.get_session() as session, session.no_autoflush:
        for forecaster in forecasters.values():
            forecaster.log_forecast_to_database(session=session)

    logger.info("Finished forecast")

    if raise_model_failure in ["any", "critical"]:
        check_model_runs_finished(
            completed_forecasts=list(forecasters.keys()),
            model_configs=model_configs,
            raise_if_missing=raise_model_failure,
        )


if __name__ == "__main__":
    typer.run(app)
