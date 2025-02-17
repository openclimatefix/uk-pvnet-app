"""App to run inference for PVNet models."""

import logging
import os
import tempfile
import warnings
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version

import dask
import fsspec
import pandas as pd
import sentry_sdk
import torch
import typer
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from ocf_datapipes.batch import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.config import get_union_of_configs, save_yaml_config
from pvnet_app.data.nwp import download_all_nwp_data, preprocess_nwp_data
from pvnet_app.data.satellite import (
    check_model_satellite_inputs_available,
    download_all_sat_data,
    preprocess_sat_data,
)
from pvnet_app.dataloader import get_dataloader, get_legacy_dataloader
from pvnet_app.forecast_compiler import ForecastCompiler
from pvnet_app.model_configs.pydantic_models import get_all_models

try:
    __version__ = version("pvnet-app")
except PackageNotFoundError:
    __version__ = "v?"

# sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"), environment=os.getenv("ENVIRONMENT", "local"), traces_sample_rate=1,
)

sentry_sdk.set_tag("app_name", "pvnet_app")
sentry_sdk.set_tag("version", __version__)

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Forecast made for these GSP IDs and summed to national with ID=>0
all_gsp_ids = list(range(1, 318))

# Batch size used to make forecasts for all GSPs
batch_size = 10

# ---------------------------------------------------------------------------
# LOGGER


class SQLAlchemyFilter(logging.Filter):
    def filter(self, record):
        return "sqlalchemy" not in record.pathname


# Create a logger
logging.basicConfig(
    level=getattr(logging, os.getenv("LOGLEVEL", "DEBUG")),
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger()

# Get rid of the verbose sqlalchemy logs
logging.getLogger("sqlalchemy").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# APP MAIN

def save_batch_to_s3(batch, model_name, s3_directory):
    """Saves a batch to a local file and uploads it to S3.

    Args:
        batch: The data batch to save (torch.Tensor).
        model_name: The name of the model (str).
        s3_directory: The S3 directory to save the batch to (str).
    """
    save_batch = f"{model_name}_latest_batch.pt"
    torch.save(batch,save_batch)

    try:
        fs = fsspec.open(s3_directory).fs
        fs.put(save_batch, f"{s3_directory}/{save_batch}")
        logger.info(
            f"Saved first batch for model {model_name} to {s3_directory}/{save_batch}",
            )
        os.remove(save_batch)
        logger.info("Removed local copy of batch")
    except Exception as e:
        logger.error(
            f"Failed to save batch to {s3_directory}/{save_batch} with error {e}",
            )


def app(
    t0=None,
    gsp_ids: list[int] = all_gsp_ids,
    write_predictions: bool = True,
    num_workers: int = -1,
):
    """Inference function for production

    This app expects these environmental variables to be available:
        - DB_URL
        - NWP_UKV_ZARR_PATH
        - NWP_ECMWF_ZARR_PATH
        - SATELLITE_ZARR_PATH
    The following are options
        - PVNET_V2_VERSION, pvnet version, default is a version above
        - PVNET_V2_SUMMATION_VERSION, the pvnet version, default is above
        - USE_ADJUSTER, option to use adjuster, defaults to true
        - SAVE_GSP_SUM, option to save gsp sum for pvnet_v2, defaults to false
        - RUN_EXTRA_MODELS, option to run extra models, defaults to false
        - DAY_AHEAD_MODEL, option to use day ahead model, defaults to false
        - SENTRY_DSN, optional link to sentry
        - ENVIRONMENT, the environment this is running in, defaults to local
        - USE_ECMWF_ONLY, option to use ecmwf only model, defaults to false
        - USE_OCF_DATA_SAMPLER, option to use ocf_data_sampler, defaults to true

    Args:
        t0 (datetime): Datetime at which forecast is made
        gsp_ids (array_like): List of gsp_ids to make predictions for. This list of GSPs are summed
            to national.
        write_predictions (bool): Whether to write prediction to the database. Else returns as
            DataArray for local testing.
        num_workers (int): Number of workers to use to load batches of data. When set to default
            value of -1, it will use one less than the number of CPU cores workers.
    """
    if num_workers == -1:
        num_workers = os.cpu_count() - 1
    if num_workers > 0:
        # Without this line the dataloader will hang if multiple workers are used
        dask.config.set(scheduler="single-threaded")

    use_day_ahead_model = os.getenv("DAY_AHEAD_MODEL", "false").lower() == "true"
    use_ecmwf_only = os.getenv("USE_ECMWF_ONLY", "false").lower() == "true"
    run_extra_models = os.getenv("RUN_EXTRA_MODELS", "false").lower() == "true"
    use_ocf_data_sampler = os.getenv("USE_OCF_DATA_SAMPLER", "true").lower() == "true"

    logger.info(f"Using `pvnet` library version: {__version__}")
    logger.info(f"Using `pvnet_app` library version: {__version__}")
    logger.info(f"Using {num_workers} workers")
    logger.info(f"Using day ahead model: {use_day_ahead_model}")
    logger.info(f"Using ecmwf only: {use_ecmwf_only}")
    logger.info(f"Running extra models: {run_extra_models}")

    # load models
    model_configs = get_all_models(
        get_ecmwf_only=use_ecmwf_only,
        get_day_ahead_only=use_day_ahead_model,
        run_extra_models=run_extra_models,
        use_ocf_data_sampler=use_ocf_data_sampler,
    )

    logger.info(f"Using adjuster: {model_configs[0].use_adjuster}")
    logger.info(f"Saving GSP sum: {model_configs[0].save_gsp_sum}")

    temp_dir = tempfile.TemporaryDirectory()

    # ---------------------------------------------------------------------------
    # 0. If inference datetime is None, round down to last 30 minutes
    if t0 is None:
        t0 = pd.Timestamp.now(tz="UTC").replace(tzinfo=None).floor(timedelta(minutes=30))
    else:
        t0 = pd.to_datetime(t0).floor(timedelta(minutes=30))

    if len(gsp_ids) == 0:
        gsp_ids = all_gsp_ids

    logger.info(f"Making forecast for init time: {t0}")
    logger.info(f"Making forecast for GSP IDs: {gsp_ids}")

    # ---------------------------------------------------------------------------
    # 1. Prepare data sources

    # Get capacities from the database
    logger.info("Loading capacities from the database")

    db_connection = DatabaseConnection(url=os.getenv("DB_URL"), base=Base_Forecast, echo=False)
    with db_connection.get_session() as session:
        #  Pandas series of most recent GSP capacities
        gsp_capacities = get_latest_gsp_capacities(
            session=session, gsp_ids=gsp_ids, datetime_utc=t0 - timedelta(days=2),
        )

        # National capacity is needed if using summation model
        national_capacity = get_latest_gsp_capacities(session, [0])[0]

    # Download satellite data
    logger.info("Downloading satellite data")
    sat_available = download_all_sat_data()

    # Preprocess the satellite data if available and store available timesteps
    if not sat_available:
        sat_datetimes = pd.DatetimeIndex([])
    else:
        sat_datetimes = preprocess_sat_data(t0, use_legacy=not use_ocf_data_sampler)

    # Download NWP data
    logger.info("Downloading NWP data")
    download_all_nwp_data(download_ukv=not use_ecmwf_only)

    # Preprocess the NWP data
    preprocess_nwp_data(use_ukv=not use_ecmwf_only)

    # ---------------------------------------------------------------------------
    # 2. Set up models

    # Prepare all the models which can be run
    forecast_compilers = {}
    data_config_paths = []
    for model_config in model_configs:
        # First load the data config
        data_config_path = PVNetBaseModel.get_data_config(
            model_config.pvnet.repo,
            revision=model_config.pvnet.version,
        )

        # Check if the data available will allow the model to run
        model_can_run = check_model_satellite_inputs_available(data_config_path, t0, sat_datetimes)

        if model_can_run:
            # Set up a forecast compiler for the model
            forecast_compilers[model_config.name] = ForecastCompiler(
                model_config=model_config,
                device=device,
                t0=t0,
                gsp_capacities=gsp_capacities,
                national_capacity=national_capacity,
                use_legacy=not use_ocf_data_sampler,
            )

            # Store the config filename so we can create batches suitable for all models
            data_config_paths.append(data_config_path)
        else:
            warnings.warn(f"The model {model_config.name} cannot be run with input data available")

    if len(forecast_compilers) == 0:
        raise Exception("No models were compatible with the available input data.")

    # Find the config with values suitable for running all models
    common_config = get_union_of_configs(data_config_paths)

    # Save the commmon config
    common_config_path = f"{temp_dir.name}/common_config_path.yaml"
    save_yaml_config(common_config, common_config_path)

    # ---------------------------------------------------------------------------
    # Set up data loader
    logger.info("Creating DataLoader")

    if not use_ocf_data_sampler:
        logger.info("Making OCF datapipes dataloader")
        # The current day ahead model uses the legacy dataloader
        dataloader = get_legacy_dataloader(
            config_filename=common_config_path,
            t0=t0,
            gsp_ids=gsp_ids,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    else:
        logger.info("Making OCF Data Sampler dataloader")
        dataloader = get_dataloader(
            config_filename=common_config_path,
            t0=t0,
            gsp_ids=gsp_ids,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    # ---------------------------------------------------------------------------
    # Make predictions
    logger.info("Processing batches")

    s3_directory = os.getenv("SAVE_BATCHES_DIR", None)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.info(f"Predicting for batch: {i}")

            if s3_directory and i == 0:
                model_name = list(forecast_compilers.keys())[0]
                
                save_batch_to_s3(batch, model_name, s3_directory) #Replaced with this function call

            for forecast_compiler in forecast_compilers.values():
                # need to do copy the batch for each model, as a model might change the batch
                device_batch = copy_batch_to_device(batch_to_tensor(batch), device)
                forecast_compiler.predict_batch(device_batch)

    # ---------------------------------------------------------------------------
    # Merge batch results to xarray DataArray
    logger.info("Processing raw predictions to DataArray")

    for forecast_compiler in forecast_compilers.values():
        forecast_compiler.compile_forecasts()

    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        temp_dir.cleanup()
        if not use_day_ahead_model:
            return forecast_compilers["pvnet_v2"].da_abs_all
        return forecast_compilers["pvnet_day_ahead"].da_abs_all

    # ---------------------------------------------------------------------------
    # Write predictions to database
    logger.info("Writing to database")

    with db_connection.get_session() as session:
        for forecast_compiler in forecast_compilers.values():
            forecast_compiler.log_forecast_to_database(session=session)

    temp_dir.cleanup()
    logger.info("Finished forecast")

def main():
    typer.run(app)

if __name__ == "__main__":
    main()
