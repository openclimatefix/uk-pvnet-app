"""App to run inference for PVNet models
"""

import logging
import os
import tempfile
import warnings
from datetime import timedelta

import dask
import pandas as pd
import pvnet
import torch
import typer
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from ocf_datapipes.batch import batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel
import sentry_sdk


import pvnet_app
from pvnet_app.config import get_union_of_configs, load_yaml_config, save_yaml_config
from pvnet_app.data.nwp import download_all_nwp_data, preprocess_nwp_data
from pvnet_app.data.satellite import (
    download_all_sat_data,
    preprocess_sat_data,
    check_model_satellite_inputs_available,
)
from pvnet_app.dataloader import get_legacy_dataloader, get_dataloader
from pvnet_app.forecast_compiler import ForecastCompiler


# sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),
    environment=f'{os.getenv("ENVIRONMENT", "local")}',
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
)
sentry_sdk.set_tag("app_name", "pvnet_app")

# ---------------------------------------------------------------------------
# GLOBAL SETTINGS

# Model will use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Forecast made for these GSP IDs and summed to national with ID=>0
all_gsp_ids = list(range(1, 318))

# Batch size used to make forecasts for all GSPs
batch_size = 10

# Dictionary of all models to run
# - The dictionary key will be used as the model name when saving to the database
# - The key "pvnet_v2" must be included
# - Batches are prepared only once, so the extra models must be able to run on the batches created
#   to run the pvnet_v2 model
models_dict = {
    
    "pvnet_v2": {
        # Huggingfacehub model repo and commit for PVNet (GSP-level model)
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region",
            "version": os.getenv('PVNET_V2_VERSION', "ae0b8006841ac6227db873a1fc7f7331dc7dadb5"),
            # We should only set PVNET_V2_VERSION in a short term solution,
            # as its difficult to track which model is being used
        },
        # Huggingfacehub model repo and commit for PVNet summation (GSP sum to national model)
        # If summation_model_name is set to None, a simple sum is computed instead
        "summation": {
            "name": "openclimatefix/pvnet_v2_summation",
            "version": os.getenv(
                'PVNET_V2_SUMMATION_VERSION',
                "ffac655f9650b81865d96023baa15839f3ce26ec"
            ),
        },
        # Whether to use the adjuster for this model - for pvnet_v2 is set by environmental variable
        "use_adjuster": os.getenv("USE_ADJUSTER", "true").lower() == "true",
        # Whether to save the GSP sum for this model - for pvnet_v2 is set by environmental variable
        "save_gsp_sum": os.getenv("SAVE_GSP_SUM", "false").lower() == "true",
        # Whether to log information through prediction steps for this model
        "verbose": True,
        "save_gsp_to_forecast_value_last_seven_days": True,
    },
    
    # Extra models which will be run on dev only
    "pvnet_v2-sat0-samples-v1": {
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region",
            "version": "8a7cc21b64d25ce1add7a8547674be3143b2e650",
        },
        "summation": {
            "name": "openclimatefix/pvnet_v2_summation",
            "version": "dcfdc17fda8e48c387122614bec8b284eaa868b9",
        },
        "use_adjuster": False,
        "save_gsp_sum": False,
        "verbose": False,
        "save_gsp_to_forecast_value_last_seven_days": False,
    },
    
    # single source models
    "pvnet_v2-sat0-only-samples-v1": {
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region",
            "version": "d7ab648942c85b6788adcdbed44c91c4e1c5604a",
        },
        "summation": {
            "name": "openclimatefix/pvnet_v2_summation",
            "version": "adbf9e7797fee9a5050beb8c13841696e72f99ef",
        },
        "use_adjuster": False,
        "save_gsp_sum": False,
        "verbose": False,
        "save_gsp_to_forecast_value_last_seven_days": False,
    },
    
    "pvnet_v2-ukv-only-samples-v1": {
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region",
            "version": "eb73bf9a176a108f2e33b809f1f6993f893a4df9",
        },
        "summation": {
            "name": "openclimatefix/pvnet_v2_summation",
            "version": "9002baf1e9dc1ec141f3c4a1fa8447b6316a4558",
        },
        "use_adjuster": False,
        "save_gsp_sum": False,
        "verbose": False,
        "save_gsp_to_forecast_value_last_seven_days": False,
    },
    
    "pvnet_v2-ecmwf-only-samples-v1": {
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region",
            "version": "0bc344fafb2232fb0b6bb0bf419f0449fe11c643",
        },
        "summation": {
            "name": "openclimatefix/pvnet_v2_summation",
            "version": "4fe6b1441b6dd549292c201ed85eee156ecc220c",
        },
        "use_adjuster": False,
        "save_gsp_sum": False,
        "verbose": False,
        "save_gsp_to_forecast_value_last_seven_days": False,
    },
}

# The day ahead model has not yet been re-trained with data-sampler. 
# It will be run with the legacy dataloader using ocf_datapipes
day_ahead_model_dict = {
    "pvnet_day_ahead": {
        # Huggingfacehub model repo and commit for PVNet day ahead models
        "pvnet": {
            "name": "openclimatefix/pvnet_uk_region_day_ahead",
            "version": "d87565731692a6003e43caac4feaed0f69e79272",
        },
        "summation": {
            "name": "openclimatefix/pvnet_summation_uk_national_day_ahead",
            "version": "ed60c5d32a020242ca4739dcc6dbc8864f783a08",
        },
        "use_adjuster": True,
        "save_gsp_sum": True,
        "verbose": True,
        "save_gsp_to_forecast_value_last_seven_days": True,
    },
}

# ---------------------------------------------------------------------------
# LOGGER


class SQLAlchemyFilter(logging.Filter):
    def filter(self, record):
        return "sqlalchemy" not in record.pathname


# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s"
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Get rid of the verbose sqlalchemy logs
stream_handler.addFilter(SQLAlchemyFilter())
sql_logger = logging.getLogger("sqlalchemy.engine.Engine")
sql_logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# APP MAIN


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
    
    logger.info(f"Using `pvnet` library version: {pvnet.__version__}")
    logger.info(f"Using `pvnet_app` library version: {pvnet_app.__version__}")
    logger.info(f"Using {num_workers} workers")
    logger.info(f"Using day ahead model: {use_day_ahead_model}")

    # Filter the models to be run
    if use_day_ahead_model:
        model_to_run_dict = day_ahead_model_dict
        main_model_key = "pvnet_day_ahead"
    else:

        if os.getenv("RUN_EXTRA_MODELS", "false").lower() == "false":
            model_to_run_dict = {"pvnet_v2": models_dict["pvnet_v2"]}
        else:
            model_to_run_dict = models_dict
        main_model_key = "pvnet_v2"

    logger.info(f"Using adjduster: {model_to_run_dict[main_model_key]['use_adjuster']}")
    logger.info(f"Saving GSP sum: {model_to_run_dict[main_model_key]['save_gsp_sum']}")

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
            session=session, gsp_ids=gsp_ids, datetime_utc=t0-timedelta(days=2)
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
        sat_datetimes = preprocess_sat_data(t0, use_legacy=use_day_ahead_model)

    # Download NWP data
    logger.info("Downloading NWP data")
    download_all_nwp_data()

    # Preprocess the NWP data
    preprocess_nwp_data()

    # ---------------------------------------------------------------------------
    # 2. Set up models

    # Prepare all the models which can be run
    forecast_compilers = {}
    data_config_paths = []
    for model_key, model_config in model_to_run_dict.items():
        # First load the data config
        data_config_path = PVNetBaseModel.get_data_config(
            model_config["pvnet"]["name"],
            revision=model_config["pvnet"]["version"],
        )

        # Check if the data available will allow the model to run
        model_can_run = check_model_satellite_inputs_available(data_config_path, t0, sat_datetimes)

        if model_can_run:
            # Set up a forecast compiler for the model
            forecast_compilers[model_key] = ForecastCompiler(
                model_tag=model_key,
                model_name=model_config["pvnet"]["name"],
                model_version=model_config["pvnet"]["version"],
                summation_name=model_config["summation"]["name"],
                summation_version=model_config["summation"]["version"],
                device=device,
                t0=t0,
                gsp_capacities=gsp_capacities,
                national_capacity=national_capacity,
                apply_adjuster=model_config["use_adjuster"],
                save_gsp_sum=model_config["save_gsp_sum"],
                save_gsp_to_recent=model_config["save_gsp_to_forecast_value_last_seven_days"],
                verbose=model_config["verbose"],
                use_legacy=use_day_ahead_model,
            )

            # Store the config filename so we can create batches suitable for all models
            data_config_paths.append(data_config_path)
        else:
            warnings.warn(f"The model {model_key} cannot be run with input data available")

    if len(forecast_compilers) == 0:
        raise Exception(f"No models were compatible with the available input data.")

    # Find the config with values suitable for running all models
    common_config = get_union_of_configs(data_config_paths)

    # Save the commmon config
    common_config_path = f"{temp_dir.name}/common_config_path.yaml"
    save_yaml_config(common_config, common_config_path)

    # ---------------------------------------------------------------------------
    # Set up data loader
    logger.info("Creating DataLoader")

    if use_day_ahead_model:
        # The current day ahead model uses the legacy dataloader
        dataloader = get_legacy_dataloader(
            config_filename=common_config_path, 
            t0=t0, 
            gsp_ids=gsp_ids,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    
    else:
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

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            logger.info(f"Predicting for batch: {i}")

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


if __name__ == "__main__":
    typer.run(app)
