"""App to run inference for PVNet multiple models."""

import logging
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version

import dask
import pandas as pd
import sentry_sdk
import torch
import typer
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from pvnet.models.base_model import BaseModel as PVNetBaseModel

from pvnet_app.utils import get_boolean_env_var, save_batch_to_s3, check_model_runs_finished
from pvnet_app.config import get_nwp_channels, get_union_of_configs, save_yaml_config
from pvnet_app.model_configs.pydantic_models import get_all_models
from pvnet_app.data.satellite import SatelliteDownloader
from pvnet_app.data.nwp import UKVDownloader, ECMWFDownloader
from pvnet_app.data.gsp import get_gsp_and_national_capacities
from pvnet_app.data.batch_validation import check_batch
from pvnet_app.dataloader import get_dataloader
from pvnet_app.forecast_compiler import ForecastCompiler
from pvnet_app.validate_forecast import validate_forecast
from pvnet_app.consts import __version__


try:
    __pvnet_version__ = version("pvnet")
except PackageNotFoundError:
    __pvnet_version__ = "v?"

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

# Forecast made for these GSP IDs and summed to national with ID=0
all_gsp_ids = list(range(1, 318))

# Batch size used to make forecasts for all GSPs
batch_size = 10

# ---------------------------------------------------------------------------
# APP MAIN


def app(
    t0: None | pd.Timestamp = None,
    gsp_ids: list[int] = all_gsp_ids,
    write_predictions: bool = True,
    num_workers: int = -1,
):
    """Inference function to run PVNet.

    Args:
        t0 (datetime): Datetime at which forecast is made
        gsp_ids (array_like): List of gsp_ids to make predictions for. This list of GSPs are summed
            to national.
        write_predictions (bool): Whether to write prediction to the database. Else returns as
            DataArray for local testing.
        num_workers (int): Number of workers to use to load batches of data. When set to default
            value of -1, it will use one less than the number of CPU cores workers.

    This app requires these environmental variables to be available:
        - DB_URL
    These variables are optional depending on the models being run:
        - NWP_UKV_ZARR_PATH
        - NWP_ECMWF_ZARR_PATH
        - SATELLITE_ZARR_PATH
    The following are optional:
        - SENTRY_DSN, optional link to sentry
        - ENVIRONMENT, the environment this is running in, defaults to local
        - ALLOW_ADJUSTER: Option to allow the adjuster to be used. If false this overwrites the 
          adjuster option in the model configs so it is not used. Defaults to true.
        - ALLOW_SAVE_GSP_SUM: Option to allow model to save the GSP sum. If false this overwrites 
          the model configs so saving of the GSP sum is not used. Defaults to false.
        - DAY_AHEAD_MODEL, option to use day ahead model, defaults to false
        - RUN_CRITICAL_MODELS_ONLY, option to run critical models only, defaults to false
        - USE_OCF_DATA_SAMPLER, option to use ocf_data_sampler, defaults to true
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
        t0 = t0.floor("30min")

    assert len(gsp_ids)>0, "No GSP IDs provided"

    if num_workers == -1:
        num_workers = os.cpu_count() - 1
    if num_workers > 0:
        # Without this line the dataloader will hang if multiple workers are used
        dask.config.set(scheduler="single-threaded")

    # --- Unpack the environment variables
    use_day_ahead_model = get_boolean_env_var("DAY_AHEAD_MODEL", default=False)
    run_critical_models_only = get_boolean_env_var("RUN_CRITICAL_MODELS_ONLY", default=False)
    use_ocf_data_sampler = get_boolean_env_var("USE_OCF_DATA_SAMPLER", default=True)
    allow_adjuster = get_boolean_env_var("ALLOW_ADJUSTER", default=True)
    allow_save_gsp_sum = get_boolean_env_var("ALLOW_SAVE_GSP_SUM", default=False)
    filter_bad_forecasts = get_boolean_env_var("FILTER_BAD_FORECASTS", default=False)
    raise_model_failure = os.getenv("RAISE_MODEL_FAILURE", None)

    zig_zag_warning_threshold = float(os.getenv('FORECAST_VALIDATE_ZIG_ZAG_WARNING', 250))
    zig_zag_error_threshold = float(os.getenv('FORECAST_VALIDATE_ZIG_ZAG_ERROR', 500))
    sun_elevation_lower_limit = float(os.getenv('FORECAST_VALIDATE_SUN_ELEVATION_LOWER_LIMIT', 10))
    
    db_url = os.environ["DB_URL"] # Will raise KeyError if not set
    s3_batch_save_dir = os.getenv("SAVE_BATCHES_DIR", None)
    ecmwf_source_path = os.getenv("NWP_ECMWF_ZARR_PATH", None)
    ukv_source_path = os.getenv("NWP_UKV_ZARR_PATH", None)
    sat_source_path_5 = os.getenv("SATELLITE_ZARR_PATH", None)
    sat_source_path_15 = (
        None if (sat_source_path_5 is None) else sat_source_path_5.replace(".zarr", "_15.zarr")
    )
    
    # --- Log version and variables
    logger.info(f"Using `pvnet` library version: {__pvnet_version__}")
    logger.info(f"Using `pvnet_app` library version: {__version__}")
    logger.info(f"Making forecast for init time: {t0}")
    logger.info(f"Making forecast for GSP IDs: {gsp_ids}")
    logger.info(f"Using {num_workers} workers")
    logger.info(f"Using day ahead model: {use_day_ahead_model}")
    logger.info(f"Running critical models only: {run_critical_models_only}")
    logger.info(f"Allow adjuster: {allow_adjuster}")
    logger.info(f"Allow saving GSP sum: {allow_save_gsp_sum}")

    # --- Get the model configurations
    model_configs = get_all_models(
        allow_adjuster=allow_adjuster,
        allow_save_gsp_sum=allow_save_gsp_sum,
        get_critical_only=run_critical_models_only,
        get_day_ahead_only=use_day_ahead_model,
        use_ocf_data_sampler=use_ocf_data_sampler,
    )

    if len(model_configs)==0:
        raise Exception("No models found after filtering")

    # Open connection to the database - used for pulling GSP capacitites and writing forecasts
    db_connection = DatabaseConnection(url=db_url, base=Base_Forecast, echo=False)

    temp_dir = tempfile.TemporaryDirectory()

    # 0. Get Model configs
    data_config_from_model = {}
    for model_config in model_configs:
        # First load the data config
        data_config_path = PVNetBaseModel.get_data_config(
            model_config.pvnet.repo,
            revision=model_config.pvnet.commit,
        )
        data_config_from_model[model_config.name] = data_config_path
    common_all_config = get_union_of_configs(data_config_from_model.values())

    # ---------------------------------------------------------------------------
    # 1. Prepare data sources

    # --- Get capacities from the database
    logger.info("Loading capacities from the database")
    gsp_capacities, national_capacity = get_gsp_and_national_capacities(
        db_connection=db_connection,
        gsp_ids=gsp_ids,
        t0=t0,
    )

    # --- Download satellite data
    logger.info("Downloading satellite data")
    
    sat_downloader = SatelliteDownloader(
        t0=t0,
        source_path_5=sat_source_path_5,
        source_path_15=sat_source_path_15,
        legacy=(not use_ocf_data_sampler), 
    )
    sat_downloader.run()

    # --- Download and process NWP data
    logger.info("Downloading NWP data")

    ukv_variables = get_nwp_channels(provider="ukv", nwp_config=common_all_config)
    ukv_downloader = UKVDownloader(source_path=ukv_source_path, nwp_variables=ukv_variables)
    ukv_downloader.run()

    ecmwf_variables = get_nwp_channels(provider="ecmwf", nwp_config=common_all_config)
    ecmwf_downloader = ECMWFDownloader(
        source_path=ecmwf_source_path,
        nwp_variables=ecmwf_variables,
        regrid_data=False if use_day_ahead_model else True,
    )
    ecmwf_downloader.run()

    # ---------------------------------------------------------------------------
    # 2. Set up models

    # Prepare all the models which can be run
    forecast_compilers = {}
    used_data_config_paths = []
    for model_config in model_configs:

        # First load the data config
        data_config_path = data_config_from_model[model_config.name]

        # Check if the data available will allow the model to run
        logger.info(f"Checking that the input data for model '{model_config.name}' exists")
        model_can_run = (
            sat_downloader.check_model_inputs_available(data_config_path, t0)
            and ecmwf_downloader.check_model_inputs_available(data_config_path, t0)
            and ukv_downloader.check_model_inputs_available(data_config_path, t0)
        )
        
        if model_can_run:
            logger.info(f"The input data for model '{model_config.name}' is available")
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
            used_data_config_paths.append(data_config_path)
        else:
            logger.warning(f"The model {model_config.name} cannot be run with input data available")

    if len(forecast_compilers) == 0:
        raise Exception("No models were compatible with the available input data.")

    # Find the config with values suitable for running all models
    common_config = get_union_of_configs(used_data_config_paths)

    # Save the commmon config
    common_config_path = f"{temp_dir.name}/common_config_path.yaml"
    save_yaml_config(common_config, common_config_path)

    # ---------------------------------------------------------------------------
    # Set up data loader
    logger.info("Creating DataLoader")

    dataloader = get_dataloader(
        config_filename=common_config_path,
        t0=t0,
        gsp_ids=gsp_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        db_url=db_url,
        use_data_sampler=use_ocf_data_sampler,
    )

    # ---------------------------------------------------------------------------
    # Make predictions
    logger.info("Processing batches")

    for i, batch in enumerate(dataloader):
        logger.info(f"Predicting for batch: {i}")

        # Do basic validation of the batch: Will raise error if the batch passes the checks
        check_batch(batch)

        if (s3_batch_save_dir is not None) and i == 0:
            # Save the batch under the name of the first model
            model_name = next(iter(forecast_compilers))
            save_batch_to_s3(batch, model_name, s3_batch_save_dir) 

        for forecast_compiler in forecast_compilers.values():
            forecast_compiler.predict_batch(batch)

    # Delete the downloaded data
    sat_downloader.clean_up()
    ecmwf_downloader.clean_up()
    ukv_downloader.clean_up()

    # ---------------------------------------------------------------------------
    # Merge batch results to xarray DataArray and make national forecast
    logger.info("Processing raw predictions to DataArray")

    for forecast_compiler in forecast_compilers.values():
        forecast_compiler.compile_forecasts()

    # ---------------------------------------------------------------------------
    # Run validation checks on the forecast values
    logger.info("Validating forecasts")
    for k in list(forecast_compilers.keys()):

        natioanl_forecast = (
            forecast_compilers[k].da_abs_all.sel(gsp_id=0, output_label="forecast_mw")
        ).to_series()

        forecast_okay = validate_forecast(
            national_forecast=natioanl_forecast,
            national_capacity=national_capacity,
            zip_zag_warning_threshold=zig_zag_warning_threshold,
            zig_zag_error_threshold=zig_zag_error_threshold,
            sun_elevation_lower_limit=sun_elevation_lower_limit,
            model_name=k,
        )

        if not forecast_okay:
            logger.warning(f"Forecast for model {k} failed validation")
            if filter_bad_forecasts:
                # This forecast will not be saved
                del forecast_compilers[k]

    if len(forecast_compilers) == 0:
        raise Exception("No models passed the forecast validation checks")


    # ---------------------------------------------------------------------------
    # Escape clause for making predictions locally
    if not write_predictions:
        return next(iter(forecast_compilers.values())).da_abs_all

    # ---------------------------------------------------------------------------
    # Write predictions to database
    logger.info("Writing to database")

    with db_connection.get_session() as session:
        with session.no_autoflush:
            for forecast_compiler in forecast_compilers.values():
                forecast_compiler.log_forecast_to_database(session=session)

    logger.info("Finished forecast")

    if raise_model_failure in ["any", "critical"]:
        check_model_runs_finished(
            completed_forecasts=list(forecast_compilers.keys()),
            model_configs=model_configs, 
            raise_if_missing=raise_model_failure,
        )

if __name__ == "__main__":
    typer.run(app)
