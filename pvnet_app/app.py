"""App to run inference

This app expects these evironmental variables to be available:
    - DB_URL
    - NWP_UKV_ZARR_PATH
    - NWP_ECMWF_ZARR_PATH
    - SATELLITE_ZARR_PATH
    - RUN_EXTRA_MODELS
    - USE_ADJUSTER
    - SAVE_GSP_SUM
"""

import logging
import os
import tempfile
import warnings
from datetime import timedelta
from pathlib import Path

import dask
import pandas as pd
import pvnet
import torch
import typer
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from ocf_data_sampler.torch_datasets.pvnet_uk_regional import PVNetUKRegionalDataset
from ocf_datapipes.batch import stack_np_examples_into_batch, batch_to_tensor, copy_batch_to_device
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from pvnet.utils import GSPLocationLookup
from torch.utils.data import DataLoader
import sentry_sdk



import pvnet_app
from pvnet_app.data.nwp import (
    download_all_nwp_data,
    preprocess_nwp_data,
)
from pvnet_app.data.satellite import (
    download_all_sat_data,
    preprocess_sat_data,
    check_model_inputs_available,
)
from pvnet_app.forecast_compiler import ForecastCompiler
from pvnet_app.utils import (
    populate_data_config_sources,
    convert_dataarray_to_forecasts,
    find_min_satellite_delay_config,
    save_yaml_config,
)

# Legacy imports 
from ocf_datapipes.load import OpenGSPFromDatabase
from torch.utils.data.datapipes.iter import IterableWrapper
from ocf_datapipes.training.pvnet import construct_sliced_data_pipeline
from ocf_datapipes.batch import BatchKey

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
            "version": os.getenv('PVNET_V2_SUMMATION_VERSION',
                                 "ffac655f9650b81865d96023baa15839f3ce26ec"),
        },
        # Whether to use the adjuster for this model - for pvnet_v2 is set by environmental variable
        "use_adjuster": os.getenv("USE_ADJUSTER", "true").lower() == "true",
        # Whether to save the GSP sum for this model - for pvnet_v2 is set by environmental variable
        "save_gsp_sum": os.getenv("SAVE_GSP_SUM", "false").lower() == "true",
        # Where to log information through prediction steps for this model
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


def get_dataloader(config_filename: str, t0: pd.Timestamp, gsp_ids: list[int], num_workers: int):
    
    # Populate the data config with production data paths    
    populated_data_config_filename = Path(config_filename).parent / "data_config.yaml"
    
    populate_data_config_sources(config_filename, populated_data_config_filename)
    
    dataset = PVNetUKRegionalDataset(
        config_filename=populated_data_config_filename, 
        start_time=t0, 
        end_time=t0,
        gsp_ids=gsp_ids,
    )

    # Set up dataloader for parallel loading
    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=batch_size,
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=stack_np_examples_into_batch,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=False,
    )

    return DataLoader(dataset, **dataloader_kwargs)


def legacy_squeeze(batch):
    batch[BatchKey.gsp_id] = batch[BatchKey.gsp_id].squeeze(1)
    return batch


def get_legacy_dataloader(
    config_filename: str, 
    t0: pd.Timestamp,
    gsp_ids: list[int],
    num_workers: int,
):
    
    # Populate the data config with production data paths
    populated_data_config_filename = Path(config_filename).parent / "data_config.yaml"
    
    populate_data_config_sources(
        config_filename, 
        populated_data_config_filename,
        gsp_path=os.environ["DB_URL"],
    
    )
    
    # Set up ID location query object
    ds_gsp = next(iter(OpenGSPFromDatabase()))
    gsp_id_to_loc = GSPLocationLookup(ds_gsp.x_osgb, ds_gsp.y_osgb)
    
    # Location and time datapipes
    location_pipe = IterableWrapper([gsp_id_to_loc(gsp_id) for gsp_id in gsp_ids])
    t0_datapipe = IterableWrapper([t0]).repeat(len(location_pipe))

    location_pipe = location_pipe.sharding_filter()
    t0_datapipe = t0_datapipe.sharding_filter()

    # Batch datapipe
    batch_datapipe = (
        construct_sliced_data_pipeline(
            config_filename=populated_data_config_filename,
            location_pipe=location_pipe,
            t0_datapipe=t0_datapipe,
            production=True,
        )
        .batch(batch_size)
        .map(stack_np_examples_into_batch)
        .map(legacy_squeeze)
    )

    # Set up dataloader for parallel loading
    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,  # batched in datapipe step
        sampler=None,
        batch_sampler=None,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=None if num_workers == 0 else 2,
        persistent_workers=False,
    )

    return DataLoader(batch_datapipe, **dataloader_kwargs)



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
        - USE_SATELLITE, option to get satellite data. defaults to true
        - USE_ADJUSTER, option to use adjuster, defaults to true
        - SAVE_GSP_SUM, option to save gsp sum, defaults to false
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
    use_satellite = os.getenv("USE_SATELLITE", "true").lower() == "true"
    logger.info(f"Using satellite data: {use_satellite}")
    logger.info(f"Using day ahead model: {use_day_ahead_model}")

    if use_day_ahead_model:
        logger.info(f"Using day ahead PVNet model")

    logger.info(f"Using `pvnet` library version: {pvnet.__version__}")
    logger.info(f"Using `pvnet_app` library version: {pvnet_app.__version__}")
    logger.info(f"Using {num_workers} workers")

    if use_day_ahead_model:
        logger.info(f"Using adjduster: {day_ahead_model_dict['pvnet_day_ahead']['use_adjuster']}")
        logger.info(f"Saving GSP sum: {day_ahead_model_dict['pvnet_day_ahead']['save_gsp_sum']}")

    else:
        logger.info(f"Using adjduster: {models_dict['pvnet_v2']['use_adjuster']}")
        logger.info(f"Saving GSP sum: {models_dict['pvnet_v2']['save_gsp_sum']}")

    # Used for temporarily storing things
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

    logger.info("Loading GSP metadata")

    # Get capacities from the database
    db_connection = DatabaseConnection(url=os.getenv("DB_URL"), base=Base_Forecast, echo=False)
    with db_connection.get_session() as session:
        #  Pandas series of most recent GSP capacities
        gsp_capacities = get_latest_gsp_capacities(
            session=session, gsp_ids=gsp_ids, datetime_utc=t0-timedelta(days=2)
        )

        # National capacity is needed if using summation model
        national_capacity = get_latest_gsp_capacities(session, [0])[0]

    # Download satellite data
    if use_satellite:
        logger.info("Downloading satellite data")
        download_all_sat_data()

        # Preprocess the satellite data and record the delay of the most recent non-nan timestep
        all_satellite_datetimes, data_freq_minutes = preprocess_sat_data(t0)
    else:
        all_satellite_datetimes = []
        data_freq_minutes = None

    # Download NWP data
    logger.info("Downloading NWP data")
    download_all_nwp_data()

    # Preprocess the NWP data
    preprocess_nwp_data()

    # ---------------------------------------------------------------------------
    # 2. Set up models

    if use_day_ahead_model:
        model_to_run_dict = {"pvnet_day_ahead": day_ahead_model_dict["pvnet_day_ahead"]}
    # Remove extra models if not configured to run them
    elif os.getenv("RUN_EXTRA_MODELS", "false").lower() == "false":
        model_to_run_dict = {"pvnet_v2": models_dict["pvnet_v2"]}
    else:
        model_to_run_dict = models_dict

    # Prepare all the models which can be run
    forecast_compilers = {}
    data_config_filenames = []
    for model_name, model_config in model_to_run_dict.items():
        # First load the data config
        data_config_filename = PVNetBaseModel.get_data_config(
            model_config["pvnet"]["name"],
            revision=model_config["pvnet"]["version"],
        )

        # Check if the data available will allow the model to run
        model_can_run = check_model_inputs_available(
            data_config_filename, all_satellite_datetimes, t0, data_freq_minutes
        )

        if model_can_run:
            # Set up a forecast compiler for the model
            forecast_compilers[model_name] = ForecastCompiler(
                model_name=model_config["pvnet"]["name"],
                model_version=model_config["pvnet"]["version"],
                summation_name=model_config["summation"]["name"],
                summation_version=model_config["summation"]["version"],
                device=device,
                t0=t0,
                gsp_capacities=gsp_capacities,
                national_capacity=national_capacity,
                verbose=model_config["verbose"],
            )

            # Store the config filename so we can create batches suitable for all models
            data_config_filenames.append(data_config_filename)
        else:
            warnings.warn(f"The model {model_name} cannot be run with input data available")

    if len(forecast_compilers) == 0:
        raise Exception(f"No models were compatible with the available input data.")

    # Find the config with satellite delay suitable for all models running
    common_config = find_min_satellite_delay_config(data_config_filenames, use_satellite=use_satellite)

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
            num_workers=num_workers
        )
    
    else:
        dataloader = get_dataloader(
            config_filename=common_config_path, 
            t0=t0, 
            gsp_ids=gsp_ids,
            num_workers=num_workers
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
        for model_name, forecast_compiler in forecast_compilers.items():
            sql_forecasts = convert_dataarray_to_forecasts(
                forecast_compiler.da_abs_all,
                session,
                model_name=model_name,
                version=pvnet_app.__version__,
            )
            if model_to_run_dict[model_name]["save_gsp_to_forecast_value_last_seven_days"]:

                save_sql_forecasts(
                    forecasts=sql_forecasts,
                    session=session,
                    update_national=True,
                    update_gsp=True,
                    apply_adjuster=model_to_run_dict[model_name]["use_adjuster"],
                )
            else:
                # national
                save_sql_forecasts(
                    forecasts=sql_forecasts[0:1],
                    session=session,
                    update_national=True,
                    update_gsp=False,
                    apply_adjuster=model_to_run_dict[model_name]["use_adjuster"],
                )
                save_sql_forecasts(
                    forecasts=sql_forecasts[1:],
                    session=session,
                    update_national=False,
                    update_gsp=True,
                    apply_adjuster=model_to_run_dict[model_name]["use_adjuster"],
                    save_to_last_seven_days=False,
                )

            if model_to_run_dict[model_name]["save_gsp_sum"]:
                # Compute the sum if we are logging the sume of GSPs independently
                da_abs_sum_gsps = (
                    forecast_compiler.da_abs_all.sel(gsp_id=slice(1, 317))
                    .sum(dim="gsp_id")
                    # Only select the central forecast for the GSP sum. The sums of different p-levels
                    # are not a meaningful qauntities
                    .sel(output_label=["forecast_mw"])
                    .expand_dims(dim="gsp_id", axis=0)
                    .assign_coords(gsp_id=[0])
                )

                # Save the sum of GSPs independently - mainly for summation model monitoring
                sql_forecasts = convert_dataarray_to_forecasts(
                    da_abs_sum_gsps,
                    session,
                    model_name=f"{model_name}_gsp_sum",
                    version=pvnet_app.__version__,
                )

                save_sql_forecasts(
                    forecasts=sql_forecasts,
                    session=session,
                    update_national=True,
                    update_gsp=False,
                    apply_adjuster=False,
                )

        temp_dir.cleanup()
        logger.info("Finished forecast")


if __name__ == "__main__":
    typer.run(app)
