import tempfile
import zarr
import os

import numpy as np
import pandas as pd
import xarray as xr

from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)

from pvnet_app.consts import sat_path, nwp_ukv_path, nwp_ecmwf_path
from pvnet_app.data.satellite import sat_5_path, sat_15_path
from pvnet.models.base_model import BaseModel as PVNetBaseModel
from ocf_datapipes.config.load import load_yaml_configuration

from ocf_data_sampler.load.gsp import open_gsp

from pvnet_app.model_configs.pydantic_models import get_all_models


def test_app(
    db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, gsp_yields_and_systems, me_latest
):

    """Test the app running the intraday models"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables
        temp_nwp_path = "temp_nwp_ukv.zarr"
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path
        nwp_ukv_data.to_zarr(temp_nwp_path)

        temp_nwp_path = "temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        # In production sat zarr is zipped
        temp_sat_path = "temp_sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data.to_zarr(store)

        # Set environmental variables
        os.environ["RUN_EXTRA_MODELS"] = "True"
        os.environ["SAVE_GSP_SUM"] = "True"
        os.environ["DAY_AHEAD_MODEL"] = "False"

        # Run prediction
        # These imports need to come after the environ vars have been set
        from pvnet_app.app import app

        app(gsp_ids=list(range(1, 318)), num_workers=2)

    all_models = get_all_models(run_extra_models=True)

    # Check correct number of forecasts have been made
    # (317 GSPs + 1 National + maybe GSP-sum) = 318 or 319 forecasts
    # Forecast made with multiple models
    expected_forecast_results = 0
    for model_config in all_models:
        expected_forecast_results += 318 + model_config.save_gsp_sum

    forecasts = db_session.query(ForecastSQL).all()
    # Doubled for historic and forecast
    assert len(forecasts) == expected_forecast_results * 2

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    # 318 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == expected_forecast_results * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == expected_forecast_results * 16

    expected_forecast_results = 0
    for model_config in all_models:
        # National
        expected_forecast_results += 1
        # GSP
        expected_forecast_results += 317 * model_config.save_gsp_to_recent
        expected_forecast_results += model_config.save_gsp_sum  # optional Sum of GSPs

    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == expected_forecast_results * 16


def test_app_day_ahead_model(
    db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, gsp_yields_and_systems, me_latest
):
    """Test the app running the day ahead model"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        temp_nwp_path = "temp_nwp_ukv.zarr"
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path
        nwp_ukv_data.to_zarr(temp_nwp_path)

        temp_nwp_path = "temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        temp_sat_path = "temp_sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data.to_zarr(store)

        os.environ["DAY_AHEAD_MODEL"] = "True"
        os.environ["RUN_EXTRA_MODELS"] = "False"
        os.environ["USE_OCF_DATA_SAMPLER"] = "False"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app

        app(gsp_ids=list(range(1, 318)), num_workers=2)

    all_models = get_all_models(get_day_ahead_only=True, use_ocf_data_sampler=False)

    # Check correct number of forecasts have been made
    # (317 GSPs + 1 National + maybe GSP-sum) = 318 or 319 forecasts
    # Forecast made with multiple models
    expected_forecast_results = 0
    for model_config in all_models:
        expected_forecast_results += 318 + model_config.save_gsp_sum

    forecasts = db_session.query(ForecastSQL).all()
    # Doubled for historic and forecast
    assert len(forecasts) == expected_forecast_results * 2

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    # 72 time steps in forecast
    expected_forecast_timesteps = 72

    assert (
        len(db_session.query(ForecastValueSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )
    assert (
        len(db_session.query(ForecastValueLatestSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )
    assert (
        len(db_session.query(ForecastValueSevenDaysSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )


def test_app_no_sat(
    db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, gsp_yields_and_systems, me_latest
):
    """Test the app for the case when no satellite data is available"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        temp_nwp_path = "temp_nwp_ukv.zarr"
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path
        nwp_ukv_data.to_zarr(temp_nwp_path)

        temp_nwp_path = "temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        # There is no satellite data available at the environ path
        os.environ["SATELLITE_ZARR_PATH"] = "nonexistent_sat.zarr.zip"

        os.environ["RUN_EXTRA_MODELS"] = "True"
        os.environ["SAVE_GSP_SUM"] = "True"
        os.environ["DAY_AHEAD_MODEL"] = "False"
        os.environ["USE_OCF_DATA_SAMPLER"] = "True"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app

        app(gsp_ids=list(range(1, 318)), num_workers=2)

    # Only the models which don't use satellite will be run in this case
    # The models below are the only ones which should have been run
    all_models = get_all_models(run_extra_models=True)
    all_models = [model for model in all_models if not model.uses_satellite_data]

    # Check correct number of forecasts have been made
    # (317 GSPs + 1 National + maybe GSP-sum) = 318 or 319 forecasts
    # Forecast made with multiple models
    expected_forecast_results = 0
    for model_config in all_models:
        expected_forecast_results += 318 + model_config.save_gsp_sum

    forecasts = db_session.query(ForecastSQL).all()
    # Doubled for historic and forecast
    assert len(forecasts) == expected_forecast_results * 2

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    # 318 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == expected_forecast_results * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == expected_forecast_results * 16

    expected_forecast_results = 0
    for model_config in all_models:
        # National
        expected_forecast_results += 1
        # GSP
        expected_forecast_results += 317 * model_config.save_gsp_to_recent
        expected_forecast_results += model_config.save_gsp_sum  # optional Sum of GSPs

    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == expected_forecast_results * 16


# test legacy models
# Its nice to have this here, so we can run the latest version in production, but still use the old models
# Once we have re trained PVnet summation models we can remove this
def test_app_ecwmf_only(db_session, nwp_ecmwf_data, gsp_yields_and_systems, me_latest):
    """Test the app for the case running model just on ecmwf"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        temp_nwp_path = "temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        # There is no satellite or ukv data available at the environ path
        os.environ["SATELLITE_ZARR_PATH"] = "nonexistent_sat.zarr.zip"
        os.environ["NWP_UKV_ZARR_PATH"] = "nonexistent_nwp.zarr.zip"

        os.environ["RUN_EXTRA_MODELS"] = "False"
        os.environ["SAVE_GSP_SUM"] = "True"
        os.environ["DAY_AHEAD_MODEL"] = "False"
        os.environ["USE_OCF_DATA_SAMPLER"] = "False"
        os.environ["USE_ECMWF_ONLY"] = "True"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app

        app(gsp_ids=list(range(1, 318)), num_workers=2)

    # Only the models which don't use satellite will be run in this case
    # The models below are the only ones which should have been run
    all_models = get_all_models(get_ecmwf_only=True)

    # Check correct number of forecasts have been made
    # (317 GSPs + 1 National + maybe GSP-sum) = 318 or 319 forecasts
    # Forecast made with multiple models
    expected_forecast_results = 0
    for model_config in all_models:
        expected_forecast_results += 318 + model_config.save_gsp_sum

    forecasts = db_session.query(ForecastSQL).all()
    # Doubled for historic and forecast
    assert len(forecasts) == expected_forecast_results * 2

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    # 318 GSPs * 16 time steps in forecast
    assert len(db_session.query(ForecastValueSQL).all()) == expected_forecast_results * 16
    assert len(db_session.query(ForecastValueLatestSQL).all()) == expected_forecast_results * 16

    expected_forecast_results = 0
    for model_config in all_models:
        # National
        expected_forecast_results += 1
        # GSP
        expected_forecast_results += 317 * model_config.save_gsp_to_recent
        expected_forecast_results += model_config.save_gsp_sum  # optional Sum of GSPs

    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == expected_forecast_results * 16


# test legacy models
# Its nice to have this here, so we can run the latest version in production, but still use the old models
# Once we have re trained PVnet summation models we can remove this
def test_app_ocf_datapipes(
    db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, gsp_yields_and_systems, me_latest
):
    """Test the app running the day ahead model"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        temp_nwp_path = "temp_nwp_ukv.zarr"
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path
        nwp_ukv_data.to_zarr(temp_nwp_path)

        temp_nwp_path = "temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        temp_sat_path = "temp_sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data.to_zarr(store)

        os.environ["DAY_AHEAD_MODEL"] = "False"
        os.environ["RUN_EXTRA_MODELS"] = "False"
        os.environ["USE_OCF_DATA_SAMPLER"] = "False"
        os.environ["USE_ECMWF_ONLY"] = "False"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app

        app(gsp_ids=list(range(1, 318)), num_workers=2)

    all_models = get_all_models(use_ocf_data_sampler=False)

    # Check correct number of forecasts have been made
    # (317 GSPs + 1 National + maybe GSP-sum) = 318 or 319 forecasts
    # Forecast made with multiple models
    expected_forecast_results = 0
    for model_config in all_models:
        expected_forecast_results += 318 + model_config.save_gsp_sum

    forecasts = db_session.query(ForecastSQL).all()
    # Doubled for historic and forecast
    assert len(forecasts) == expected_forecast_results * 2

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    # 72 time steps in forecast
    expected_forecast_timesteps = 16

    assert (
        len(db_session.query(ForecastValueSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )
    assert (
        len(db_session.query(ForecastValueLatestSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )
    assert (
        len(db_session.query(ForecastValueSevenDaysSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )


# NEW TESTING - DA with data-sampler utilisation
def test_app_day_ahead_model_data_sampler(
    db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, gsp_yields_and_systems, me_latest
):
    """ Test app running day ahead model with OCF data sampler """
    import xarray as xr
    import zarr
    import numpy as np
    import pandas as pd
    import os
    import json

    from pvnet_app.config import ensure_zarr_metadata

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        print(f"Working in temporary directory: {tmpdirname}")

        # Save NWP data with explicit metadata
        def save_consolidated_zarr(dataset, path):
            store = zarr.DirectoryStore(path)
            zarr_group = zarr.group(store=store, overwrite=True)
            
            # Save variables
            for var_name, var_data in dataset.data_vars.items():
                zarr_group.create_dataset(var_name, data=var_data.values)
            
            # Manually create .zmetadata with all dataset metadata
            metadata = {
                "zarr_format": 2,
                "variables": {},
                "attributes": dict(dataset.attrs)
            }
            
            for var_name, var_data in dataset.data_vars.items():
                metadata["variables"][var_name] = dict(var_data.attrs)
            
            store['.zmetadata'] = json.dumps(metadata).encode('utf-8')

        # Save NWP data
        temp_nwp_ukv_path = "temp_nwp_ukv.zarr"
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_ukv_path
        ensure_zarr_metadata(temp_nwp_ukv_path)
        save_consolidated_zarr(nwp_ukv_data, temp_nwp_ukv_path)

        temp_nwp_ecmwf_path = "temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_ecmwf_path
        ensure_zarr_metadata(temp_nwp_ecmwf_path)
        save_consolidated_zarr(nwp_ecmwf_data, temp_nwp_ecmwf_path)

        # Save satellite data
        temp_sat_path = "temp_sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        ensure_zarr_metadata(temp_sat_path)

        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data.to_zarr(store)

        # Verify Zarr store contents
        def print_zarr_contents(path):
            print(f"\nContents of {path}:")
            try:
                print(os.listdir(path))
                store = zarr.DirectoryStore(path)
                
                # Try to load .zmetadata explicitly
                try:
                    metadata_str = store['.zmetadata']
                    metadata = json.loads(metadata_str.decode('utf-8'))
                    print("Metadata:")
                    print(json.dumps(metadata, indent=2))
                except Exception as e:
                    print(f"Error reading .zmetadata: {e}")
                
                zgroup = zarr.open_group(store=store)
                print("Zarr group keys:")
                print(list(zgroup.keys()))
            except Exception as e:
                print(f"Error examining Zarr store: {e}")

        print_zarr_contents(temp_nwp_ukv_path)
        print_zarr_contents(temp_nwp_ecmwf_path)

        # Debug: Print NWP configuration
        try:
            from pvnet_app.model_configs.pydantic_models import get_all_models
            
            print("\nModel Configurations:")
            all_models = get_all_models(get_day_ahead_only=True, use_ocf_data_sampler=True)
            
            for model_config in all_models:
                print(f"Model: {model_config}")
                if hasattr(model_config, 'input_data'):
                    print("NWP Configuration:")
                    nwp_config = model_config.input_data.get('nwp', {})
                    for source, config in nwp_config.items():
                        print(f"  {source}: {config}")
        except Exception as e:
            print(f"Error loading model configurations: {e}")

        # Attempt manual NWP loading
        try:
            from ocf_data_sampler.load.nwp.nwp import open_nwp
            
            print("\nTrying to open NWP data manually:")
            ukv_nwp = open_nwp(temp_nwp_ukv_path, provider='ukv')
            print("UKV NWP opened successfully")
            
            ecmwf_nwp = open_nwp(temp_nwp_ecmwf_path, provider='ecmwf')
            print("ECMWF NWP opened successfully")
        except Exception as e:
            print(f"Error opening NWP data: {e}")
            import traceback
            traceback.print_exc()

        # Set environment variables
        os.environ["DAY_AHEAD_MODEL"] = "True"
        os.environ["RUN_EXTRA_MODELS"] = "False"
        os.environ["USE_OCF_DATA_SAMPLER"] = "True"

        for key in ['NWP_UKV_ZARR_PATH', 'NWP_ECMWF_ZARR_PATH', 'SATELLITE_ZARR_PATH', 
                    'DAY_AHEAD_MODEL', 'RUN_EXTRA_MODELS', 'USE_OCF_DATA_SAMPLER']:
            print(f"{key}: {os.environ.get(key)}")

        print("\n--- Pre-App Debugging ---")
        # Additional detailed checks before app() call
        try:
            import xarray as xr
            print("\nAttempting to open UKV NWP Zarr:")
            ukv_ds = xr.open_zarr(temp_nwp_ukv_path)
            print("UKV NWP Zarr opened successfully")
            print("UKV NWP Dataset info:")
            print(ukv_ds)
        except Exception as e:
            print(f"Error opening UKV NWP Zarr: {e}")
            import traceback
            traceback.print_exc()

        try:
            print("\nAttempting to open ECMWF NWP Zarr:")
            ecmwf_ds = xr.open_zarr(temp_nwp_ecmwf_path)
            print("ECMWF NWP Zarr opened successfully")
            print("ECMWF NWP Dataset info:")
            print(ecmwf_ds)
        except Exception as e:
            print(f"Error opening ECMWF NWP Zarr: {e}")
            import traceback
            traceback.print_exc()

        # Use gsp_yields_and_systems directly instead of creating our own data
        print("Starting app...")
        try:
            from pvnet_app.app import app
            print("\nCalling app with GSP IDs: 1-317")
            print("Num workers: 2")
            app(gsp_ids=list(range(1, 318)), num_workers=2)
            print("App completed successfully")
        except Exception as e:
            print(f"Error type: {type(e)}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

    # ASSERTIONS COMMENCE
    all_models = get_all_models(get_day_ahead_only=True, use_ocf_data_sampler=False)
    expected_forecast_results = 0
    for model_config in all_models:
        expected_forecast_results += 318 + model_config.save_gsp_sum

    forecasts = db_session.query(ForecastSQL).all()
    assert len(forecasts) == expected_forecast_results * 2

    # # Check probabilistic added
    # assert "90" in forecasts[0].forecast_values[0].properties
    # assert "10" in forecasts[0].forecast_values[0].properties

    # # 72 time steps in forecast
    # expected_forecast_timesteps = 72

    # assert (
    #     len(db_session.query(ForecastValueSQL).all())
    #     == expected_forecast_results * expected_forecast_timesteps
    # )
    # assert (
    #     len(db_session.query(ForecastValueLatestSQL).all())
    #     == expected_forecast_results * expected_forecast_timesteps
    # )
    # assert (
    #     len(db_session.query(ForecastValueSevenDaysSQL).all())
    #     == expected_forecast_results * expected_forecast_timesteps
    # )
