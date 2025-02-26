"""Test the app using the legacy dataloader - ocf_datapipes"""

import os
import tempfile

import zarr
from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)

from pvnet_app.model_configs.pydantic_models import get_all_models



def test_app_ecwmf_only(test_t0, db_session, nwp_ecmwf_data, db_url):
    """Test the app for the case running model just on ecmwf"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        os.environ["DB_URL"] = db_url

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
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app

        app(t0=test_t0, gsp_ids=list(range(1, 318)), num_workers=2)

    # Only the models which don't use satellite will be run in this case
    # The models below are the only ones which should have been run
    all_models = get_all_models(get_ecmwf_only=True, use_ocf_data_sampler=False)

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



def test_app(test_t0, db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, db_url):
    """Test the app running the day ahead model"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        os.environ["DB_URL"] = db_url

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
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app

        app(t0=test_t0, gsp_ids=list(range(1, 318)), num_workers=2)

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


def test_app_day_ahead_model(test_t0, db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, db_url):
    """Test the app running the day ahead model"""

    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        os.environ["DB_URL"] = db_url

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
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app

        app(t0=test_t0, gsp_ids=list(range(1, 318)), num_workers=2)

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
