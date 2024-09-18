import tempfile
import zarr
import os

from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)

from pvnet_app.consts import sat_path, nwp_ukv_path, nwp_ecmwf_path
from pvnet_app.data.satellite import sat_5_path, sat_15_path


def test_app(
    db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, gsp_yields_and_systems, me_latest
):
    # Environment variable DB_URL is set in engine_url, which is called by db_session
    # set NWP_ZARR_PATH
    # save nwp_data to temporary file, and set NWP_ZARR_PATH
    # SATELLITE_ZARR_PATH
    # save sat_data to temporary file, and set SATELLITE_ZARR_PATH
    # GSP data


    with tempfile.TemporaryDirectory() as tmpdirname:

        os.chdir(tmpdirname)

        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables
        temp_nwp_path = f"temp_nwp_ukv.zarr"
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path
        nwp_ukv_data.to_zarr(temp_nwp_path)

        temp_nwp_path = f"temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        # In production sat zarr is zipped
        temp_sat_path = f"temp_sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data.to_zarr(store)

        # Set environmental variables
        os.environ["RUN_EXTRA_MODELS"] = "True"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app, models_dict

        app(gsp_ids=list(range(1, 318)), num_workers=2)

    # Check correct number of forecasts have been made
    # (317 GSPs + 1 National) = 318 forecasts
    # Forecast made with multiple models
    expected_forecast_results = 0
    for model_config in models_dict.values():
        expected_forecast_results += 318

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
    for model_config in models_dict.values():
        # National
        expected_forecast_results += 1
        # GSP
        expected_forecast_results += 317 * model_config["save_gsp_to_forecast_value_last_seven_days"]
        expected_forecast_results += model_config["save_gsp_sum"]  # optional Sum of GSPs

    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == expected_forecast_results * 16


def test_app_day_ahead_model(
    db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, gsp_yields_and_systems, me_latest
):
    # Test app with day ahead model config
    # Environment variable DB_URL is set in engine_url, which is called by db_session
    # set NWP_ZARR_PATH
    # save nwp_data to temporary file, and set NWP_ZARR_PATH
    # SATELLITE_ZARR_PATH
    # save sat_data to temporary file, and set SATELLITE_ZARR_PATH
    # GSP data

    with tempfile.TemporaryDirectory() as tmpdirname:
        
        os.chdir(tmpdirname)

        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables
        temp_nwp_path = f"temp_nwp_ukv.zarr"
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path
        nwp_ukv_data.to_zarr(temp_nwp_path)

        temp_nwp_path = f"temp_nwp_ecmwf.zarr"
        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        # In production sat zarr is zipped
        temp_sat_path = f"temp_sat.zarr.zip"
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data.to_zarr(store)

        # Set environmental variables
        os.environ["DAY_AHEAD_MODEL"] = "True"
        os.environ["RUN_EXTRA_MODELS"] = "False"

        # Run prediction
        # Thes import needs to come after the environ vars have been set
        from pvnet_app.app import app, day_ahead_model_dict

        app(gsp_ids=list(range(1, 318)), num_workers=2)

    # Check correct number of forecasts have been made
    # (317 GSPs + 1 National + maybe GSP-sum) = 318 or 319 forecasts
    # Forecast made with multiple models
    expected_forecast_results = 0
    for model_config in day_ahead_model_dict.values():
        expected_forecast_results += 318 + model_config["save_gsp_sum"]

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
