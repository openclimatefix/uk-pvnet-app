import os
import tempfile

import zarr
from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)

from pvnet_app.app import run
from pvnet_app.model_configs.pydantic_models import get_all_models

NUM_GSPS = 331


def check_number_of_forecasts(model_configs, db_session):
    """Check the app has added the expected number of forecast values to the database"""

    # Check correct number of forecasts have been made
    # (Number of GSPs + 1 National + maybe GSP-sum) forecasts
    # Forecast made with multiple models
    expected_num_forecasts = 0
    expected_num_forecast_values = 0
    for model_config in model_configs:
        # The number of forecasts
        num_forecasts = NUM_GSPS + 1 + model_config.save_gsp_sum
        expected_num_forecasts += num_forecasts
        # The number of forecast values - 16 for intraday, 36 for day-ahead)
        expected_num_forecast_values += num_forecasts * (72 if model_config.is_day_ahead else 16)

    forecasts = db_session.query(ForecastSQL).all()
    # Doubled for historic and forecast
    assert len(forecasts) == expected_num_forecasts * 2

    # Check probabilistic added
    assert "90" in forecasts[0].forecast_values[0].properties
    assert "10" in forecasts[0].forecast_values[0].properties

    assert len(db_session.query(ForecastValueSQL).all()) == expected_num_forecast_values
    assert len(db_session.query(ForecastValueLatestSQL).all()) == expected_num_forecast_values

    expected_num_forecast_values = 0
    for model_config in model_configs:
        num_forecasts = 1 + NUM_GSPS * model_config.save_gsp_to_recent + model_config.save_gsp_sum
        expected_num_forecast_values += num_forecasts * (72 if model_config.is_day_ahead else 16)

    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == expected_num_forecast_values


def test_app(
    test_t0,
    db_session,
    nwp_ukv_data,
    nwp_ecmwf_data,
    sat_5_data_zero_delay,
    cloudcasting_data,
    db_url,
):
    """Test the app running the intraday models"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        os.environ["DB_URL"] = db_url

        # The app loads sat and NWP data from environment variable
        # Save out data, and set paths as environmental variables
        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(temp_nwp_path)

        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        os.environ["CLOUDCASTING_ZARR_PATH"] = temp_nwp_path = "temp_cloudcasting.zarr"
        cloudcasting_data.to_zarr(temp_nwp_path)

        # In production sat zarr is zipped
        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path = "temp_sat.zarr.zip"
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data_zero_delay.to_zarr(store)

        # Set environmental variables
        os.environ["RUN_CRITICAL_MODELS_ONLY"] = "False"
        os.environ["ALLOW_SAVE_GSP_SUM"] = "True"
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"
        os.environ["FORECAST_VALIDATION_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        # Run prediction
        run(t0=test_t0)

    model_configs = get_all_models(get_critical_only=False)
    check_number_of_forecasts(model_configs, db_session)


def test_app_no_sat(test_t0, db_session, nwp_ukv_data, nwp_ecmwf_data, db_url):
    """Test the app for the case when no satellite data is available"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        os.environ["DB_URL"] = db_url

        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(temp_nwp_path)

        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        # There is no satellite data available at the environ path
        os.environ["SATELLITE_ZARR_PATH"] = "nonexistent_sat.zarr.zip"

        os.environ["RUN_CRITICAL_MODELS_ONLY"] = "False"
        os.environ["ALLOW_SAVE_GSP_SUM"] = "True"
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"
        os.environ["FORECAST_VALIDATION_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        run(t0=test_t0)

    # Only the models which don't use satellite will be run in this case
    # The models below are the only ones which should have been run
    model_configs = get_all_models()
    model_configs = [model for model in model_configs if not model.uses_satellite_data]

    check_number_of_forecasts(model_configs, db_session)
