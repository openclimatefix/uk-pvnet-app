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
from pvnet_app.app import app




def test_app(test_t0, db_session, nwp_ukv_data, nwp_ecmwf_data, sat_5_data, db_url):
    """Test the app running the legacy day ahead model"""

    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)

        os.environ["DB_URL"] = db_url

        os.environ["NWP_UKV_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ukv.zarr"
        nwp_ukv_data.to_zarr(temp_nwp_path)

        os.environ["NWP_ECMWF_ZARR_PATH"] = temp_nwp_path = "temp_nwp_ecmwf.zarr"
        nwp_ecmwf_data.to_zarr(temp_nwp_path)

        os.environ["SATELLITE_ZARR_PATH"] = temp_sat_path = "temp_sat.zarr.zip"
        with zarr.storage.ZipStore(temp_sat_path, mode="x") as store:
            sat_5_data.to_zarr(store)

        os.environ["DAY_AHEAD_MODEL"] = "False"
        os.environ["RUN_CRITICAL_MODELS_ONLY"] = "False"
        os.environ["ALLOW_SAVE_GSP_SUM"] = "True"
        os.environ["USE_OCF_DATA_SAMPLER"] = "False"
        os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "100000"
        os.environ["FORECAST_VALIDATION_SUN_ELEVATION_LOWER_LIMIT"] = "90"

        app(t0=test_t0, gsp_ids=list(range(1, 318)), num_workers=2)

    all_models = get_all_models(get_critical_only=False, use_ocf_data_sampler=False)

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

    # 16 time steps in forecast
    expected_forecast_timesteps = 16

    assert (
        len(db_session.query(ForecastValueSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )
    assert (
        len(db_session.query(ForecastValueLatestSQL).all())
        == expected_forecast_results * expected_forecast_timesteps
    )

    expected_forecast_results = 0
    for model_config in all_models:
        # National
        expected_forecast_results += 1
        # GSP
        expected_forecast_results += 317 * model_config.save_gsp_to_recent
        expected_forecast_results += model_config.save_gsp_sum  # optional Sum of GSPs

    assert len(db_session.query(ForecastValueSevenDaysSQL).all()) == expected_forecast_results * 16
