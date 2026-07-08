import logging

import numpy as np
import pandas as pd
import xarray as xr

from pvnet_app.validate_forecast import (
    check_forecast_fluctuations,
    check_forecast_max,
    check_forecast_positive_during_daylight,
    validate_forecast,
)

national_max_forecast_mw = 20_000


def make_forecast_dataarray(
    forecast_values: list[float],
    valid_times: pd.DatetimeIndex,
) -> xr.DataArray:
    """Helper function to create a forecast DataArray for testing purposes."""
    return xr.DataArray(
        np.array(forecast_values)[None, :, None],
        coords={
            "location_id": [0],
            "valid_times_utc": valid_times,
            "output_label": ["p50"],
            # Set some dummy coordinates for the location of roughly the UK
            "longitude": ("location_id", [-3]),
            "latitude": ("location_id", [55]),
        },
        dims=["location_id", "valid_times_utc", "output_label"],
    )


def test_validate_forecast_ok():
    """Test that validate_forecast passes when forecast is valid"""

    national_capacity_mw = 50
    zig_zag_warning_threshold_mw = 500
    zig_zag_error_threshold_mw = 1000
    sun_elevation_lower_limit = 10  # degrees

    # Forecast is significantly below capacity => should pass
    da_forecast_mw = make_forecast_dataarray(
        forecast_values=[10, 20, 30],
        valid_times=pd.date_range("2025-01-01 00:00", periods=3, freq="30min"),
    )

    national_forecast_mw = da_forecast_mw.sel(location_id=0, output_label="p50")

    assert check_forecast_max(
        national_forecast_mw=national_forecast_mw,
        national_capacity_mw=national_capacity_mw,
        national_max_forecast_mw=national_max_forecast_mw,
        model_name="test_model",
    )

    assert check_forecast_fluctuations(
        national_forecast_mw=national_forecast_mw,
        warning_threshold_mw=zig_zag_warning_threshold_mw,
        error_threshold_mw=zig_zag_error_threshold_mw,
        model_name="test_model",
    )

    assert check_forecast_positive_during_daylight(
        national_forecast_mw=national_forecast_mw,
        sun_elevation_lower_limit=sun_elevation_lower_limit,
        model_name="test_model",
    )

    assert validate_forecast(
        da_forecast=da_forecast_mw / national_capacity_mw,
        national_capacity_mw=national_capacity_mw,
        zig_zag_warning_threshold_mw=zig_zag_warning_threshold_mw,
        zig_zag_error_threshold_mw=zig_zag_error_threshold_mw,
        national_max_forecast_mw=national_max_forecast_mw,
        sun_elevation_lower_limit=sun_elevation_lower_limit,
        model_name="test_model",
    )


def test_validate_forecast_above_110percent():
    """Test that validate_forecast returns False when forecast is above 100% of capacity"""

    da_forecast = make_forecast_dataarray(
        forecast_values=[1.2],
        valid_times=pd.to_datetime(["2025-01-01 00:00"]),
    )

    forecast_passes = validate_forecast(
        da_forecast=da_forecast,
        national_capacity_mw=50,
        zig_zag_warning_threshold_mw=500,
        zig_zag_error_threshold_mw=1000,
        national_max_forecast_mw=national_max_forecast_mw,
        sun_elevation_lower_limit=10,
        model_name="test_model",
    )

    assert not forecast_passes


def test_validate_forecast_over_20gw():
    """Test that validate_forecast fails if the forecast is above 20 GW"""

    national_capacity_mw = 100_000

    da_forecast = make_forecast_dataarray(
        forecast_values=[21_000 / national_capacity_mw],
        valid_times=pd.to_datetime(["2025-01-01 00:00"]),
    )

    # 21,000 MW is above 20 GW => Should fail
    forecast_passes = validate_forecast(
        da_forecast=da_forecast,
        national_capacity_mw=national_capacity_mw,
        zig_zag_warning_threshold_mw=500,
        zig_zag_error_threshold_mw=1000,
        national_max_forecast_mw=national_max_forecast_mw,
        sun_elevation_lower_limit=10,
        model_name="test_model",
    )
    assert not forecast_passes


def test_validate_forecast_no_fluctuations(caplog):
    """Test case with no significant fluctuations."""

    national_capacity_mw = 2000

    da_forecast = (
        make_forecast_dataarray(
            forecast_values=[1000, 1100, 1050, 1200, 1150],
            valid_times=pd.date_range(start="2025-01-01 00:00", periods=5, freq="30min"),
        )
        / national_capacity_mw
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            da_forecast=da_forecast,
            national_capacity_mw=national_capacity_mw,
            zig_zag_warning_threshold_mw=500,
            zig_zag_error_threshold_mw=1000,
            national_max_forecast_mw=national_max_forecast_mw,
            sun_elevation_lower_limit=10,
            model_name="test_model",
        )

    # Forecast should pass
    assert forecast_passes

    # No warning messages should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_logged) == 0, f"Unexpected warning messages were logged: {warnings_logged}"


def test_validate_forecast_with_zigzag_warning(caplog):
    """Test case where a warning should be logged due to fluctuations."""

    national_capacity_mw = 2000

    da_forecast = (
        make_forecast_dataarray(
            forecast_values=[1000, 1300, 800, 1200, 500],
            valid_times=pd.date_range(start="2025-01-01 00:00", periods=5, freq="30min"),
        )
        / national_capacity_mw
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            da_forecast=da_forecast,
            national_capacity_mw=national_capacity_mw,
            zig_zag_warning_threshold_mw=40,
            zig_zag_error_threshold_mw=1000,
            national_max_forecast_mw=national_max_forecast_mw,
            sun_elevation_lower_limit=10,
            model_name="test_model",
        )

    # Forecast should pass
    assert forecast_passes

    # A warning message should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    warning_string = "Forecast has fluctuations"
    assert any(warning_string in msg for msg in warnings_logged), "Expected warning not found!"


def test_validate_forecast_with_zigzag_failure(caplog):
    """Test case where validation should fail due to fluctuations."""

    national_capacity_mw = 2000

    da_forecast = (
        make_forecast_dataarray(
            forecast_values=[1000, 1600, 800, 1301, 500],
            valid_times=pd.date_range(start="2025-01-01 00:00", periods=5, freq="30min"),
        )
        / national_capacity_mw
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            da_forecast=da_forecast,
            national_capacity_mw=national_capacity_mw,
            zig_zag_warning_threshold_mw=10,
            zig_zag_error_threshold_mw=40,
            national_max_forecast_mw=national_max_forecast_mw,
            sun_elevation_lower_limit=10,
            model_name="test_model",
        )

    # Forecast should not pass
    assert not forecast_passes

    # A warning message should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    warning_string = "Forecast has critical fluctuations"
    assert any(warning_string in msg for msg in warnings_logged), "Expected warning not found!"


def test_validate_forecast_sun_elevation_check(caplog):
    """Test case where a validation should fail due to sun elevation check."""

    national_capacity_mw = 2000
    # Create forecast values (some values are ≤ 0 to trigger the exception)
    da_forecast = (
        make_forecast_dataarray(
            forecast_values=[0, 50, 100, -1, 75],
            valid_times=pd.date_range("2025-01-01 12:00", periods=5, freq="30min"),
        )
        / national_capacity_mw
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            da_forecast=da_forecast,
            national_capacity_mw=national_capacity_mw,
            zig_zag_warning_threshold_mw=10,
            zig_zag_error_threshold_mw=40,
            national_max_forecast_mw=national_max_forecast_mw,
            sun_elevation_lower_limit=10,
            model_name="test_model",
        )

    # Forecast should not pass
    assert not forecast_passes

    # A warning message should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    warning_string = "Forecast values must be > 0 when sun elevation"
    assert any(warning_string in msg for msg in warnings_logged), "Expected warning not found!"
