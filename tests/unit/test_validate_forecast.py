import logging

import numpy as np
import pandas as pd

from pvnet_app.validate_forecast import (
    check_forecast_fluctuations,
    check_forecast_max,
    check_forecast_positive_during_daylight,
    validate_forecast,
)


def test_validate_forecast_ok():
    """Test that validate_forecast passes when forecast is valid"""

    # Forecast is significantly below capacity => should pass
    national_forecast = pd.Series(
        [10, 20, 30],  # MW
        index=pd.date_range("2025-01-01 00:00", periods=3, freq="30min"),
    )
    national_capacity = 50  # MW
    zip_zag_warning_threshold = 500  # MW
    zig_zag_error_threshold = 1000  # MW
    sun_elevation_lower_limit = 10  # degrees

    assert check_forecast_max(
        national_forecast=national_forecast,
        national_capacity=national_capacity,
        model_name="test_model",
    )

    assert check_forecast_fluctuations(
        national_forecast=national_forecast,
        warning_threshold=zip_zag_warning_threshold,
        error_threshold=zig_zag_error_threshold,
        model_name="test_model",
    )

    assert check_forecast_positive_during_daylight(
        national_forecast=national_forecast,
        sun_elevation_lower_limit=sun_elevation_lower_limit,
        model_name="test_model",
    )

    assert validate_forecast(
        national_forecast=national_forecast,
        national_capacity=national_capacity,
        zip_zag_warning_threshold=zip_zag_warning_threshold,
        zig_zag_error_threshold=zig_zag_error_threshold,
        sun_elevation_lower_limit=sun_elevation_lower_limit,
        model_name="test_model",
    )


def test_validate_forecast_above_110percent():
    """Test that validate_forecast returns False when forecast is above 110% of capacity"""

    national_forecast = pd.Series(np.array([60]), index=pd.to_datetime(["2025-01-01 00:00"]))

    # 60 MW > 1.1 * 50 MW => should raise an Exception
    forecast_passes = validate_forecast(
        national_forecast=national_forecast,
        national_capacity=50,
        zip_zag_warning_threshold=500,
        zig_zag_error_threshold=1000,
        sun_elevation_lower_limit=10,
        model_name="test_model",
    )

    assert not forecast_passes


def test_validate_forecast_over_15gw():
    """Test that validate_forecast fails if the forecast is above 15 GW"""

    national_forecast = pd.Series(
        np.array([16_000]),
        index=pd.to_datetime(["2025-01-01 00:00"]),
    )

    # 16,000 MW is above 15 GW => Should fail
    forecast_passes = validate_forecast(
        national_forecast=national_forecast,
        national_capacity=100_000,
        zip_zag_warning_threshold=500,
        zig_zag_error_threshold=1000,
        sun_elevation_lower_limit=10,
        model_name="test_model",
    )
    assert not forecast_passes


def test_validate_forecast_no_fluctuations(caplog):
    """Test case with no significant fluctuations."""

    national_forecast = pd.Series(
        [1000, 1100, 1050, 1200, 1150],
        index=pd.date_range(start="2025-01-01 00:00", periods=5, freq="30min"),
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            national_forecast=national_forecast,
            national_capacity=2000,
            zip_zag_warning_threshold=500,
            zig_zag_error_threshold=1000,
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

    national_forecast = pd.Series(
        [1000, 1300, 800, 1200, 500],
        index=pd.date_range(start="2025-01-01 00:00", periods=5, freq="30min"),
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            national_forecast=national_forecast,
            national_capacity=2000,
            zip_zag_warning_threshold=40,
            zig_zag_error_threshold=1000,
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

    national_forecast = pd.Series(
        [1000, 1600, 800, 1301, 500],
        index=pd.date_range(start="2025-01-01 00:00", periods=5, freq="30min"),
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            national_forecast=national_forecast,
            national_capacity=2000,
            zip_zag_warning_threshold=10,
            zig_zag_error_threshold=40,
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

    national_capacity = 2000
    # Create forecast values (some values are ≤ 0 to trigger the exception)
    national_forecast = pd.Series(
        [0, 50, 100, -1, 75],
        index=pd.date_range("2025-01-01 12:00", periods=5, freq="30min"),
    )

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            national_forecast=national_forecast,
            national_capacity=national_capacity,
            zip_zag_warning_threshold=10,
            zig_zag_error_threshold=40,
            sun_elevation_lower_limit=10,
            model_name="test_model",
        )

    # Forecast should not pass
    assert not forecast_passes

    # A warning message should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    warning_string = "Forecast values must be > 0 when sun elevation"
    assert any(warning_string in msg for msg in warnings_logged), "Expected warning not found!"
