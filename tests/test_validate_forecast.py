import logging

import numpy as np
from pvnet_app.validate_forecast import (
    check_forecast_max, 
    check_forecast_fluctuations, 
    validate_forecast,
)



def test_validate_forecast_ok():
    """Test that validate_forecast passes when forecast is valid"""

    # Forecast is significantly below capacity => should pass
    national_forecast_values = np.array([10, 20, 30])  # MW
    national_capacity = 50  # MW
    zip_zag_warning_threshold = 500  # MW
    zig_zag_error_threshold = 1000  # MW


    assert check_forecast_max(
        national_forecast_values=national_forecast_values,
        national_capacity=national_capacity,
        model_name="test_model",
    )


    assert check_forecast_fluctuations(
        national_forecast_values=national_forecast_values,
        warning_threshold=zip_zag_warning_threshold,
        error_threshold=zig_zag_error_threshold,
        model_name="test_model",
    )

    assert validate_forecast(
        national_forecast_values=national_forecast_values,
        national_capacity=national_capacity,
        zip_zag_warning_threshold=zip_zag_warning_threshold,
        zig_zag_error_threshold=zig_zag_error_threshold,
        model_name="test_model",
    )


def test_validate_forecast_above_110percent():
    """Test that validate_forecast returns False when forecast is above 110% of capacity"""

    # 60 MW > 1.1 * 50 MW => should raise an Exception
    forecast_passes = validate_forecast(
        national_forecast_values=np.array([60]),
        national_capacity=50,
        zip_zag_warning_threshold=500,
        zig_zag_error_threshold=1000,
        model_name="test_model",
    )

    assert not forecast_passes


def test_validate_forecast_over_15gw(caplog):
    """Test that validate_forecast fails if the forecast is above 15 GW"""
    # 16,000 MW is above 15 GW => Should fail
    forecast_passes = validate_forecast(
        national_forecast_values=np.array([16_000]),
        national_capacity=100_000,
        zip_zag_warning_threshold=500,
        zig_zag_error_threshold=1000,
        model_name="test_model",
    )
    assert not forecast_passes
    

def test_validate_forecast_no_fluctuations(caplog):
    """Test case with no significant fluctuations."""

    national_forecast_values = np.array([1000, 1100, 1050, 1200, 1150])
    national_capacity = 2000

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            national_forecast_values=national_forecast_values,
            national_capacity=national_capacity,
            zip_zag_warning_threshold=500,
            zig_zag_error_threshold=1000,
            model_name="test_model",
        )

    # Forecast should pass
    assert forecast_passes

    # No warning messages should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings_logged, f"Unexpected warning messages were logged: {warnings_logged}"


def test_validate_forecast_with_zigzag_warning(caplog):
    """Test case where a warning should be logged due to fluctuations."""

    national_forecast_values = np.array([1000, 1100, 1050, 1200, 1150])
    national_capacity = 2000

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            national_forecast_values=national_forecast_values,
            national_capacity=national_capacity,
            zip_zag_warning_threshold=40,
            zig_zag_error_threshold=1000,
            model_name="test_model",
        )

    # Forecast should pass
    assert forecast_passes

    # A warning message should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Forecast has fluctuations" in msg for msg in warnings_logged), \
        "Expected warning not found!"



def test_validate_forecast_with_zigzag_failure(caplog):
    """Test case where a warning should be logged and check failed due to fluctuations."""

    national_forecast_values = np.array([1000, 1100, 1050, 1200, 1150])
    national_capacity = 2000

    # Capture warning messages
    with caplog.at_level(logging.WARNING):
        forecast_passes = validate_forecast(
            national_forecast_values=national_forecast_values,
            national_capacity=national_capacity,
            zip_zag_warning_threshold=10,
            zig_zag_error_threshold=40,
            model_name="test_model",
        )

    # Forecast should not pass
    assert not forecast_passes

    # A warning message should be logged
    warnings_logged = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Forecast has critical fluctuations" in msg for msg in warnings_logged), \
        "Expected warning not found!"
