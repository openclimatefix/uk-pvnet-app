import logging

import numpy as np
import os
import pytest

from pvnet_app.validate_forecast import validate_forecast


def test_validate_forecast_ok():
    """
    Test that validate_forecast does not raise an error when the forecast
    values are below 110% of the national capacity.
    """
    # Ccapture log messages in a list so assertions can be done on them if needed
    logs = []

    def dummy_logger(msg: str): logs.append(msg)

    # Forecast is significantly below capacity => no warnings or errors
    national_forecast_values = np.array([10, 20, 30])  # MW
    national_capacity = 50  # MW

    validate_forecast(
        national_forecast_values=national_forecast_values,
        national_capacity=national_capacity,
        logger_func=dummy_logger,
    )

    # Assert that we didn't raise any Exceptions and no logs were produced
    assert len(logs) == 0


def test_validate_forecast_above_110percent_raises():
    """
    Test that validate_forecast raises an Exception when the maximum
    forecast value exceeds 110% of capacity.
    """
    # 60 MW > 1.1 * 50 MW => should raise an Exception
    with pytest.raises(Exception) as excinfo:
        validate_forecast(
            national_forecast_values=np.array([60]),
            national_capacity=50,
            logger_func=lambda x: None,  # We don't care about logs here
        )
    assert "greater than 10% above the national capacity" in str(excinfo.value)


def test_validate_forecast_warns_when_over_30gw(caplog):
    """
    Test that validate_forecast warns if the forecast exceeds 30 GW (30,000 MW).
    We'll use pytest's 'caplog' fixture to check for the warning message.
    """
    # 31,000 MW is above 30 GW => Should generate a warning log
    with caplog.at_level(logging.INFO):
        validate_forecast(
            national_forecast_values=np.array([31_000]),
            national_capacity=100_000,
            logger_func=logging.info,
        )
    # Check that the warning message is in the logs
    assert "WARNING: National forecast exceeds 30 GW (31.00 GW)." in caplog.text


def test_validate_forecast_above_100_gw_raises():
    """
    Test that validate_forecast raises an Exception if forecast is above 100 GW.
    """
    # 101,000 MW is above 100 GW => Should raise an Exception
    with pytest.raises(Exception) as excinfo:
        validate_forecast(
            national_forecast_values=np.array([101_000]),
            national_capacity=200_000,
            logger_func=lambda x: None,
        )
    assert "Hard FAIL: The maximum of the forecast is above 100 GW!" in str(
        excinfo.value)


def test_validate_forecast_no_fluctuations():
    """Test case with no significant fluctuations."""
    logged_messages = []

    def logger_func(message):
        logged_messages.append(message)

    os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "500"
    national_forecast_values = np.array([1000, 1100, 1050, 1200, 1150])
    national_capacity = 2000

    # No warnings or exceptions expected
    validate_forecast(national_forecast_values, national_capacity, logger_func)

    assert not logged_messages, "Unexpected warnings logged!"


def test_validate_forecast_with_warning():
    """Test case where a warning should be logged due to fluctuations ≥250 MW up and down."""
    logged_messages = []

    def logger_func(message):
        logged_messages.append(message)

    os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "500"
    national_forecast_values = np.array([1000, 1300, 800, 1200, 500])
    national_capacity = 2000

    validate_forecast(national_forecast_values, national_capacity, logger_func)

    assert any("WARNING: Forecast has sudden fluctuations" in msg for msg in logged_messages), \
        "Expected warning not found!"


def test_validate_forecast_with_exception():
    """Test case where an exception should be raised due to critical fluctuations ≥500 MW up and down."""
    logged_messages = []

    def logger_func(message):
        logged_messages.append(message)

    os.environ["FORECAST_VALIDATE_ZIG_ZAG_ERROR"] = "500"
    national_forecast_values = np.array([1000, 1600, 800, 1301, 500])
    national_capacity = 2000

    with pytest.raises(Exception, match="FAIL: Forecast has critical fluctuations"):
        validate_forecast(national_forecast_values,
                          national_capacity, logger_func)
