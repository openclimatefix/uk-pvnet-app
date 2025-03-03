import logging
import numpy as np


logger = logging.getLogger()


RELATIVE_MAX_FORECAST = 1.1  # 110% of the national capacity

# The all time peak generation as of 2025-03-03 was 11.5 GW
#  - See https://www.solar.sheffield.ac.uk/pvlive/
# Set a limit of 15 GW for now
ABSOLUTE_MAX_FORECAST = 15_000


def check_forecast_max(
    national_forecast_values: np.ndarray,
    national_capacity: float,
    model_name: str,
):
    """Check that the forecast doesn't exceed some limits.

    - Check the forecast doesn't exceed the national capacity.
    - Check the forecast doesn't exceed some arbitrary limit.
    
    Args:
        national_forecast_values: The forecast values for the nation (in MW)
        national_capacity: The national PV capacity (in MW)
        model_name: The name of the model that generated the forecast
    """
    forecast_okay = True

    # Compute the maximum from the entire forecast array
    max_forecast_mw = np.max(national_forecast_values)

    # Check it doesn't exceed the national capacity
    if max_forecast_mw > RELATIVE_MAX_FORECAST * national_capacity:

        cap_frac = max_forecast_mw / national_capacity

        logger.warning(
            f"{model_name}: The maximum of the national forecast is {max_forecast_mw} which is "
            f"greater than {cap_frac:.2%} of the national capacity ({national_capacity}).",
        )
        forecast_okay = False

    if max_forecast_mw > ABSOLUTE_MAX_FORECAST:
        logger.warning(
            f"{model_name}: National forecast exceeds {ABSOLUTE_MAX_FORECAST/1e3:.2f} GW. "
            f"Max forecast value is {max_forecast_mw / 1e3:.2f} GW).",
        )
        forecast_okay = False
    
    return forecast_okay


def check_forecast_fluctuations(
    national_forecast_values: np.ndarray,
    warning_threshold: float,
    error_threshold: float,
    model_name: str,
):
    """Check for fluctuations in the forecast values.
    
    This function checks to see if the forecast values go up, then down, then up again by some
    thresholds.

    Args:
        national_forecast_values: The forecast values for the nation (in MW)
        warning_threshold: The threshold in MW for a warning
        error_threshold: The threshold in MW where the forecast is considered to be in error
        model_name: The name of the model that generated
    """

    forecast_okay = True

    diff = np.diff(national_forecast_values)

    def zig_zag_over_threshold(threshold):
        return (
            (diff[0:-2] > threshold) # forecast goes up
            & (diff[1:-1] < -threshold) # goes down
            & (diff[2:] > threshold) # goes up
        ).any()

    large_jumps = zig_zag_over_threshold(warning_threshold)
    critical_jumps = zig_zag_over_threshold(error_threshold)

    if large_jumps:
        logger.warning(f"{model_name}: Forecast has fluctuations (>{warning_threshold} MW)")

    if critical_jumps:
        logger.warning(f"{model_name}: Forecast has critical fluctuations (>{error_threshold} MW)")
        forecast_okay = False

    return forecast_okay


def validate_forecast(
    national_forecast_values: np.ndarray,
    national_capacity: float,
    zip_zag_warning_threshold: float,
    zig_zag_error_threshold: float,
    model_name: str,
) -> bool:
    """Performs various checks on the forecast values.

    - Checks the forecast doesn't exceed some values. See check_forecast_max()
    - Checks for fluctuations in the forecast values. See check_forecast_fluctuations()

    Args:
        national_forecast_values: All the forecast values for the nation (in MW).
        national_capacity: The national PV capacity (in MW).
        zip_zag_warning_threshold: The threshold in MW for zig-zag check warning.
        zig_zag_error_threshold:  The threshold in MW for zig-zag check failure.
        model_name: The name of the model that generated the forecast.
    """

    forecast_max_okay = check_forecast_max(
        national_forecast_values=national_forecast_values,
        national_capacity=national_capacity,
        model_name=model_name,
    )

    forecast_fluctuations_okay = check_forecast_fluctuations(
        national_forecast_values=national_forecast_values,
        model_name=model_name,
        warning_threshold=zip_zag_warning_threshold,
        error_threshold=zig_zag_error_threshold,
    )
    
    return forecast_max_okay & forecast_fluctuations_okay
