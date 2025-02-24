import os
import numpy as np
from collections.abc import Callable

def validate_forecast(
    national_forecast_values: np.ndarray,
    national_capacity: float,
    logger_func: Callable[[str], None],
) -> bool:  # Now returns a boolean instead of raising exceptions
    """Checks various conditions using the full forecast values (in MW).

    Args:
        national_forecast_values: All the forecast values for the nation (in MW).
        national_capacity: The national PV capacity (in MW).
        logger_func: A function that takes a string and logs it.

    Returns:
        bool: True if validation passes, False if validation fails.
    """

    # Make validation optional using an environment variable
    strict_validation = os.getenv("STRICT_VALIDATION", "False").lower() == "true"

    # Compute the maximum from the entire forecast array
    max_forecast_mw = float(np.max(national_forecast_values))

    # Check it doesn't exceed 10% above national capacity
    if max_forecast_mw > 1.1 * national_capacity:
        msg = (
            f"Validation Failed: The max national forecast is {max_forecast_mw} MW, "
            f"exceeding 10% above national capacity ({national_capacity} MW)."
        )
        if strict_validation:  # Only raise an exception in strict mode
            raise Exception(msg)
        logger_func(f"ERROR: {msg}")  # Log the error instead of stopping execution
        return False  # Return False so validation failure does not stop other models

    # Warn if forecast > 30 GW
    if max_forecast_mw > 30_000:  # 30 GW in MW
        logger_func(
            f"WARNING: National forecast exceeds 30 GW ({max_forecast_mw / 1e3:.2f} GW)."
        )

    # Hard fail if forecast > 100 GW
    if max_forecast_mw > 100_000:  # 100 GW in MW
        msg = (
            f"Validation Failed: The forecast is above 100 GW! "
            f"Forecast is {max_forecast_mw / 1e3:.2f} GW."
        )
        if strict_validation:  #  Raise an exception only in strict mode
            raise Exception(msg)
        logger_func(f"ERROR: {msg}")  #  Log the error instead
        return False  #  Return False instead of stopping execution

    # New Validation: Detect Sudden Fluctuations
    # Compute differences between consecutive timestamps
    zig_zag_gap_warning = float(os.getenv('FORECAST_VALIDATE_ZIG_ZAG_WARNING', 250))
    zig_zag_gap_error = float(os.getenv('FORECAST_VALIDATE_ZIG_ZAG_ERROR', 500))
    diff = np.diff(national_forecast_values)

    large_jumps = \
        (diff[0:-2] > zig_zag_gap_warning) & \
        (diff[1:-1] < -zig_zag_gap_warning) & \
        (diff[2:] > zig_zag_gap_warning)  # Up then down, then up, by 250 MW

    critical_jumps = \
        (diff[0:-2] > zig_zag_gap_error) & \
        (diff[1:-1] < -zig_zag_gap_error) & \
        (diff[2:] > zig_zag_gap_error)  # Up then down, then up, by 500 MW

    if np.any(large_jumps):
        logger_func(
            "WARNING: Forecast has sudden fluctuations (≥250 MW up and down)."
        )

    if np.any(critical_jumps):
        msg = "Validation Failed: Forecast has critical fluctuations (≥500 MW up and down)."
        if strict_validation:  # NEW: Raise exception only in strict mode
            raise Exception(msg)
        logger_func(f"ERROR: {msg}")  #  Log the error instead
        return False  # Return False instead of stopping execution

    return True  # Return True if validation passes
