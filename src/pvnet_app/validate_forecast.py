import os
import numpy as np
import pandas as pd
import pvlib
from collections.abc import Callable

def validate_forecast(
    national_forecast_values: np.ndarray,
    national_capacity: float,
    logger_func: Callable[[str], None],
) -> None:
    """Checks various conditions using the full forecast values (in MW).

    Args:
        national_forecast_values: All the forecast values for the nation (in MW).
        national_capacity: The national PV capacity (in MW).
        logger_func: A function that takes a string and logs it 
                     (e.g. self.log_info or logging.info).

    Raises:
        Exception: if above certain critical thresholds.
    """
    # Compute the maximum from the entire forecast array
    max_forecast_mw = float(np.max(national_forecast_values))

    # Check it doesn't exceed 10% above national capacity
    if max_forecast_mw > 1.1 * national_capacity:
        raise Exception(
            f"The maximum of the national forecast is {max_forecast_mw} which is "
            f"greater than 10% above the national capacity ({national_capacity}).",
        )

    # Warn if forecast > 30 GW
    if max_forecast_mw > 30_000:  # 30 GW in MW
        logger_func(
            f"WARNING: National forecast exceeds 30 GW ({max_forecast_mw / 1e3:.2f} GW).",
        )

    # Hard fail if forecast > 100 GW
    if max_forecast_mw > 100_000:  # 100 GW in MW
        raise Exception(
            f"Hard FAIL: The maximum of the forecast is above 100 GW! "
            f"Forecast is {max_forecast_mw / 1e3:.2f} GW.",
        )

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
            "WARNING: Forecast has sudden fluctuations (≥250 MW up and down).")

    if np.any(critical_jumps):
        raise Exception(
            "FAIL: Forecast has critical fluctuations (≥500 MW up and down).")
    
    # Set default value for sun elevation lowr limit
    sun_elevation_lower_limit = float(os.getenv('SUN_ELEVATION_LOWER_LIMIT', 10))

    # Validate based on sun elevation > 10 degrees
    if isinstance(national_forecast_values, pd.Series):
        solpos = pvlib.solarposition.get_solarposition(
            time=national_forecast_values.index,
            latitude=55.3781,  # UK central latitude
            longitude=-3.4360,  # UK central longtitude
            method='nrel_numpy'
        )

        # Check if forecast values are > 0 when sun elevation > 10 degrees
        elevation_above_limit = solpos["elevation"] > sun_elevation_lower_limit

        # Ensure the index of elevation_above_limit matches the index of national_forecast_values
        elevation_above_limit = elevation_above_limit.reindex(national_forecast_values.index, fill_value=False)

        if (national_forecast_values[elevation_above_limit] <= 0).any():
            raise Exception(f"Forecast values must be > 0 when sun elevation > {sun_elevation_lower_limit} degree.")
