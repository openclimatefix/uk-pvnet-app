"""Function to help validate PV forecasts."""

import logging

import numpy as np
import pandas as pd
import pvlib
import xarray as xr

logger = logging.getLogger(__name__)

# A forecast is bad if above this fraction of national capacity
RELATIVE_MAX_FORECAST = 1.1

# A forecast is bad if above this absolute value
# Note: The all time peak generation as of 2025-04-27 was ~15.4 GW
#  - See https://www.solar.sheffield.ac.uk/pvlive/
ABSOLUTE_MAX_FORECAST = 20_000

# The UK's longitude and latitude - used for solar position calculations
UK_LONGITUDE = -3.4360
UK_LATITUDE = 55.3781


def check_forecast_max(
    national_forecast_mw: pd.Series,
    national_capacity_mw: float,
    model_name: str,
) -> bool:
    """Check that the forecast doesn't exceed some limits.

    - Check the forecast doesn't exceed the national capacity.
    - Check the forecast doesn't exceed some arbitrary limit.

    Args:
        national_forecast_mw: The forecast values for the nation (in MW)
        national_capacity_mw: The national PV capacity (in MW)
        model_name: The name of the model that generated the forecast
    """
    forecast_okay = True

    # Compute the maximum from the entire forecast array
    max_forecast_mw = national_forecast_mw.values.max()

    # Check it doesn't exceed the national capacity
    max_forecast_frac = max_forecast_mw / national_capacity_mw
    if max_forecast_frac > RELATIVE_MAX_FORECAST:
        logger.warning(
            f"{model_name}: The maximum of the national forecast is {max_forecast_mw} which is "
            f"{max_forecast_frac:.2%} of the national capacity ({national_capacity_mw}MW).",
        )
        forecast_okay = False

    if max_forecast_mw > ABSOLUTE_MAX_FORECAST:
        logger.warning(
            f"{model_name}: National forecast exceeds {ABSOLUTE_MAX_FORECAST / 1e3:.2f} GW. "
            f"Max forecast value is {max_forecast_mw / 1e3:.2f} GW).",
        )
        forecast_okay = False

    return forecast_okay


def check_forecast_fluctuations(
    national_forecast_mw: pd.Series,
    warning_threshold_mw: float,
    error_threshold_mw: float,
    model_name: str,
) -> bool:
    """Check for fluctuations in the forecast values.

    This function checks to see if the forecast values go up, then down, then up again by some
    thresholds.

    Args:
        national_forecast_mw: The national forecast values (in MW)
        warning_threshold_mw: The threshold in MW for a warning
        error_threshold_mw: The threshold in MW where the forecast is considered to be in error
        model_name: The name of the model that generated the forecast
    """
    diff = np.diff(national_forecast_mw.values)

    def zig_zag_over_threshold(threshold: float) -> bool:
        return (
            (diff[0:-2] > threshold)  # forecast goes up
            & (diff[1:-1] < -threshold)  # goes down
            & (diff[2:] > threshold)  # goes up
        ).any()

    has_large_jumps = zig_zag_over_threshold(warning_threshold_mw)
    has_critical_jumps = zig_zag_over_threshold(error_threshold_mw)

    if has_large_jumps:
        logger.warning(f"{model_name}: Forecast has fluctuations (>{warning_threshold_mw} MW)")

    if has_critical_jumps:
        logger.warning(
            f"{model_name}: Forecast has critical fluctuations (>{error_threshold_mw} MW)",
        )

    return not has_critical_jumps


def check_forecast_positive_during_daylight(
    national_forecast_mw: pd.Series,
    sun_elevation_lower_limit: float,
    model_name: str,
) -> bool:
    """Check that the forecast values are positive when the sun is up.

    Args:
        national_forecast_mw: The forecast values for the nation (in MW)
        sun_elevation_lower_limit: The lower limit for the sun elevation (in degrees)
        model_name: The name of the model that generated the forecast
    """
    # Calculate the solar position throughout the forecast
    solpos = pvlib.solarposition.get_solarposition(
        time=national_forecast_mw.index,  # The index is expect to be the valid times
        longitude=UK_LONGITUDE,
        latitude=UK_LATITUDE,
        method="nrel_numpy",
    )

    # Check if forecast values are > 0 when sun elevation is over the threshold
    is_daylight = solpos["elevation"] > sun_elevation_lower_limit
    bad_times = national_forecast_mw[is_daylight & (national_forecast_mw <= 0)]

    if (num_bad_times := len(bad_times)) > 0:
        logger.warning(
            f"{model_name}: Forecast values must be > 0 when sun elevation > "
            f"{sun_elevation_lower_limit} degrees. "
            f"Found {num_bad_times} offending timestamps: {bad_times.index.tolist()}",
        )
        return False
    else:
        return True


def validate_forecast(
    da_forecast: xr.DataArray,
    national_capacity_mw: float,
    zig_zag_warning_threshold_mw: float,
    zig_zag_error_threshold_mw: float,
    sun_elevation_lower_limit: float,
    model_name: str,
) -> bool:
    """Performs various checks on the forecast values.

    - Checks the forecast doesn't exceed some values. See `check_forecast_max()`
    - Checks for fluctuations in the forecast values. See `check_forecast_fluctuations()`
    - Checks that forecast values are positive when the sun is up. See
      `check_forecast_positive_during_daylight()`

    Args:
        da_forecast: The normalised forecast values.
        national_capacity_mw: The national PV capacity (in MW).
        zig_zag_warning_threshold_mw: The threshold in MW for zig-zag check warning.
        zig_zag_error_threshold_mw:  The threshold in MW for zig-zag check failure.
        sun_elevation_lower_limit: The lower limit for the sun elevation (in degrees). The forecast
            values must be positive when the sun is above this angle.
        model_name: The name of the model that generated the forecast.
    """
    # Compute the national forecast in MW from the normalised forecast
    # Validation is only performed on the national forecast
    national_forecast_mw = (
        da_forecast.sel(gsp_id=0, output_label="p50").to_series() * national_capacity_mw
    )

    forecast_max_okay = check_forecast_max(
        national_forecast_mw=national_forecast_mw,
        national_capacity_mw=national_capacity_mw,
        model_name=model_name,
    )

    forecast_fluctuations_okay = check_forecast_fluctuations(
        national_forecast_mw=national_forecast_mw,
        warning_threshold_mw=zig_zag_warning_threshold_mw,
        error_threshold_mw=zig_zag_error_threshold_mw,
        model_name=model_name,
    )

    forecast_positive_during_daylight = check_forecast_positive_during_daylight(
        national_forecast_mw=national_forecast_mw,
        sun_elevation_lower_limit=sun_elevation_lower_limit,
        model_name=model_name,
    )

    return forecast_max_okay & forecast_fluctuations_okay & forecast_positive_during_daylight
