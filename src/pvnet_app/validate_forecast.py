"""Functions to validate solar forecasts."""

import logging

import numpy as np
import pvlib
import xarray as xr

logger = logging.getLogger(__name__)

# A forecast fails validation if the national forecast is above this fraction of capacity
# The GSP forecasts are not checked on this criteria since a few regions can have generations
# significantly above the capacity
NATIONAL_RELATIVE_MAX_FORECAST: float = 1.0

# A forecast fails validation if the national forecast is above this absolute value in MW
# Note: The all time peak generation as of 2026-07-06 was ~16.3 GW
#       See https://www.solar.sheffield.ac.uk/pvlive/
NATIONAL_ABSOLUTE_MAX_FORECAST_MW: int = 20_000


def check_forecast_max(
    national_forecast_mw: xr.DataArray,
    national_capacity_mw: float,
    model_name: str,
) -> bool:
    """Check that the forecast doesn't exceed some limits.

    - Check the forecast doesn't exceed the national capacity.
    - Check the forecast doesn't exceed some arbitrary limit.

    Args:
        national_forecast_mw: The national forecast values (in MW)
        national_capacity_mw: The national PV capacity (in MW)
        model_name: The name of the model that generated the forecast
    """
    forecast_okay = True

    max_forecast_mw = national_forecast_mw.values.max()

    # Check forecast doesn't exceed relative limit of the national capacity
    max_forecast_frac = max_forecast_mw / national_capacity_mw
    if max_forecast_frac > NATIONAL_RELATIVE_MAX_FORECAST:
        logger.warning(
            f"{model_name}: The maximum of the national forecast is {max_forecast_mw} which is "
            f"{max_forecast_frac:.2%} of the national capacity ({national_capacity_mw}MW).",
        )
        forecast_okay = False

    # Check forecast doesn't exceed absolute limit
    if max_forecast_mw > NATIONAL_ABSOLUTE_MAX_FORECAST_MW:
        logger.warning(
            f"{model_name}: The maximum of the national forecast is {max_forecast_mw} which "
            f"exceeds the limit of {NATIONAL_ABSOLUTE_MAX_FORECAST_MW} MW."
        )
        forecast_okay = False

    return forecast_okay


def check_forecast_fluctuations(
    national_forecast_mw: xr.DataArray,
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
    national_forecast_mw: xr.DataArray,
    sun_elevation_lower_limit: float,
    model_name: str,
) -> bool:
    """Check that the forecast values are positive when the sun is up.

    Args:
        national_forecast_mw: The national forecast values (in MW)
        sun_elevation_lower_limit: The lower limit for the sun elevation (in degrees)
        model_name: The name of the model that generated the forecast
    """
    # Calculate the solar position throughout the forecast
    solar_elevation = pvlib.solarposition.get_solarposition(
        time=national_forecast_mw["valid_times_utc"].values,
        longitude=national_forecast_mw["longitude"].item(),
        latitude=national_forecast_mw["latitude"].item(),
        method="nrel_numpy",
    )["elevation"].values

    # Check if forecast values are > 0 when sun elevation is over the threshold
    is_daylight_and_zero = (solar_elevation > sun_elevation_lower_limit) & (
        national_forecast_mw.values <= 0
    )

    if is_daylight_and_zero.sum().item() > 0:
        offending_times = national_forecast_mw["valid_times_utc"].values[is_daylight_and_zero]
        logger.warning(
            f"{model_name}: Forecast values must be > 0 when sun elevation > "
            f"{sun_elevation_lower_limit} degrees. "
            f"Found {len(offending_times)} offending timestamps: {offending_times}",
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
    national_forecast_mw = da_forecast.sel(location_id=0, output_label="p50") * national_capacity_mw

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
