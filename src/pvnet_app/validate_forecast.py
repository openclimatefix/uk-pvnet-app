import logging
import numpy as np
import pandas as pd
import pvlib

logger = logging.getLogger()

# A forecast is bad if above this fraction of national capacity
RELATIVE_MAX_FORECAST = 1.1

# A forecast is bad if above this absolute value
# Note: The all time peak generation as of 2025-05-20 was 13.2 GW
#  - See https://www.solar.sheffield.ac.uk/pvlive/
ABSOLUTE_MAX_FORECAST = 15_000

# The UK's longitude and latitude - used for solar position calculations
UK_LONGITUDE = -3.4360
UK_LATITUDE = 55.3781


def check_forecast_max(
    national_forecast: pd.Series,
    national_capacity: float,
    model_name: str,
):
    """Check that the forecast doesn't exceed some limits."""
    forecast_okay = True
    max_forecast_mw = national_forecast.values.max()

    if max_forecast_mw > RELATIVE_MAX_FORECAST * national_capacity:
        cap_frac = max_forecast_mw / national_capacity
        logger.warning(
            f"{model_name}: The maximum of the national forecast is {max_forecast_mw} which is "
            f"greater than {cap_frac:.2%} of the national capacity ({national_capacity}).",
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
    national_forecast: pd.Series,
    warning_threshold: float,
    error_threshold: float,
    model_name: str,
):
    """Check for fluctuations in the forecast values."""
    forecast_okay = True
    diff = np.diff(national_forecast.values)

    def zig_zag_over_threshold(threshold):
        return (
            (diff[0:-2] > threshold) &
            (diff[1:-1] < -threshold) &
            (diff[2:] > threshold)
        ).any()

    large_jumps = zig_zag_over_threshold(warning_threshold)
    critical_jumps = zig_zag_over_threshold(error_threshold)

    if large_jumps:
        logger.warning(f"{model_name}: Forecast has fluctuations (>{warning_threshold} MW)")

    if critical_jumps:
        logger.warning(f"{model_name}: Forecast has critical fluctuations (>{error_threshold} MW)")
        forecast_okay = False

    return forecast_okay


def check_forecast_positive_during_daylight(
    national_forecast: pd.Series,
    sun_elevation_lower_limit: float,
    model_name: str,
):
    """Check that the forecast values are positive when the sun is up."""
    forecast_okay = True

    solpos = pvlib.solarposition.get_solarposition(
        time=national_forecast.index,
        longitude=UK_LONGITUDE,
        latitude=UK_LATITUDE,
        method='nrel_numpy'
    )

    daylight_mask = solpos["elevation"] > sun_elevation_lower_limit
    bad_times = national_forecast[daylight_mask][national_forecast[daylight_mask] <= 0]

    if not bad_times.empty:
        logger.warning(
            f"{model_name}: Forecast values must be > 0 when sun elevation > "
            f"{sun_elevation_lower_limit} degrees. "
            f"Found {len(bad_times)} offending timestamps: {bad_times.index.tolist()}"
        )
        forecast_okay = False

    return forecast_okay


def validate_forecast(
    national_forecast: pd.Series,
    national_capacity: float,
    zip_zag_warning_threshold: float,
    zig_zag_error_threshold: float,
    sun_elevation_lower_limit: float,
    model_name: str,
) -> bool:
    """Performs various checks on the forecast values."""
    forecast_max_okay = check_forecast_max(
        national_forecast=national_forecast,
        national_capacity=national_capacity,
        model_name=model_name,
    )

    forecast_fluctuations_okay = check_forecast_fluctuations(
        national_forecast=national_forecast,
        model_name=model_name,
        warning_threshold=zip_zag_warning_threshold,
        error_threshold=zig_zag_error_threshold,
    )

    forecast_positive_during_daylight = check_forecast_positive_during_daylight(
        national_forecast=national_forecast,
        sun_elevation_lower_limit=sun_elevation_lower_limit,
        model_name=model_name,
    )

    return forecast_max_okay & forecast_fluctuations_okay & forecast_positive_during_daylight
