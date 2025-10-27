"""Functions to save forecasts to the database."""
import logging
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_datamodel.models import ForecastSQL, ForecastValue
from nowcasting_datamodel.read.read import get_latest_input_data_last_updated, get_location
from nowcasting_datamodel.read.read_models import get_model
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from sqlalchemy.orm import Session

from grpclib.client import Channel
from dp_sdk.ocf import dp

logger = logging.getLogger(__name__)


def save_forecast(
    session: Session,
    forecast_da: xr.DataArray,
    model_tag: str,
    save_gsp_to_recent: bool,
    apply_adjuster: bool,
    save_gsp_sum: bool,
) -> None:
    """Save forecast DataArray to database.

    Args:
        session: Database session
        forecast_da: DataArray of absolute forecasts for all GSPs
        model_tag: the name of the model to saved to the database
        save_gsp_to_recent: Whether to save GSP forecasts to the last_seven_days table
        apply_adjuster: Whether to apply the adjuster when saving forecasts
        save_gsp_sum: Whether to save the sum of GSPs as a separate forecast
    """
    logger.info("Converting DataArray to ForecastSQL objects")

    sql_forecasts = convert_dataarray_to_forecasts(
        forecast_da,
        session,
        model_tag=model_tag,
    )

    logger.info("Saving ForecastSQL to database")

    if save_gsp_to_recent:
        # Save all forecasts and save to last_seven_days table
        save_sql_forecasts(
            forecasts=sql_forecasts,
            session=session,
            update_national=True,
            update_gsp=True,
            apply_adjuster=apply_adjuster,
            save_to_last_seven_days=True,
        )
    else:
        # Save national and save to last_seven_days table
        save_sql_forecasts(
            forecasts=sql_forecasts[0:1],
            session=session,
            update_national=True,
            update_gsp=False,
            apply_adjuster=apply_adjuster,
            save_to_last_seven_days=True,
        )

        # Save GSP results but not to last_seven_dats table
        save_sql_forecasts(
            forecasts=sql_forecasts[1:],
            session=session,
            update_national=False,
            update_gsp=True,
            apply_adjuster=apply_adjuster,
            save_to_last_seven_days=False,
        )

    if save_gsp_sum:
        # Compute the sum if we are logging the sum of GSPs independently
        da_abs_sum_gsps = (
            forecast_da.sel(gsp_id=slice(1, None))
            .sum(dim="gsp_id")
            # Only select the central forecast for the GSP sum. The sums of different p-levels
            # are not a meaningful qauntities
            .sel(output_label=["forecast_mw"])
            .expand_dims(dim="gsp_id", axis=0)
            .assign_coords(gsp_id=[0])
        )

        # Save the sum of GSPs independently - mainly for summation model monitoring
        gsp_sum_sql_forecasts = convert_dataarray_to_forecasts(
            da_abs_sum_gsps,
            session,
            model_tag=f"{model_tag}_gsp_sum",
        )

        save_sql_forecasts(
            forecasts=gsp_sum_sql_forecasts,
            session=session,
            update_national=True,
            update_gsp=False,
            apply_adjuster=False,
            save_to_last_seven_days=True,
        )


def convert_dataarray_to_forecasts(
    da_preds: xr.DataArray,
    session: Session,
    model_tag: str,
) -> list[ForecastSQL]:
    """Make a ForecastSQL object from a DataArray.

    Args:
        da_preds: DataArray of forecasted values
        session: Database session
        model_tag: the name of the model to saved to the database
    Return:
        List of ForecastSQL objects
    """
    # Get time when the input data was last updated
    # TODO: This time will probably be wrong. It can take 15 mins to run the app, so the
    # forecast will have downloaded older data than is reflected here
    input_data_last_updated = get_latest_input_data_last_updated(session=session)

    model = get_model(name=model_tag, version=version("pvnet_app"), session=session)

    forecasts = []

    for gsp_id in da_preds.gsp_id.values:
        da_gsp = da_preds.sel(gsp_id=gsp_id)

        forecast_values = []

        for target_time in pd.to_datetime(da_gsp.target_datetime_utc.values):
            da_gsp_time = da_gsp.sel(target_datetime_utc=target_time)

            forecast_value_sql = ForecastValue(
                target_time=target_time.replace(tzinfo=UTC),
                expected_power_generation_megawatts=(
                    da_gsp_time.sel(output_label="forecast_mw").item()
                ),
            ).to_orm()

            properties = {}

            for p_level in ["10", "90"]:
                if f"forecast_mw_plevel_{p_level}" in da_gsp_time.output_label:
                    p_val = da_gsp_time.sel(output_label=f"forecast_mw_plevel_{p_level}").item()
                    # `p[10, 90]` can be NaN if PVNet has probabilistic outputs and
                    # PVNet_summation doesn't, or vice versa. Do not log the value if NaN
                    if not np.isnan(p_val):
                        properties[p_level] = p_val

            if len(properties) > 0:
                forecast_value_sql.properties = properties

            forecast_values.append(forecast_value_sql)

        location = get_location(session=session, gsp_id=int(gsp_id))

        forecast = ForecastSQL(
            model=model,
            # TODO: Should this time reflect when the forecast is saved, or the forecast
            # init-time?
            forecast_creation_time=datetime.now(tz=UTC),
            location=location,
            input_data_last_updated=input_data_last_updated,
            forecast_values=forecast_values,
            historic=False,
        )

        forecasts.append(forecast)

    return forecasts


async def save_forecast_to_data_platform(forecast_da: xr.DataArray, model_tag:str, init_time_utc:datetime) -> None:

    # 1. big try and except
    try:

        # 2. setup connection / session / thing
        # TODO should make this a with statement
        channel = Channel(host="localhost", port=50051)
        client = dp.DataPlatformServiceStub(channel)

        # 3. get or update or create forecaster version ( this is similar to ml_model before)
        name = model_tag
        app_version = version("pvnet_app")
        # TODO what if this already existis? Should we use a get functions instead
        cf_request = dp.CreateForecasterRequest(
            name=name,
            version=app_version, 
        )
        forecaster = await client.create_forecaster(cf_request)
        # TODO, get the forecastser
        
        # 4a. get Location (get the UK National location)
        # TODO internal/database/postgres/testdata/uk_gsps
        # TODO get location by name

        # 4b Save National, create forecast
        forecast_values = get_forecast_values_from_dataarray(forecast_da, gsp_id=0)
        forecast_request = dp.CreateForecastRequest(
            forecast=dp.CreateForecastRequestForecast(
                forecaster=forecaster,
                location_uuid="0199f281-3721-7b66-a1c5-f5cf625088bf", # TODO change
                energy_source=dp.EnergySource.SOLAR,
                init_time_utc=init_time_utc,
            ),
        values=forecast_values
        )
        _ = await client.create_forecast(forecast_request)

        # TODO GSP
        # 5a. get all GSP locations
        # 5b. For GSP ), create Forecast object

    except Exception as e:
        logger.error(f"Error saving forecast to data platform: {e}")
        raise e


def get_forecast_values_from_dataarray(forecast_da: xr.DataArray, gsp_id: int) -> list[dp.CreateForecastRequestForecastValue]:
    """Convert a DataArray for a single GSP to a list of ForecastValue objects.

    Args:
        da_gsp: DataArray for a single GSP

    """

    da_gsp = forecast_da.sel(gsp_id=gsp_id)

    init_time_utc = pd.to_datetime(da_gsp.init_datetime_utc.values[0])

    forecast_values = []
    for target_time in pd.to_datetime(da_gsp.target_datetime_utc.values):
        da_gsp_time = da_gsp.sel(target_datetime_utc=target_time)

        horizon_mins = int((target_time - init_time_utc).total_seconds() / 60)

        forecast_value = dp.CreateForecastRequestForecastValue(
            horizon_mins=horizon_mins,
            p50_watts=da_gsp_time.sel(output_label="forecast_mw").item()*10**6,
        )
        
        # TODO add p10 and p90 if they exist

        forecast_values.append(forecast_value)

    return forecast_values



