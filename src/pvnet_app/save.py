"""Functions to save forecasts to the database."""

import logging
import os
from datetime import UTC, datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from grpclib.client import Channel
from nowcasting_datamodel.models import ForecastSQL, ForecastValue
from nowcasting_datamodel.read.read import get_latest_input_data_last_updated, get_location
from nowcasting_datamodel.read.read_models import get_model
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# TODO this might move up to app level later on
data_platform_host = os.getenv("DATA_PLATFORM_HOST", "localhost")
data_platform_port = int(os.getenv("DATA_PLATFORM_PORT", "50051"))


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


async def save_forecast_to_data_platform(
    forecast_da: xr.DataArray,
    model_tag: str,
    init_time_utc: datetime,
    client: dp.DataPlatformDataServiceStub | None = None,
) -> None:
    """Save forecast DataArray to data platform.

    We do the following steps:
    1. setup connection and client if not provided
    2. Get all locations from data platform
    3. get Forecaster
    4. loop over all gsps: get the location object
    5. Forecast the forecast values
    6. Save to the data platform

    Args:
        forecast_da: DataArray of forecasts for all GSPs
        model_tag: the name of the model to saved to the database
        init_time_utc: Forecast initialization time
        client: Data platform client. If None, a new client will be created.
    """
    logger.info("Saving forecast to data platform")

    # 1. setup connection / session / thing
    if client is None:
        channel = Channel(host=data_platform_host, port=data_platform_port)
        client = dp.DataPlatformDataServiceStub(channel)

    # 2. Get all locations (Uk national + GSPs)
    all_locations = await get_all_gsp_and_national_locations(client)

    # 3. get or update or create forecaster version ( this is similar to ml_model before)
    name = model_tag.replace("-", "_")
    app_version = version("pvnet_app")

    list_forecasters_request = dp.ListForecastersRequest(latest_versions_only=True)
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)
    forecasters = list_forecasters_response.forecasters

    forecasters_filtered = [f for f in forecasters if f.forecaster_name == name]

    if len(forecasters_filtered) > 0:
        forecaster = forecasters_filtered[0]
    else:
        cf_request = dp.CreateForecasterRequest(
            name=name,
            version=app_version,
        )
        forecaster_response = await client.create_forecaster(cf_request)
        forecaster = forecaster_response.forecaster

    # now loop over all gsps
    for gsp_id in forecast_da.gsp_id.values:
        logger.debug(f"Saving forecast for GSP ID: {gsp_id}")

        # 4. get Location
        location = all_locations[int(gsp_id)]

        # 5. Format the forecast values
        forecast_values = get_forecast_values_from_dataarray(
            forecast_da,
            gsp_id=gsp_id,
            init_time_utc=init_time_utc,
            capacity_watts=location.effective_capacity_watts,
        )
        # 6. Save to data platform
        forecast_request = dp.CreateForecastRequest(
            forecaster=forecaster,
            location_uuid=location.location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            init_time_utc=init_time_utc.replace(tzinfo=UTC),
            values=forecast_values,
        )
        _ = await client.create_forecast(forecast_request)


def get_forecast_values_from_dataarray(
    forecast_da: xr.DataArray,
    gsp_id: int,
    init_time_utc: datetime,
    capacity_watts: int,
) -> list[dp.CreateForecastRequestForecastValue]:
    """Convert a DataArray for a single GSP to a list of ForecastValue objects.

    Args:
        forecast_da: DataArray for a single GSP
        gsp_id: GSP ID
        init_time_utc: Forecast initialization time
        capacity_watts: Capacity of the location in watts

    """
    da_gsp = forecast_da.sel(gsp_id=gsp_id)

    forecast_values = []
    for target_time in pd.to_datetime(da_gsp.target_datetime_utc.values):
        da_gsp_time = da_gsp.sel(target_datetime_utc=target_time)

        target_time = target_time.replace(tzinfo=UTC)
        horizon_mins = int((target_time - init_time_utc).total_seconds() / 60)
        p50_fraction = da_gsp_time.sel(output_label="forecast_mw").item() * 10**6 / capacity_watts

        # TODO tidy this up
        metadata = Struct(fields={"temp": Value(string_value="temp")})

        forecast_value = dp.CreateForecastRequestForecastValue(
            horizon_mins=horizon_mins,
            p50_fraction=p50_fraction,
            p10_fraction=0.01,  # TODO add p10
            p90_fraction=0.99,  # TODO add p90
            metadata=metadata,
        )

        forecast_values.append(forecast_value)

    return forecast_values


async def get_all_gsp_and_national_locations(
    client: dp.DataPlatformDataServiceStub,
) -> dict[int, dp.ListLocationsResponseLocationSummary]:
    """Get all GSP and National locations for solar energy source."""
    all_locations = {}

    # National location
    all_location_request = dp.ListLocationsRequest(
        location_type_filter=dp.LocationType.NATION,
        energy_source_filter=dp.EnergySource.SOLAR,
    )
    location_response = await client.list_locations(all_location_request)
    all_uk_location = [
        loc for loc in location_response.locations if "uk" in loc.location_name.lower()
    ]
    if len(all_uk_location) >= 1:
        all_locations[0] = all_uk_location[0]

    # GSP locations
    all_location_gsp_request = dp.ListLocationsRequest(
        location_type_filter=dp.LocationType.GSP,
        energy_source_filter=dp.EnergySource.SOLAR,
    )
    location_response = await client.list_locations(all_location_gsp_request)
    for loc in location_response.locations:
        all_locations[loc.metadata.to_dict()["gsp_id"]["numberValue"]] = loc

    return all_locations
