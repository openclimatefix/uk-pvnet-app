"""Functions to save forecasts to the database."""

import asyncio
import itertools
import logging
from datetime import UTC, datetime
from importlib.metadata import version

import betterproto
import numpy as np
import pandas as pd
import xarray as xr
from betterproto.lib.google.protobuf import Struct
from dp_sdk.ocf import dp
from nowcasting_datamodel.models import ForecastSQL, ForecastValue
from nowcasting_datamodel.read.read import get_latest_input_data_last_updated, get_location
from nowcasting_datamodel.read.read_models import get_model
from nowcasting_datamodel.save.save import save as save_sql_forecasts
from sqlalchemy.orm import Session

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


async def save_forecast_to_data_platform(
    forecast_normed_da: xr.DataArray,
    locations_gsp_uuid_map: dict[int, str],
    model_tag: str,
    init_time_utc: datetime,
    client: dp.DataPlatformDataServiceStub,
) -> None:
    """Save forecast DataArray to data platform.

    We do the following steps:
    1. Get all locations from data platform
    2. get Forecaster
    3. loop over all gsps: get the location object
    4. Forecast the forecast values
    5. Save to the data platform

    Args:
        forecast_normed_da: DataArray of normalized forecasts for all GSPs
        locations_gsp_uuid_map: Mapping of GSP IDs to location UUIDs
        model_tag: the name of the model to saved to the database
        init_time_utc: Forecast initialization time
        client: Data platform client. If None, a new client will be created.
    """
    logger.info("Saving forecast to data platform")

    # strip out timezone from init_time_utc, this works better with xarray datetime formats
    init_time_utc = init_time_utc.replace(tzinfo=None)

    # 2. get or update or create forecaster version ( this is similar to ml_model before)
    forecaster = await create_forecaster_if_not_exists(client=client, model_tag=model_tag)

    # now loop over all gsps
    logger.debug("Processing forecasts for Data Platform")
    tasks = []
    for gsp_id in forecast_normed_da.gsp_id.values:
        # 4. Format the forecast values
        forecast_values = map_values_da_to_dp_requests(
            forecast_normed_da.sel(gsp_id=gsp_id),
            init_time_utc=init_time_utc,
        )
        # 5. Save to data platform
        forecast_request = dp.CreateForecastRequest(
            forecaster=forecaster,
            location_uuid=locations_gsp_uuid_map[int(gsp_id)],
            energy_source=dp.EnergySource.SOLAR,
            init_time_utc=init_time_utc.replace(tzinfo=UTC),
            values=forecast_values,
        )
        tasks.append(asyncio.create_task(client.create_forecast(forecast_request)))

        if gsp_id == 0:
            # make forecast
            adjusted_forecast_request = await make_forecaster_adjuster(
                client,
                location_uuid=locations_gsp_uuid_map[int(gsp_id)],
                init_time_utc=init_time_utc,
                forecast_values=forecast_values,
                model_tag=model_tag,
                forecaster=forecaster,
            )
            tasks.append(asyncio.create_task(client.create_forecast(adjusted_forecast_request)))

    logger.info(f"Saving {len(tasks)} forecasts to Data Platform")
    list_results = await asyncio.gather(*tasks, return_exceptions=True)
    for exc in filter(lambda x: isinstance(x, Exception), list_results):
        raise exc

    logger.info("Saved forecast to Data Platform")


def map_values_da_to_dp_requests(
    gsp_normed_da: xr.DataArray,
    init_time_utc: datetime,
) -> list[dp.CreateForecastRequestForecastValue]:
    """Convert a DataArray for a single GSP to a list of ForecastValue objects.

    Args:
        gsp_normed_da: Normalized DataArray for a single GSP
        init_time_utc: Forecast initialization time
    """
    # create horizon mins
    target_datetime_utc = pd.to_datetime(gsp_normed_da.target_datetime_utc.values)
    horizons_mins = (target_datetime_utc - init_time_utc).total_seconds() / 60
    horizons_mins = horizons_mins.astype(int)

    # Reduce singular dimensions
    gsp_normed_da = gsp_normed_da.squeeze(drop=True)
    p50s = gsp_normed_da.sel(output_label="forecast_fraction").values.astype(float)
    p10s = gsp_normed_da.sel(output_label="forecast_fraction_plevel_10").values.astype(float)
    p90s = gsp_normed_da.sel(output_label="forecast_fraction_plevel_90").values.astype(float)

    forecast_values = []
    for h, p50, p10, p90 in zip(horizons_mins, p50s, p10s, p90s, strict=True):
        forecast_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=h,
                p50_fraction=p50,
                metadata=Struct().from_pydict({}),
                other_statistics_fractions={
                    "p10": p10,
                    "p90": p90,
                },
            ),
        )

    return forecast_values


async def fetch_dp_gsp_uuid_map(
    client: dp.DataPlatformDataServiceStub,
) -> dict[int, str]:
    """Fetch all GSP locations from data platform and map to their uuids."""
    tasks = [
        asyncio.create_task(
            client.list_locations(
                dp.ListLocationsRequest(
                    location_type_filter=loc_type,
                    energy_source_filter=dp.EnergySource.SOLAR,
                ),
            ),
        )
        for loc_type in [dp.LocationType.GSP, dp.LocationType.NATION]
    ]
    list_results = await asyncio.gather(*tasks, return_exceptions=True)
    for exc in filter(lambda x: isinstance(x, Exception), list_results):
        raise exc

    locations_df = (
        # Convert and combine the location lists from the responses into a single DataFrame
        pd.DataFrame.from_dict(
            itertools.chain(
                *[
                    r.to_dict(casing=betterproto.Casing.SNAKE, include_default_values=True)[
                        "locations"
                    ]
                    for r in list_results
                ],
            ),
        )
        # Filter the returned locations to those with a gsp_id in the metadata; extract it
        .loc[lambda df: df["metadata"].apply(lambda x: "gsp_id" in x)]
        .assign(gsp_id=lambda df: df["metadata"].apply(lambda x: int(x["gsp_id"]["number_value"])))
        .set_index("gsp_id", drop=False, inplace=False)
    )
    return locations_df.apply(lambda row: row["location_uuid"], axis=1).to_dict()


async def create_forecaster_if_not_exists(
    client: dp.DataPlatformDataServiceStub,
    model_tag: str = "pvnet_app",
) -> dp.Forecaster:
    """Create the current forecaster if it does not exist."""
    name = model_tag.replace("-", "_")
    app_version = version("pvnet_app")

    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    if len(list_forecasters_response.forecasters) > 0:
        filtered_forecasters = [
            f for f in list_forecasters_response.forecasters if f.forecaster_version == app_version
        ]
        if len(filtered_forecasters) == 1:
            # Forecaster exists, return it
            return filtered_forecasters[0]
        else:
            # Forecaster version does not exist, update it
            update_forecaster_request = dp.UpdateForecasterRequest(
                name=name,
                new_version=app_version,
            )
            update_forecaster_response = await client.update_forecaster(update_forecaster_request)
            return update_forecaster_response.forecaster
    else:
        # Forecaster does not exist, create it
        create_forecaster_request = dp.CreateForecasterRequest(
            name=name,
            version=app_version,
        )
        create_forecaster_response = await client.create_forecaster(create_forecaster_request)
        return create_forecaster_response.forecaster


async def make_forecaster_adjuster(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    init_time_utc: datetime,
    forecast_values: list[dp.CreateForecastRequestForecastValue],
    model_tag: str,
    forecaster: dp.Forecaster,
) -> dp.CreateForecastRequest:
    """Make a forecaster adjuster based on week average deltas."""
    # get delta values
    deltas_request = dp.GetWeekAverageDeltasRequest(
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        pivot_timestamp_utc=init_time_utc.replace(tzinfo=UTC),
        forecaster=forecaster,
        observer_name="pvlive_day_after",
    )
    deltas_response = await client.get_week_average_deltas(deltas_request)
    deltas = deltas_response.deltas

    # adjust the current forecast values
    new_forecast_values = []
    for fv in forecast_values:
        horizon_mins = fv.horizon_mins
        delta_fractions = [d.delta_fraction for d in deltas if d.horizon_mins == horizon_mins]
        delta_fraction = delta_fractions[0] if len(delta_fractions) > 0 else 0

        # limit adjusted fractions to 10% of fv.p50_fraction
        max_delta = 0.1 * fv.p50_fraction
        if delta_fraction > max_delta:
            delta_fraction = max_delta
        elif delta_fraction < -max_delta:
            delta_fraction = -max_delta

        # limit adjust to 1000 MW
        list_location_response = await client.list_locations(
                dp.ListLocationsRequest(
                    location_type_filter=dp.LocationType.NATION,
                    energy_source_filter=dp.EnergySource.SOLAR,
                ),
            )
        locations = list_location_response.locations
        location = next(loc for loc in locations if loc.location_uuid == location_uuid)
        capacity_mw = location.effective_capacity_watts / 1_000_000.0
        max_delta_absolute = 1000.0 / capacity_mw
        if delta_fraction > max_delta_absolute:
            delta_fraction = max_delta_absolute
        elif delta_fraction < -max_delta_absolute:
            delta_fraction = -max_delta_absolute

        # delta values are forecast - observed, so we need to subtract
        new_p50 = max(0.0, min(1.0, fv.p50_fraction - delta_fraction))

        # adjust p10 and p90s
        new_other_statistics = {}
        for key, val in fv.other_statistics_fractions.items():
            new_val = max(0.0, min(1.0, val - delta_fraction))
            new_other_statistics[key] = new_val

        new_forecast_values.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=fv.horizon_mins,
                p50_fraction=new_p50,
                metadata=fv.metadata,
                other_statistics_fractions=new_other_statistics,
            ),
        )

    # make new forecast
    forecaster = await create_forecaster_if_not_exists(
        client=client,
        model_tag=model_tag + "_adjuster",
    )

    # make forecast
    adjusted_forecast_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc.replace(tzinfo=UTC),
        values=new_forecast_values,
    )

    return adjusted_forecast_request
