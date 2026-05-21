"""Functions to save forecasts to the data platform."""

import asyncio
import itertools
import logging
import os
from datetime import UTC, datetime
from importlib.metadata import version

import betterproto
import fsspec
import pandas as pd
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp

logger = logging.getLogger(__name__)


async def save_forecast_to_data_platform(
    forecast_normed_da: xr.DataArray,
    locations_gsp_uuid_map: dict[int, str],
    model_tag: str,
    init_time_utc: datetime,
    client: dp.DataPlatformDataServiceStub,
) -> None:
    """Save forecast DataArray to data platform.

    We do the following steps:
    1. Get the metadata for the forecast
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

    # 1. get metadata for the forecast
    metadata = await get_metadata_for_forecast(
        client=client, location_uuid=locations_gsp_uuid_map[0],
    )

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
            model_tag=model_tag,
        )

        # 5. Save to data platform
        forecast_request = dp.CreateForecastRequest(
            forecaster=forecaster,
            location_uuid=locations_gsp_uuid_map[int(gsp_id)],
            energy_source=dp.EnergySource.SOLAR,
            init_time_utc=init_time_utc.replace(tzinfo=UTC),
            values=forecast_values,
            metadata=metadata,
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
                metadata=metadata,
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
    model_tag: str = "",
) -> list[dp.CreateForecastRequestForecastValue]:
    """Convert a DataArray for a single GSP to a list of ForecastValue objects.

    Args:
        gsp_normed_da: Normalized DataArray for a single GSP
        init_time_utc: Forecast initialization time
        model_tag: Name of the model (used for logging)
    """
    # create horizon mins
    target_datetime_utc = pd.to_datetime(gsp_normed_da.target_datetime_utc.values)
    horizons_mins = (target_datetime_utc - init_time_utc).total_seconds() / 60
    horizons_mins = horizons_mins.astype(int)

    gsp_id = int(gsp_normed_da.gsp_id.values)

    # Reduce singular dimensions
    gsp_normed_da = gsp_normed_da.squeeze(drop=True)
    p50s = gsp_normed_da.sel(output_label="forecast_fraction").values.astype(float)
    p10s = gsp_normed_da.sel(output_label="forecast_fraction_plevel_10").values.astype(float)
    p90s = gsp_normed_da.sel(output_label="forecast_fraction_plevel_90").values.astype(float)

    forecast_values = []
    for h, p50, p10, p90 in zip(horizons_mins, p50s, p10s, p90s, strict=True):
        if p90 >= 1.1:
            logger.warning(
                f"p90 value {p90} exceeds 1.1 for model={model_tag}, gsp_id={gsp_id}, "
                f"horizon_mins={h}; clamping to 1.1",
            )
            p90 = 1.1
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
    # we are not using app version any more,
    # this is stored in the forecast metadata
    version = "2.8.0"

    list_forecasters_request = dp.ListForecastersRequest(
        forecaster_names_filter=[name],
    )
    list_forecasters_response = await client.list_forecasters(list_forecasters_request)

    if len(list_forecasters_response.forecasters) > 0:
        filtered_forecasters = [
            f for f in list_forecasters_response.forecasters if f.forecaster_version == version
        ]
        if len(filtered_forecasters) == 1:
            # Forecaster exists, return it
            return filtered_forecasters[0]
        else:
            # Forecaster version does not exist, update it
            update_forecaster_request = dp.UpdateForecasterRequest(
                name=name,
                new_version=version,
            )
            update_forecaster_response = await client.update_forecaster(update_forecaster_request)
            return update_forecaster_response.forecaster
    else:
        # Forecaster does not exist, create it
        create_forecaster_request = dp.CreateForecasterRequest(
            name=name,
            version=version,
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
    metadata: Struct | None = None,
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

        # get location
        location = await client.get_location(
            dp.GetLocationRequest(
                location_uuid=location_uuid,
                energy_source=dp.EnergySource.SOLAR,
                include_geometry=False,
            ),
        )
        capacity_mw = location.effective_capacity_watts / 1_000_000.0

        # limit adjuster
        delta_fraction = limit_adjuster(
            delta_fraction=delta_fraction, value_fraction=fv.p50_fraction, capacity_mw=capacity_mw,
        )

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
        model_tag=model_tag + "_adjust",
    )

    # make forecast
    adjusted_forecast_request = dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc.replace(tzinfo=UTC),
        values=new_forecast_values,
        metadata=metadata,
    )

    return adjusted_forecast_request


def limit_adjuster(delta_fraction: float, value_fraction: float, capacity_mw: float) -> float:
    """Limit the adjuster to 10% of forecast and max 1000 MW."""
    # limit adjusted fractions to 10% of fv.p50_fraction
    max_delta = 0.1 * value_fraction
    if delta_fraction > max_delta:
        delta_fraction = max_delta
    elif delta_fraction < -max_delta:
        delta_fraction = -max_delta

    # limit adjust to 1000 MW
    max_delta_absolute = 1000.0 / capacity_mw
    if delta_fraction > max_delta_absolute:
        delta_fraction = max_delta_absolute
    elif delta_fraction < -max_delta_absolute:
        delta_fraction = -max_delta_absolute

    return delta_fraction


async def get_metadata_for_forecast(
    client: dp.DataPlatformDataServiceStub, location_uuid: str,
) -> Struct:
    """Get metadata for the forecast."""
    metadata = {"app_version": Value(string_value=version("pvnet_app"))}

    # add gsp last updated time
    gsp_request = dp.GetLatestObservationsRequest(
        location_uuids=[location_uuid],
        energy_source=dp.EnergySource.SOLAR,
        observer_name="pvlive_in_day",
    )
    gsp_last_updated = await client.get_latest_observations(gsp_request)
    if len(gsp_last_updated.observations) > 0:
        metadata["gsp_last_updated"] \
            = Value(string_value=gsp_last_updated.observations[-1].timestamp_utc.isoformat())

    # add nwp and satellite last updated time, load file from s3 if exists
    env_vars = ["NWP_ECMWF_ZARR_PATH",
                "NWP_UKV_ZARR_PATH",
                "SATELLITE_ZARR_PATH",
                "SATELLITE_15_ZARR_PATH"]
    for env_var in env_vars:
        file = os.getenv(env_var)
        if file is not None:
            try:
                fs = fsspec.open(f"{file}/.zattrs").fs
                if "zip" in file:
                    modified_date = fs.modified(file)
                else:
                    modified_date = fs.modified(f"{file}/.zattrs")

                name = env_var.lower().replace("_zarr_path", "")
                metadata[f"{name}_last_modified"] = Value(string_value=modified_date.isoformat())
            except Exception as e:
                logger.debug(f"Could not get metadata for {env_var}: {e}")

    metadata = Struct(fields=metadata)
    return metadata

