"""Functions to save forecasts to the data platform."""

import asyncio
import logging

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp

logger = logging.getLogger(__name__)


DATAPLATFORM_MAX_VALUE = 1.09

async def fetch_locations(
    client: dp.DataPlatformDataServiceStub,
) -> dict[int, dp.ListLocationsResponseLocationSummary]:
    """Fetch all UK locations from data platform."""
    # Get the UK national location
    national_locations = (
        await client.list_locations(
            dp.ListLocationsRequest(
                location_type_filter=dp.LocationType.NATION,
                energy_source_filter=dp.EnergySource.SOLAR,
                location_names_filter=["uk"],
            ),
        )
    ).locations

    if len(national_locations) != 1:
        raise ValueError(f"Expected exactly one location for UK nation, got: {national_locations}")

    national_location = national_locations[0]

    # Get all GSPs within the UK nation
    gsp_locations = (
        await client.list_locations(
            dp.ListLocationsRequest(
                location_type_filter=dp.LocationType.GSP,
                energy_source_filter=dp.EnergySource.SOLAR,
                # TODO: We should filter specifically within the UK, but the current locations in
                # the data-platform don't allow this yet. See commented line below for future use.
                #enclosing_location_uuid_filter=national_location.location_uuid,
            ),
        )
    ).locations

    locations_lookup = {0: national_location}
    for loc in gsp_locations:
        gsp_id = int(loc.metadata.fields["gsp_id"].number_value)
        if gsp_id in locations_lookup:
            raise ValueError(f"Duplicate GSP ID {gsp_id} found in locations")
        locations_lookup[gsp_id] = loc

    return locations_lookup


def extract_location_capacities_mwp(
    locations: dict[int, dp.ListLocationsResponseLocationSummary],
) -> dict[int, float]:
    """Extract capacities from location summaries."""
    return {gsp_id: loc.effective_capacity_watts / 1e6 for gsp_id, loc in locations.items()}


async def fetch_or_create_forecaster(
    client: dp.DataPlatformDataServiceStub,
    model_tag: str,
) -> dp.Forecaster:
    """Create the current forecaster if it does not exist."""
    name = model_tag.replace("-", "_")
    # we are not using app version any more,
    # this is stored in the forecast metadata
    version = "2.8.0"

    forecasters = (
        await client.list_forecasters(
            dp.ListForecastersRequest(forecaster_names_filter=[name]),
        )
    ).forecasters

    # Forecaster does not exist, create it
    if len(forecasters) == 0:
        create_forecaster_response = await client.create_forecaster(
            dp.CreateForecasterRequest(name=name, version=version),
        )
        return create_forecaster_response.forecaster

    else:
        filtered_forecasters = [f for f in forecasters if f.forecaster_version == version]

        # Forecaster version does not exist, update it
        if len(filtered_forecasters) == 0:

            update_forecaster_response = await client.update_forecaster(
                dp.UpdateForecasterRequest(name=name, new_version=version),
            )
            return update_forecaster_response.forecaster

        # Forecaster exists
        else:
            return filtered_forecasters[0]


async def build_input_metadata(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    input_s3_paths: dict[str, str],
    app_version: str,
) -> Struct:
    """Get metadata for the forecast."""
    metadata = {"app_version": Value(string_value=app_version)}

    # Add timestamp when ground truths were last updated
    latest_observations = (
        await client.get_latest_observations(
            dp.GetLatestObservationsRequest(
                location_uuids=[location_uuid],
                energy_source=dp.EnergySource.SOLAR,
                observer_name="pvlive_in_day",
            ),
        )
    ).observations

    if len(latest_observations) > 0:
        metadata["gsp_last_updated"] = Value(
            string_value=latest_observations[-1].timestamp_utc.isoformat(),
        )

    # Add timestamp when the NWP and satellite were last updated
    fs = fsspec.filesystem("s3", anon=False)

    for name, s3_path in input_s3_paths.items():
        if s3_path is not None:
            try:
                if s3_path.endswith(".zarr"):
                    modified_date = fs.modified(f"{s3_path}/.zattrs")
                elif s3_path.endswith(".icechunk"):
                    modified_date = fs.modified(f"{s3_path}/refs/branch.main")
                metadata[f"{name}_last_modified"] = Value(string_value=modified_date.isoformat())
            except Exception as e:
                logger.debug(f"Could not get metadata for {name}: {e}")

    return Struct(fields=metadata)


async def build_multi_forecast_creation_request(
    forecast_normed_da: xr.DataArray,
    locations: dict[int, dp.ListLocationsResponseLocationSummary],
    model_tag: str,
    init_time_utc: pd.Timestamp,
    client: dp.DataPlatformDataServiceStub,
    metadata: Struct,
) -> list[dp.CreateForecastRequest]:
    """Save forecast DataArray to data platform.

    Args:
        forecast_normed_da: DataArray of normalized forecasts for all GSPs
        locations: Mapping of GSP IDs to location summaries
        model_tag: the name of the model to saved to the database
        init_time_utc: Forecast initialization time
        client: Data platform client. If None, a new client will be created.
        metadata: Optional metadata to include with the forecast.
    """
    # Fetch the forecaster and adjuster forecaster in parallel
    forecaster, adjuster_forecaster = await asyncio.gather(
        fetch_or_create_forecaster(client=client, model_tag=model_tag),
        fetch_or_create_forecaster(client=client, model_tag=f"{model_tag}_adjust"),
    )

    forecast_requests: list[dp.CreateForecastRequest] = []

    for loc_id in forecast_normed_da.gsp_id.values.tolist():

        request = build_forecast_creation_request(
            forecast_normed_da.sel(gsp_id=loc_id),
            forecaster=forecaster,
            location_uuid=locations[loc_id].location_uuid,
            init_time_utc=init_time_utc,
            metadata=metadata,
        )

        forecast_requests.append(request)

    # Only make adjuster forecasts for national
    da_adjusted_forecast = await calculate_adjusted_forecast(
        client=client,
        location=locations[0],
        init_time_utc=init_time_utc,
        da_forecast=forecast_normed_da.sel(gsp_id=0),
        forecaster=forecaster, # We get the adjuster values for the original forecaster
    )

    request = build_forecast_creation_request(
        gsp_normed_da=da_adjusted_forecast,
        forecaster=adjuster_forecaster,
        location_uuid=locations[0].location_uuid,
        init_time_utc=init_time_utc,
        metadata=metadata,
    )

    forecast_requests.append(request)

    return forecast_requests


def build_forecast_creation_request(
    gsp_normed_da: xr.DataArray,
    forecaster: dp.Forecaster,
    location_uuid: str,
    init_time_utc: pd.Timestamp,
    metadata: Struct | None,
) -> dp.CreateForecastRequest:
    """Build a create-forecast request from a DataArray forecast for a single location.

    Args:
        gsp_normed_da: Normalized DataArray for a single GSP
        forecaster: Forecaster object
        location_uuid: UUID of the location
        init_time_utc: Forecast initialization time
        metadata: Optional metadata to include with the forecast.
    """
    gsp_id = int(gsp_normed_da.gsp_id.values)
    horizons_mins = gsp_normed_da.horizon_mins.values.tolist()
    plevels = ["p10", "p50", "p90"]

    forecast_array = (
        gsp_normed_da.transpose("valid_times_utc", "output_label")
        .sel(output_label=plevels)
    ).values

    forecast_value_requests = []
    for h, row in zip(horizons_mins, forecast_array, strict=False):

        if (row > DATAPLATFORM_MAX_VALUE).any():
            high_plevels = np.array(plevels)[row > DATAPLATFORM_MAX_VALUE].tolist()
            logger.warning(
                f"p-levels={high_plevels} exceed {DATAPLATFORM_MAX_VALUE} for model="
                f"{forecaster.forecaster_name}, gsp_id={gsp_id}, horizon_mins={h}; clipping to "
                f"{DATAPLATFORM_MAX_VALUE}",
            )
            p10, p50, p90 = row.clip(None, DATAPLATFORM_MAX_VALUE)
        else:
            p10, p50, p90 = row

        forecast_value_requests.append(
            dp.CreateForecastRequestForecastValue(
                horizon_mins=h,
                p50_fraction=p50,
                other_statistics_fractions={"p10": p10, "p90": p90},
            ),
        )

    return dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=init_time_utc.tz_localize("UTC").to_pydatetime(),
        values=forecast_value_requests,
        metadata=metadata,
    )

async def fetch_adjuster_values(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    init_time_utc: pd.Timestamp,
    forecaster: dp.Forecaster,
) -> dict[int, float]:
    """Make a forecaster adjuster based on week average deltas."""
    deltas = (
        await client.get_week_average_deltas(
            dp.GetWeekAverageDeltasRequest(
                location_uuid=location_uuid,
                energy_source=dp.EnergySource.SOLAR,
                pivot_timestamp_utc=init_time_utc.tz_localize("UTC").to_pydatetime(),
                forecaster=forecaster,
                observer_name="pvlive_day_after",
            ),
        )
    ).deltas

    return {d.horizon_mins: d.delta_fraction for d in deltas}


def apply_adjuster_values(
    da_forecast: xr.DataArray,
    adjuster_values: dict[int, float],
    effective_capacity_watts: float,
) -> xr.DataArray:
    """Apply adjuster values to a forecast DataArray."""
    adjuster_values_array = np.zeros(len(da_forecast.horizon_mins.values.tolist()), dtype=float)
    for i, h in enumerate(da_forecast.horizon_mins.values.tolist()):
        if h in adjuster_values:
            adjuster_values_array[i] = adjuster_values[h]
        else:
            logger.warning(f"No adjuster value found for horizon_mins={h}; using 0.0 as default")

    # Limit adjuster values to be no more than 1 GW
    frac_1gw = 1e9 / effective_capacity_watts
    adjuster_values_array = np.clip(adjuster_values_array, -frac_1gw, frac_1gw)

    # Limit adjuster values to be no more than 10% of the forecast value
    frac_10pc = da_forecast.sel(output_label="p50").values * 0.1
    adjuster_values_array = np.clip(adjuster_values_array, -frac_10pc, frac_10pc)

    da_adjuster_values = xr.DataArray(
        data=adjuster_values_array,
        dims=["valid_times_utc"],
        coords={"valid_times_utc": da_forecast.valid_times_utc},
    )

    # Adjuster values are the average of (forecast - observed) so we need to subtract
    # Also force the adjusted forecast to be positive by clipping at 0.0
    ds_adjusted_forecast = (da_forecast - da_adjuster_values).clip(0, 1)

    return ds_adjusted_forecast


async def calculate_adjusted_forecast(
    client: dp.DataPlatformDataServiceStub,
    location: dp.ListLocationsResponseLocationSummary,
    init_time_utc: pd.Timestamp,
    da_forecast: xr.DataArray,
    forecaster: dp.Forecaster,
) -> xr.DataArray:
    """Make an adjusted forecast based on week average deltas."""
    adjuster_values = await fetch_adjuster_values(
        client=client,
        location_uuid=location.location_uuid,
        init_time_utc=init_time_utc,
        forecaster=forecaster,
    )

    da_adjusted_forecast = apply_adjuster_values(
        da_forecast=da_forecast,
        adjuster_values=adjuster_values,
        effective_capacity_watts=location.effective_capacity_watts,
    )

    return da_adjusted_forecast


async def write_forecasts_to_data_platform(
    client: dp.DataPlatformDataServiceStub,
    forecasts: dict[str, "xr.DataArray"],
    locations: dict[int, dp.ListLocationsResponseLocationSummary],
    t0: pd.Timestamp,
    input_s3_paths: dict[str, str],
    app_version: str,
) -> None:
    """Build requests and write all model forecasts to the data platform.

    Builds the input metadata once, then builds and writes a forecast request for
    every model and location. All writes are issued concurrently; if any fail, the
    rest still complete and the failures are raised together as an ExceptionGroup.

    Args:
        client: A connected data-platform service client
        forecasts: Normed national + regional forecasts keyed by model name
        locations: Mapping of GSP ID to location summary, including national (0)
        t0: The forecast init-time, as a naive-UTC timestamp
        input_s3_paths: Source data S3 paths, keyed by source name, for metadata
        app_version: The app version to record against the forecasts

    Raises:
        ExceptionGroup: If one or more forecast writes fail
    """
    input_metadata = await build_input_metadata(
        client=client,
        location_uuid=locations[0].location_uuid,
        input_s3_paths=input_s3_paths,
        app_version=app_version,
    )

    all_requests: list[list[dp.CreateForecastRequest]] = await asyncio.gather(
        *(
            build_multi_forecast_creation_request(
                forecast_normed_da=da_normed_forecast,
                locations=locations,
                model_tag=model_name,
                init_time_utc=t0,
                client=client,
                metadata=input_metadata,
            )
            for model_name, da_normed_forecast in forecasts.items()
        ),
    )

    write_results = await asyncio.gather(
        *(client.create_forecast(req) for reqs in all_requests for req in reqs),
        return_exceptions=True,
    )

    errors = [r for r in write_results if isinstance(r, Exception)]
    if errors:
        raise ExceptionGroup("Failed writing forecasts to data platform", errors)
