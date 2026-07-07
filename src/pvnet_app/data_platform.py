"""Functions to load and save from data-platform."""

import asyncio
import logging

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp

from pvnet_app.adjuster import calculate_adjusted_forecast
from pvnet_app.consts import forecast_version
from pvnet_app.utils import convert_to_utc_datetime

logger = logging.getLogger(__name__)

# Forecast values above this will be clipped before writing to the data-platform. Data-platform
# currently only supports values up to this limit.
DATAPLATFORM_MAX_VALUE: float = 1.09
# Only allow these p-levels to be written to the data-platform
ALLOWED_PLEVELS: tuple[str, ...] = ("p10", "p50", "p90")


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
    locations = (
        await client.list_locations(
            dp.ListLocationsRequest(
                location_type_filter=dp.LocationType.GSP,
                energy_source_filter=dp.EnergySource.SOLAR,
                # TODO: We should filter specifically within the UK, but the current locations in
                # the data-platform don't allow this yet. See commented line below for future use.
                # enclosing_location_uuid_filter=national_location.location_uuid,
            ),
        )
    ).locations

    locations_lookup = {0: national_location}
    for loc in locations:
        location_id = int(loc.metadata.fields["gsp_id"].number_value)
        if location_id in locations_lookup:
            raise ValueError(f"Duplicate GSP ID {location_id} found in locations")
        locations_lookup[location_id] = loc

    return locations_lookup


def extract_location_capacities_mwp(
    locations: dict[int, dp.ListLocationsResponseLocationSummary],
) -> dict[int, float]:
    """Extract capacities in MW from location summaries."""
    return {loc_id: loc.effective_capacity_watts / 1e6 for loc_id, loc in locations.items()}


async def fetch_or_create_forecaster(
    client: dp.DataPlatformDataServiceStub,
    model_tag: str,
) -> dp.Forecaster:
    """Create the current forecaster if it does not exist."""
    forecasters = (
        await client.list_forecasters(
            dp.ListForecastersRequest(forecaster_names_filter=[model_tag]),
        )
    ).forecasters

    # Forecaster does not exist, create it
    if len(forecasters) == 0:
        create_forecaster_response = await client.create_forecaster(
            dp.CreateForecasterRequest(name=model_tag, version=forecast_version),
        )
        return create_forecaster_response.forecaster

    else:
        filtered_forecasters = [f for f in forecasters if f.forecaster_version == forecast_version]

        # Forecaster version does not exist, update it
        if len(filtered_forecasters) == 0:
            update_forecaster_response = await client.update_forecaster(
                dp.UpdateForecasterRequest(name=model_tag, new_version=forecast_version),
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
                else:
                    logger.warning(f"Unknown file type for {name}: {s3_path}; skipping")
                    continue
                metadata[f"{name}_last_modified"] = Value(string_value=modified_date.isoformat())
            except Exception as e:
                logger.warning(f"Could not get metadata for {name}: {e}")

    return Struct(fields=metadata)


async def build_multi_forecast_creation_request(
    da_forecast: xr.DataArray,
    locations: dict[int, dp.ListLocationsResponseLocationSummary],
    model_tag: str,
    init_time_utc: pd.Timestamp,
    client: dp.DataPlatformDataServiceStub,
    metadata: Struct | None,
) -> list[dp.CreateForecastRequest]:
    """Build a list of create-forecast requests for all forecasted locations.

    Args:
        da_forecast: DataArray of normalized forecasts for all locations
        locations: Mapping of location IDs to location summaries
        model_tag: The name of the model to saved to the database
        init_time_utc: Forecast initialization time
        client: A connected data-platform service client
        metadata: Optional metadata to assign to each location forecast
    """
    # Fetch the forecaster and adjuster forecaster in parallel
    forecaster, adjuster_forecaster = await asyncio.gather(
        fetch_or_create_forecaster(client=client, model_tag=model_tag),
        fetch_or_create_forecaster(client=client, model_tag=f"{model_tag}_adjust"),
    )

    forecast_requests: list[dp.CreateForecastRequest] = []

    for loc_id in da_forecast.location_id.values.tolist():
        request = build_forecast_creation_request(
            da_forecast.sel(location_id=loc_id),
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
        da_forecast=da_forecast.sel(location_id=0),
        forecaster=forecaster,  # We get the adjuster values for the original forecaster
    )

    request = build_forecast_creation_request(
        da_forecast=da_adjusted_forecast,
        forecaster=adjuster_forecaster,
        location_uuid=locations[0].location_uuid,
        init_time_utc=init_time_utc,
        metadata=metadata,
    )

    forecast_requests.append(request)

    return forecast_requests


def _build_forecast_value(
    horizon_mins: int,
    pvalues: np.ndarray,
    plevels: list[str],
    forecaster_name: str,
    location_id: int,
) -> dp.CreateForecastRequestForecastValue:
    if (pvalues > DATAPLATFORM_MAX_VALUE).any():
        high_plevels = np.array(plevels)[pvalues > DATAPLATFORM_MAX_VALUE].tolist()
        logger.warning(
            f"p-levels={high_plevels} exceed {DATAPLATFORM_MAX_VALUE} for model="
            f"{forecaster_name}, location_id={location_id}, horizon_mins={horizon_mins}; "
            f"clipping to {DATAPLATFORM_MAX_VALUE}",
        )
        pvalues = pvalues.clip(None, DATAPLATFORM_MAX_VALUE)

    return dp.CreateForecastRequestForecastValue(
        horizon_mins=horizon_mins,
        p50_fraction=pvalues[plevels.index("p50")],
        other_statistics_fractions={
            k: v for k, v in zip(plevels, pvalues, strict=True) if k != "p50"
        },
    )


def build_forecast_creation_request(
    da_forecast: xr.DataArray,
    forecaster: dp.Forecaster,
    location_uuid: str,
    init_time_utc: pd.Timestamp,
    metadata: Struct | None,
) -> dp.CreateForecastRequest:
    """Build a create-forecast request from a DataArray forecast for a single location.

    Args:
        da_forecast: Normalized DataArray for a single location
        forecaster: Forecaster object
        location_uuid: UUID of the location
        init_time_utc: Forecast initialization time
        metadata: Optional metadata to assign to the forecast
    """
    location_id = int(da_forecast.location_id.values)
    horizons_mins = da_forecast.horizon_mins.values.tolist()

    # Filter the p-levels to the allowed set
    plevels = [level for level in ALLOWED_PLEVELS if level in da_forecast.output_label.values]
    forecast_array = (
        da_forecast.sel(output_label=plevels).transpose("valid_times_utc", "output_label")
    ).values

    forecast_value_requests = [
        _build_forecast_value(
            horizon_mins=h,
            pvalues=pvalues,
            plevels=plevels,
            forecaster_name=forecaster.forecaster_name,
            location_id=location_id,
        )
        for h, pvalues in zip(horizons_mins, forecast_array, strict=True)
    ]

    return dp.CreateForecastRequest(
        forecaster=forecaster,
        location_uuid=location_uuid,
        energy_source=dp.EnergySource.SOLAR,
        init_time_utc=convert_to_utc_datetime(init_time_utc),
        values=forecast_value_requests,
        metadata=metadata,
    )


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
                da_forecast=da_normed_forecast,
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
