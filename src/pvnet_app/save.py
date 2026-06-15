"""Functions to save forecasts to the data platform."""

import asyncio
import logging
import os
from importlib.metadata import version

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp

logger = logging.getLogger(__name__)


async def fetch_locations(
    client: dp.DataPlatformDataServiceStub
) -> dict[int, dp.ListLocationsResponseLocationSummary]:
    """Fetch all UK locations from data platform."""

    # Get the UK nation location
    nat_resp = await client.list_locations(
        dp.ListLocationsRequest(
            location_type_filter=dp.LocationType.NATION,
            energy_source_filter=dp.EnergySource.SOLAR,
            location_names_filter=["uk"],
        )
    )

    if len(nat_resp.locations) != 1:
        raise ValueError(f"Expected exactly one location for UK nation, got: {nat_resp.locations}")

    national_location = nat_resp.locations[0]
    
    # Get all GSPs within the UK nation
    gsp_resp = await client.list_locations(
        dp.ListLocationsRequest(
            location_type_filter=dp.LocationType.GSP,
            energy_source_filter=dp.EnergySource.SOLAR,
            # TODO: We should filter specifically within the UK, but the current locations in the 
            # data platform don't allow this yet. See commented line below for future use.
            #enclosing_location_uuid_filter=national_location.location_uuid,
        )
    )

    locations_lookup = {0: national_location}
    for loc in gsp_resp.locations:
        gsp_id = int(loc.metadata.fields["gsp_id"].number_value)
        if gsp_id in locations_lookup:
            raise ValueError(f"Duplicate GSP ID {gsp_id} found in locations")
        locations_lookup[gsp_id] = loc

    return locations_lookup


async def fetch_or_create_forecaster(
    client: dp.DataPlatformDataServiceStub,
    model_tag: str,
) -> dp.Forecaster:
    """Create the current forecaster if it does not exist."""
    name = model_tag.replace("-", "_")
    # we are not using app version any more,
    # this is stored in the forecast metadata
    version = "2.8.0"

    list_forecasters_response = await client.list_forecasters(
        dp.ListForecastersRequest(forecaster_names_filter=[name])
    )

    # Forecaster does not exist, create it
    if len(list_forecasters_response.forecasters) == 0:
        create_forecaster_response = await client.create_forecaster(
            dp.CreateForecasterRequest(name=name, version=version)
        )
        return create_forecaster_response.forecaster

    else:
        filtered_forecasters = [
            f for f in list_forecasters_response.forecasters if f.forecaster_version == version
        ]

        # Forecaster version does not exist, update it
        if len(filtered_forecasters) == 0:
            
            update_forecaster_response = await client.update_forecaster(
                dp.UpdateForecasterRequest(name=name, new_version=version)
            )
            return update_forecaster_response.forecaster
        
        # Forecaster exists
        else:
            return filtered_forecasters[0]


async def build_input_metadata(
    client: dp.DataPlatformDataServiceStub, 
    location_uuid: str,
) -> Struct:
    """Get metadata for the forecast."""
    metadata = {"app_version": Value(string_value=version("pvnet_app"))}

    # Add timestamp when ground truths were last updated
    gsp_last_updated = await client.get_latest_observations(
        dp.GetLatestObservationsRequest(
            location_uuids=[location_uuid],
            energy_source=dp.EnergySource.SOLAR,
            observer_name="pvlive_in_day",
        )
    )
    if len(gsp_last_updated.observations) > 0:
        metadata["gsp_last_updated"] = Value(
            string_value=gsp_last_updated.observations[-1].timestamp_utc.isoformat(),
        )

    # Add timestamp when the NWP and satellite were last updated
    fs = fsspec.filesystem("s3", anon=False)
    input_sources = {
        "nwp_ecmwf": "NWP_ECMWF_ZARR_PATH",
        "nwp_ukv": "NWP_UKV_ZARR_PATH",
        "satellite": "SATELLITE_ICECHUNK_PATH_5",
        "satellite_15": "SATELLITE_ICECHUNK_PATH_15",
    }
    for name, env_var in input_sources.items():
        path = os.getenv(env_var)
        if path is not None:
            try:
                if path.endswith(".zarr"):
                    modified_date = fs.modified(f"{path}/.zattrs")
                elif path.endswith(".icechunk"):
                    modified_date = fs.modified(f"{path}/refs/branch.main")
                metadata[f"{name}_last_modified"] = Value(string_value=modified_date.isoformat())
            except Exception as e:
                logger.debug(f"Could not get metadata for {env_var}: {e}")

    return Struct(fields=metadata)


async def build_multi_forecast_creation_request(
    forecast_normed_da: xr.DataArray,
    locations: dict,
    model_tag: str,
    init_time_utc: pd.Timestamp,
    client: dp.DataPlatformDataServiceStub,
    metadata: Struct,
) -> list[dp.CreateForecastRequest]:
    """Save forecast DataArray to data platform.

    Args:
        forecast_normed_da: DataArray of normalized forecasts for all GSPs
        locations_gsp_uuid_map: Mapping of GSP IDs to location UUIDs
        model_tag: the name of the model to saved to the database
        init_time_utc: Forecast initialization time
        client: Data platform client. If None, a new client will be created.
        metadata: Optional metadata to include with the forecast.
    """

    # Fetch the forecaster and adjuster forecaster in parallel
    forecaster, adjuster_forecaster = await asyncio.gather(
        fetch_or_create_forecaster(client=client, model_tag=model_tag),
        fetch_or_create_forecaster(client=client, model_tag=f"{model_tag}_adjust")
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
    """Convert a DataArray for a single GSP to a list of ForecastValue objects.

    Args:
        gsp_normed_da: Normalized DataArray for a single GSP
        forecaster: Forecaster object
        location_uuid: UUID of the location
        init_time_utc: Forecast initialization time
        metadata: Optional metadata to include with the forecast.
    """

    gsp_id = int(gsp_normed_da.gsp_id.values)

    p50s = gsp_normed_da.sel(output_label="p50").values
    p10s = gsp_normed_da.sel(output_label="p10").values
    p90s = gsp_normed_da.sel(output_label="p90").values
    horizons_mins = gsp_normed_da.horizon_mins.values.tolist()

    forecast_values = []
    for h, p50, p10, p90 in zip(horizons_mins, p50s, p10s, p90s, strict=True):
        if p90 > 1.1:
            logger.warning(
                f"p90 value {p90} exceeds 1.1 for model={forecaster.forecaster_name}, "
                f"gsp_id={gsp_id}, horizon_mins={h}; clamping to 1.1",
            )
            p90 = 1.1
        
        forecast_values.append(
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
        values=forecast_values,
        metadata=metadata,
    )


async def fetch_adjuster_values(
    client: dp.DataPlatformDataServiceStub,
    location_uuid: str,
    init_time_utc: pd.Timestamp,
    forecaster: dp.Forecaster,
) -> dict[int, float]:
    """Make a forecaster adjuster based on week average deltas."""
    
    deltas_response = await client.get_week_average_deltas(
        dp.GetWeekAverageDeltasRequest(
            location_uuid=location_uuid,
            energy_source=dp.EnergySource.SOLAR,
            pivot_timestamp_utc=init_time_utc.tz_localize("UTC").to_pydatetime(),
            forecaster=forecaster,
            observer_name="pvlive_day_after",
        )
    )
    
    return {d.horizon_mins: d.delta_fraction for d in deltas_response.deltas}


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
    location,
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