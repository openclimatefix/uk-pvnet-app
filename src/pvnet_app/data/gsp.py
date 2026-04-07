"""Functions to get GSP data from the database or data platform."""
import asyncio
import itertools
from importlib.resources import files

import betterproto
import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from ocf import dp
from grpclib.client import Channel


def get_gsp_locations() -> pd.DataFrame:
    """Load the GSP locations metadata."""
    gsp_coordinates_path = files("pvnet_app.data").joinpath("uk_gsp_locations_20250109.csv")
    return pd.read_csv(gsp_coordinates_path, index_col="gsp_id")


def get_gsp_capacities(
    db_connection: DatabaseConnection,
    gsp_ids: list[int],
    t0: pd.Timestamp,
) -> pd.Series:
    """Get GSP capacities from the database.

    Args:
        db_connection: Database connection object
        gsp_ids: List of GSP IDs to get capacities for
        t0: Reference timestamp for getting capacities
    """
    with db_connection.get_session() as session:
        # Get GSP capacities
        all_capacities = get_latest_gsp_capacities(
            session=session,
            gsp_ids=gsp_ids,
            datetime_utc=t0 - pd.Timedelta(days=2),
        )

    # Do basic sanity checking
    if np.isnan(all_capacities).any():
        raise ValueError("Capacities contain NaNs")

    if len(all_capacities)!=len(gsp_ids):
        raise ValueError(
            f"Capacities length ({len(all_capacities)}) "
            f"does not match GSP IDs length ({len(gsp_ids)})",
        )

    return all_capacities


async def get_gsp_capacities_from_dp(
    client: dp.DataPlatformDataServiceStub,
    gsp_ids: list[int],
) -> pd.Series:
    """Get GSP capacities from the data platform.

    Args:
        client: Data platform client
        gsp_ids: List of GSP IDs to get capacities for
    """
    tasks = [
        client.list_locations(
            dp.ListLocationsRequest(
                location_type_filter=loc_type,
                energy_source_filter=dp.EnergySource.SOLAR,
            ),
        )
        for loc_type in [dp.LocationType.GSP, dp.LocationType.NATION]
    ]
    responses = await asyncio.gather(*tasks)

    locations_df = (
        pd.DataFrame.from_dict(
            itertools.chain(
                *[
                    r.to_dict(casing=betterproto.Casing.SNAKE, include_default_values=True)[
                        "locations"
                    ]
                    for r in responses
                ],
            ),
        )
        .loc[lambda df: df["metadata"].apply(lambda x: "gsp_id" in x)]
        .assign(
            gsp_id=lambda df: df["metadata"].apply(
                lambda x: int(x["gsp_id"]["number_value"]),
            ),
            capacity_mwp=lambda df: df["effective_capacity_watts"].astype(float) / 1_000_000.0,
        ).set_index("gsp_id")
    )

    all_capacities = locations_df.loc[gsp_ids, "capacity_mwp"]

    # Do basic sanity checking
    if np.isnan(all_capacities).any():
        raise ValueError("Capacities contain NaNs")

    if len(all_capacities)!=len(gsp_ids):
        raise ValueError(
            f"Capacities length ({len(all_capacities)}) "
            f"does not match GSP IDs length ({len(gsp_ids)})",
        )

    return all_capacities


async def create_null_generation_data(
    db_connection: DatabaseConnection | None,
    dp_address: tuple[str, int] | None,
    t0: pd.Timestamp,
    read_from_data_platform: bool,
) -> xr.Dataset:
    """Create generation-like xarray-data.

    The generation values are all set to NaN. The capacities are loaded from the database.

    Args:
        db_connection: Database connection object
        dp_address: Tuple containing the data platform host and port
        t0: The forecast init-time
        read_from_data_platform: Whether to read capacities from the data platform or the database
    """
    # Load the GSP location data
    df_locs = get_gsp_locations()

    if read_from_data_platform:

        dp_channel = Channel(*dp_address)
        dp_client = dp.DataPlatformDataServiceStub(dp_channel)

        capacities = await get_gsp_capacities_from_dp(
            client=dp_client,
            gsp_ids=df_locs.index.values.tolist(),
        )

        dp_channel.close()
        
    else:
        capacities = get_gsp_capacities(
            db_connection=db_connection,
            gsp_ids=df_locs.index.values.tolist(),
            t0=t0,
        )

    # Generate null genration values
    interval_start = -pd.Timedelta("2D")
    interval_end = pd.Timedelta("3D")
    time_utc = pd.date_range(t0 + interval_start, t0 + interval_end, freq="30min")

    gen_data = np.full((len(time_utc), len(df_locs)), fill_value=-1, dtype=np.float32)


    cap_data = np.tile(capacities,(len(time_utc),1))

    # Conststruct generation dataset
    ds_gen = xr.Dataset(
        data_vars={
            "generation_mw": (("time_utc", "location_id"), gen_data),
            "capacity_mwp": (("time_utc", "location_id"), cap_data),
        },
        coords={
            "time_utc": ("time_utc", time_utc),
            "location_id": ("location_id", df_locs.index.values),
            "longitude": ("location_id", df_locs.longitude.values),
            "latitude": ("location_id", df_locs.latitude.values),
        },
    )

    return ds_gen
