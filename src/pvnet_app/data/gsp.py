"""Functions to get GSP data from the database."""

import asyncio
import itertools
import logging
import os

import betterproto
import numpy as np
import pandas as pd
from dp_sdk.ocf import dp
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities

logger = logging.getLogger(__name__)


read_data_platform_flag = os.getenv("DATA_PLATFORM_READ_CAPACITIES", "false").lower() == "true"


async def get_gsp_and_national_capacities(
    gsp_ids: list[int],
    t0: pd.Timestamp,
    db_connection: DatabaseConnection | None,
    client: dp.DataPlatformDataServiceStub | None,
    read_data_platform: bool = read_data_platform_flag,
) -> tuple[pd.Series, float]:
    """Get GSP and national capacities from the database.

    Note that one of `db_connection` or `client` must be provided.

    Args:
        db_connection: Database connection object
        gsp_ids: List of GSP IDs to get capacities for
        t0: Reference timestamp for getting capacities
        client: Data platform client
        read_data_platform: Whether to read capacities from data platform or nowcasting database

    Returns:
        Tuple containing:
        - Pandas series of most recent GSP capacities
        - National capacity value
    """
    if read_data_platform:
        logger.info("Reading capacities from data platform")

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
            .assign(
                gsp_id=lambda df: df["metadata"].apply(lambda x: int(x["gsp_id"]["number_value"])),
            )
            .set_index("gsp_id", drop=False, inplace=False)
        )

        # reduce to the columns we need (in mw)
        locations_df = (
            locations_df["effective_capacity_watts"].astype(float) / 1000
        )  # convert to MW

        # only select the gsp_ids we want + national (gsp_id 0)
        locations_df = locations_df.loc[[0, *gsp_ids]]

        # order by index (gsp_id)
        all_capacities_mw = locations_df.sort_index()

    else:
        logger.info("Reading capacities from nowcasting database")

        with db_connection.get_session() as session:
            # Get GSP capacities
            all_capacities_mw = get_latest_gsp_capacities(
                session=session,
                gsp_ids=[0, *gsp_ids],
                datetime_utc=t0 - pd.Timedelta(days=2),
            )

    # Do basic sanity checking
    if np.isnan(all_capacities_mw).any():
        raise ValueError("Capacities contain NaNs")

    national_capacity = all_capacities_mw[0].item()
    gsp_capacities = all_capacities_mw[1:]

    return gsp_capacities, national_capacity
