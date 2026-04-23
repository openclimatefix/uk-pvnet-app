"""Functions to get GSP data from the database or data platform."""
import asyncio
import itertools
import logging
from importlib.resources import files

import betterproto
import numpy as np
import pandas as pd
from ocf import dp
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities

logger = logging.getLogger()


BACKUP_CAPACITIES: pd.Series = pd.read_csv(
    files("pvnet_app.data").joinpath("gsp_backup_capacities_2026_02_04.csv"),
    index_col="gsp_id",
)["capacity_mwp"]


def get_gsp_and_national_capacities(
    db_connection: DatabaseConnection,
    gsp_ids: list[int],
    t0: pd.Timestamp,
) -> tuple[pd.Series, float]:
    """Get GSP and national capacities from the database.

    Args:
        db_connection: Database connection object
        gsp_ids: List of GSP IDs to get capacities for
        t0: Reference timestamp for getting capacities

    Returns:
        Tuple containing:
        - Pandas series of most recent GSP capacities
        - National capacity value
    """
    try:
        with db_connection.get_session() as session:
            # Get GSP capacities
            all_capacities = get_latest_gsp_capacities(
                session=session,
                gsp_ids=[0, *gsp_ids],
                datetime_utc=t0 - pd.Timedelta(days=2),
            )

        # Do basic sanity checking
        if np.isnan(all_capacities).any():
            raise ValueError("Capacities contain NaNs")

        if len(all_capacities)!=(len(gsp_ids)+1):
            raise ValueError("Not enough capacities returned")

        national_capacity = all_capacities[0].item()
        gsp_capacities = all_capacities[1:]

    except Exception as e:
        logger.error(f"Error in loading GSP capacities from database: {e}")
        logger.warning(
            "We couldnt load all the capacities from the database, "
            "so we are using back up ones from 2026-02-04",
        )
        national_capacity = BACKUP_CAPACITIES.loc[0].item()
        gsp_capacities = BACKUP_CAPACITIES.loc[gsp_ids]

    return gsp_capacities, national_capacity


async def get_gsp_and_national_capacities_from_dp(
    client: dp.DataPlatformDataServiceStub,
    gsp_ids: list[int],
) -> tuple[pd.Series, float]:
    """Get GSP and national capacities from the data platform.

    Args:
        client: Data platform client
        gsp_ids: List of GSP IDs to get capacities for

    Returns:
        Tuple containing:
        - Pandas series of GSP capacities (MWp)
        - National capacity value (MWp)
    """
    try:
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

        missing = [gid for gid in [0, *gsp_ids] if gid not in locations_df.index]
        if missing:
            raise ValueError(f"Missing capacities from data platform for GSP IDs: {missing}")

        if locations_df.loc[[0, *gsp_ids], "capacity_mwp"].isna().any():
            raise ValueError("Capacities from data platform contain NaNs")

        national_capacity = float(locations_df.loc[0, "capacity_mwp"])
        gsp_capacities = locations_df.loc[gsp_ids, "capacity_mwp"]

    except Exception as e:
        logger.error(f"Error in loading GSP capacities from data platform: {e}")
        logger.warning(
            "We couldnt load all the capacities from the data platform, "
            "so we are using back up ones from 2026-02-04",
        )
        national_capacity = BACKUP_CAPACITIES.loc[0].item()
        gsp_capacities = BACKUP_CAPACITIES.loc[gsp_ids]

    return gsp_capacities, national_capacity
