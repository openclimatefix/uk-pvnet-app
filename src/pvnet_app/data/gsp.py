"""Functions to get GSP data from the data platform."""
import asyncio
import itertools
import logging
from importlib.resources import files

import betterproto
import pandas as pd
from ocf import dp

logger = logging.getLogger()


BACKUP_CAPACITIES: pd.Series = pd.read_csv(
    files("pvnet_app.data").joinpath("gsp_backup_capacities_2026_02_04.csv"),
    index_col="gsp_id",
)["capacity_mwp"]


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
