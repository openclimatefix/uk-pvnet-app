"""Functions to get GSP data from the database."""
import logging
from importlib.resources import files

import numpy as np
import pandas as pd
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
