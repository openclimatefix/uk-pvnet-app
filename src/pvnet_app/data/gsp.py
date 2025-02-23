from datetime import timedelta
from typing import Tuple

import pandas as pd
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities
from nowcasting_datamodel.models.base import Base_Forecast

def get_gsp_and_national_capacities(
    db_connection: DatabaseConnection,
    gsp_ids: list[int],
    t0: pd.Timestamp,
) -> Tuple[pd.Series, float]:
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
    with db_connection.get_session() as session:
        # Get GSP capacities
        gsp_capacities = get_latest_gsp_capacities(
            session=session,
            gsp_ids=gsp_ids,
            datetime_utc=t0 - timedelta(days=2),
        )

        # Get national capacity (needed if using summation model)
        national_capacity = get_latest_gsp_capacities(session, [0])[0]

    return gsp_capacities, national_capacity