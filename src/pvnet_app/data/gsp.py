from datetime import timedelta

import numpy as np
import pandas as pd
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities

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
    with db_connection.get_session() as session:
        # Get GSP capacities
        all_capacities = get_latest_gsp_capacities(
            session=session,
            gsp_ids=[0]+gsp_ids,
            datetime_utc=t0 - timedelta(days=2),
        )

    # Do basic sanity checking
    if np.isnan(all_capacities).any():
        raise ValueError("Capacities contain NaNs")
        
    national_capacity = all_capacities[0].item()
    gsp_capacities = all_capacities[1:]

    return gsp_capacities, national_capacity