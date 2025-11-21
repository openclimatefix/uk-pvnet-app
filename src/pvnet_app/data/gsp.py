"""Functions to get GSP data from the database."""
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.read.read_gsp import get_latest_gsp_capacities


def get_gsp_locations() -> pd.DataFrame:
    """Load the GSP locations metadata."""
    gsp_coordinates_path = files("pvnet_app.data").joinpath("uk_gsp_locations_20250109.csv")
    return pd.read_csv(gsp_coordinates_path, index_col="gsp_id")


def get_gsp_and_national_capacities(
    db_connection: DatabaseConnection,
    gsp_ids: list[int],
    t0: pd.Timestamp,
) -> pd.Series:
    """Get GSP and national capacities from the database.

    Args:
        db_connection: Database connection object
        gsp_ids: List of GSP IDs to get capacities for
        t0: Reference timestamp for getting capacities
    """
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

    return all_capacities


def create_null_generation_data(db_connection: DatabaseConnection, t0: pd.Timestamp) -> xr.Dataset:
    """Create generation-like xarray-data.

    The generation values are all set to NaN. The capacities are loaded from the database.

    Args:
        db_connection: Database connection object
        t0: The forecast init-time
    """
    # Load the GSP location data
    df_locs = get_gsp_locations()

    # Generate null genration values
    interval_start = -pd.Timedelta("2D")
    interval_end = pd.Timedelta("3D")
    time_utc = pd.date_range(t0 + interval_start, t0 + interval_end, freq="30min")

    gen_data = np.full((len(time_utc), len(df_locs)), fill_value=np.nan)

    # Get capacities from database
    capacities = get_gsp_and_national_capacities(
        db_connection=db_connection,
        gsp_ids=list(df_locs.index.values),
        t0=t0,
    )

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
