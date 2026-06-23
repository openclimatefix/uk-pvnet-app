"""Functions to get GSP data from the data platform."""
from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr


def get_gsp_locations() -> pd.DataFrame:
    """Load the GSP locations metadata."""
    return pd.read_csv(
        files("pvnet_app.data").joinpath("uk_gsp_locations_20260209_no_shetlands.csv"), 
        index_col="gsp_id",
    )


def create_null_generation_data(t0: pd.Timestamp, capacities_mwp: dict[int, float]) -> xr.Dataset:
    """Create generation-like xarray-data filled with value -1.

    Args:
        t0: The forecast init-time
        capacities_mwp: A dictionary mapping location IDs to their capacities in MWp
    """
    # Load the GSP location data
    df_locs = get_gsp_locations()

    capacities_array = np.array(
        [capacities_mwp[gsp_id] for gsp_id in df_locs.index.tolist()],
        dtype=np.float32,
    )

    # Generate null generation values
    time_utc = pd.date_range(t0 - pd.Timedelta("2D"), t0 + pd.Timedelta("3D"), freq="30min")
    gen_data = np.full((len(time_utc), len(df_locs)), fill_value=-1, dtype=np.float32)
    cap_data = np.tile(capacities_array, (len(time_utc), 1))

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
