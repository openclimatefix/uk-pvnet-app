import numpy as np
import pandas as pd
import xarray as xr


def make_mock_gsp_data(
    start_datetime: pd.Timestamp,
    end_datime: pd.Timestamp,
    n_gsp: int = 318,
    filename: str = "gsp.zarr",
):
    """Make a mock GSP dataset for data pipeline

    Args:
        start_datetime: Start datetime
        end_datime: End datetime
        n_gsp: Number of GSPs
        filename: Filename to save the data
    """
    times = pd.date_range(start_datetime, end_datime, freq="30min")
    gsp_ids = np.arange(0, n_gsp)
    capacity = np.ones((len(times), len(gsp_ids)))
    generation = np.random.uniform(0, 200, size=(len(times), len(gsp_ids))).astype(np.float32)

    coords = (
        ("datetime_gmt", times),
        ("gsp_id", gsp_ids),
    )

    da_cap = xr.DataArray(
        capacity,
        coords=coords,
    )

    da_gen = xr.DataArray(
        generation,
        coords=coords,
    )

    data_xr = xr.Dataset(
        {"capacity_mwp": da_cap, "installedcapacity_mwp": da_cap, "generation_mw": da_gen}
    )

    # save to gsp.zarr
    data_xr.to_zarr(filename, mode="w")
