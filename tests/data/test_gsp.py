import tempfile

import pandas as pd
import xarray as xr

from pvnet_app.data.gsp import make_mock_gsp_data


def test_make_mock_gsp_data():

    with tempfile.TemporaryDirectory() as tmpdirname:

        make_mock_gsp_data(
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-02"),
            n_gsp=318,
            filename=f"{tmpdirname}/test.zarr",
        )

        # load zarr
        ds = xr.open_zarr(f"{tmpdirname}/test.zarr")

        # check the data# check has the a time dim
        assert "datetime_gmt" in ds.dims
