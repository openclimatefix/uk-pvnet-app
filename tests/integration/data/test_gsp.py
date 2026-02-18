
import pandas as pd
import pytest

from src.pvnet_app.data.gsp import get_gsp_and_national_capacities


@pytest.mark.asyncio(loop_scope="module")
async def test_get_gsp_and_national_capacities(
    client,
    national_location, #noqa ARG001
    gsp_1_location, #noqa ARG001
):

    # lets call the function we are testing
    gsp_capacities, national_capacity = await get_gsp_and_national_capacities(
        gsp_ids=[1],
        t0=pd.Timestamp("2020-01-01T00:00:00Z"),
        db_connection=None,
        client=client,
        read_data_platform=True,
    )
    assert national_capacity == 1_000
    assert gsp_capacities.iloc[0] == 999
