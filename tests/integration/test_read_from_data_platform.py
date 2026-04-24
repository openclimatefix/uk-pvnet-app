import datetime

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp

from pvnet_app.data.gsp import get_gsp_capacities_from_dp


@pytest.mark.asyncio(loop_scope="session")
async def test_read_gsp_and_national_capacities_from_dp(
    client: dp.DataPlatformDataServiceStub,
    setup_dp_locations,  # noqa: ARG001 - ensures locations exist before this test
):
    """
    Test reading GSP and national capacities from the Data Platform.

    The setup_dp_locations fixture creates 1 NATION location (gsp_id=0) and
    342 GSP locations (gsp_id=1..342), each with effective_capacity_watts=1_000_000 (1 MW).

    In this test we
    1. request capacities for a subset of GSP ids that were created in setup
    2. check that the national capacity is read from the gsp_id=0 NATION location
    3. check that the GSP capacities are returned in the requested order with the right values
    """
    gsp_ids = [1, 2, 3, 10, 50, 342]

    capacities = await get_gsp_capacities_from_dp(
        client=client,
        gsp_ids=gsp_ids,
    )

    # 3. GSP capacities are a Series indexed by gsp_id, in the order requested
    assert isinstance(capacities, pd.Series)
    assert list(capacities.index) == gsp_ids
    assert (capacities == 1.0).all()


@pytest.mark.asyncio(loop_scope="session")
async def test_read_gsp_and_national_capacities_from_dp_with_custom_capacity(
    client: dp.DataPlatformDataServiceStub,
    setup_dp_locations,  # noqa: ARG001 - ensures observer + locations exist before this test
):
    """
    Test that a newly created GSP location's capacity is returned by
    get_gsp_and_national_capacities_from_dp.

    We pick a gsp_id outside the range created by setup_dp_locations (1..342)
    so the only matching GSP location for that id is the one created here.
    """
    custom_gsp_id = 500
    custom_capacity_watts = 50_000_000  # 50 MW

    metadata = Struct(fields={"gsp_id": Value(number_value=custom_gsp_id)})
    create_location_request = dp.CreateLocationRequest(
        location_name=f"test_read_gsp{custom_gsp_id}",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(20 20)",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=custom_capacity_watts,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    await client.create_location(create_location_request)

    capacities = await get_gsp_capacities_from_dp(
        client=client,
        gsp_ids=[custom_gsp_id],
    )

    assert capacities.loc[custom_gsp_id] == custom_capacity_watts / 1_000_000.0
