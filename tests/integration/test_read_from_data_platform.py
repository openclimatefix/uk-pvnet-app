import datetime

import pandas as pd
import pytest
import pytest_asyncio
from betterproto.lib.google.protobuf import Struct, Value
from ocf import dp
from grpclib.client import Channel

from pvnet_app.data.gsp import (
    BACKUP_CAPACITIES,
    get_gsp_and_national_capacities_from_dp,
)


@pytest_asyncio.fixture(scope="session")
async def client(dp_client):
    """
    Fixture to create a gRPC client connected to the shared Data Platform server.
    """
    host, port = dp_client
    channel = Channel(host=host, port=port)
    client_stub = dp.DataPlatformDataServiceStub(channel)

    yield client_stub
    channel.close()


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

    gsp_capacities, national_capacity = await get_gsp_and_national_capacities_from_dp(
        client=client,
        gsp_ids=gsp_ids,
    )

    # 2. national capacity comes from the NATION location for gsp_id=0 (1 MW)
    assert national_capacity == 1.0

    # 3. GSP capacities are a Series indexed by gsp_id, in the order requested
    assert isinstance(gsp_capacities, pd.Series)
    assert list(gsp_capacities.index) == gsp_ids
    assert (gsp_capacities == 1.0).all()


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

    gsp_capacities, national_capacity = await get_gsp_and_national_capacities_from_dp(
        client=client,
        gsp_ids=[custom_gsp_id],
    )

    assert national_capacity == 1.0
    assert gsp_capacities.loc[custom_gsp_id] == custom_capacity_watts / 1_000_000.0


@pytest.mark.asyncio(loop_scope="session")
async def test_read_gsp_and_national_capacities_fallback(
    client: dp.DataPlatformDataServiceStub,
    setup_dp_locations,  # noqa: ARG001 - ensures locations exist before this test
):
    """
    Test that the fallback to BACKUP_CAPACITIES is used when a missing GSP ID is requested.
    
    If we request a GSP ID that is not found on the data platform, the function 
    should internally catch the ValueError and return the capacities from the backup CSV.
    """
    # GSP ID 348 is in the backup CSV but not created by setup_dp_locations (which creates 1..342)
    missing_gsp_id = 348
    
    gsp_capacities, national_capacity = await get_gsp_and_national_capacities_from_dp(
        client=client,
        gsp_ids=[missing_gsp_id],
    )

    # When fallback occurs, national capacity comes from the backup CSV
    assert national_capacity == BACKUP_CAPACITIES.loc[0].item()
    
    # And GSP capacities come from the backup CSV
    assert gsp_capacities.loc[missing_gsp_id] == BACKUP_CAPACITIES.loc[missing_gsp_id]
