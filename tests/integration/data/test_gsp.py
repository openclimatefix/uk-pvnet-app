import datetime

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp

from src.pvnet_app.data.gsp import get_gsp_and_national_capacities


@pytest.mark.asyncio(loop_scope="session")
async def test_get_gsp_and_national_capacities(client):
    # 1. setup: add location - gsp 0
    metadata = Struct(fields={"gsp_id": Value(number_value=0)})
    create_location_request = dp.CreateLocationRequest(
        location_name="gsp0",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.NATION,
        effective_capacity_watts=10_002_001,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    _ = await client.create_location(create_location_request)

    # setup: add location - gsp 1
    metadata = Struct(fields={"gsp_id": Value(number_value=1)})
    create_location_request = dp.CreateLocationRequest(
        location_name="gsp1",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=1_003_000,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    _ = await client.create_location(create_location_request)

    # lets call the function we are testing
    gsp_capacities, national_capacity = await get_gsp_and_national_capacities(
        gsp_ids=[1],
        t0=pd.Timestamp("2020-01-01T00:00:00Z"),
        db_connection=None,
        client=client,
        read_data_platform=True,
    )
    assert national_capacity == 10_002  # TODO check this should be rounded (or not)
    assert gsp_capacities.iloc[0] == 1_003
