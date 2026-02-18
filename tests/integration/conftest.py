
import datetime
import time
from uuid import UUID

import pytest_asyncio
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from grpclib.client import Channel
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer


@pytest_asyncio.fixture(scope="module")
async def client():
    """
    Fixture to spin up a PostgreSQL container for the entire test session.
    This fixture uses `testcontainers` to start a fresh PostgreSQL container and provides
    the connection URL dynamically for use in other fixtures.
    """

    # we use a specific postgres image with postgis and pgpartman installed
    # TODO make a release of this, not using logging tag.
    with PostgresContainer(
        "ghcr.io/openclimatefix/data-platform-pgdb:logging",
        username="postgres",
        password="postgres",  # noqa: S106
        dbname="postgres",
        env={"POSTGRES_HOST": "db"},
    ) as postgres:
        database_url = postgres.get_connection_url()
        # we need to get ride of psycopg2, so the go driver works
        database_url = database_url.replace("postgresql+psycopg2", "postgres")
        # we need to change to host.docker.internal so the data platform container can see it
        # https://stackoverflow.com/questions/46973456/docker-access-localhost-port-from-container
        database_url = database_url.replace("localhost", "host.docker.internal")

        with DockerContainer(
            image="ghcr.io/openclimatefix/data-platform:0.14.0",
            env={"DATABASE_URL": database_url},
            ports=[50051],
        ) as data_platform_server:
            time.sleep(1)  # Give some time for the server to start

            port = data_platform_server.get_exposed_port(50051)
            host = data_platform_server.get_container_host_ip()
            channel = Channel(host=host, port=port)
            client = dp.DataPlatformDataServiceStub(channel)

            yield client
            channel.close()


@pytest_asyncio.fixture(scope="module")
async def national_location(client) -> UUID:
      # 1. setup: add location - gsp 0
    metadata = Struct(fields={"gsp_id": Value(number_value=0)})
    create_location_request = dp.CreateLocationRequest(
        location_name="gsp0",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.NATION,
        effective_capacity_watts=1_000_001,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    create_location_response = await client.create_location(create_location_request)
    return create_location_response.location_uuid

@pytest_asyncio.fixture(scope="module")
async def gsp_1_location(client) -> UUID:
    # setup: add location - gsp 1
    metadata = Struct(fields={"gsp_id": Value(number_value=1)})
    create_location_request = dp.CreateLocationRequest(
        location_name="gsp1",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=999_000,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    create_location_response = await client.create_location(create_location_request)
    return create_location_response.location_uuid
