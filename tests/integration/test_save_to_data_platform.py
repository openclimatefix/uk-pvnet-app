import datetime
import time

import pandas as pd
import pytest
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from grpclib.client import Channel
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer

from src.pvnet_app.save import save_forecast_to_data_platform

def client():
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
        password="postgres",  #noqa: S106
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
            image="ghcr.io/openclimatefix/data-platform:0.11.0",
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


@pytest.mark.asyncio(loop_scope="session")
async def test_save_to_generation_to_data_platform(client):
    """
    Test saving data to the Data Platform.
    This test uses the `data_platform` fixture to ensure that the Data Platform service
    is running and can accept data.
    """
    # setup: add location - gsp 1
    metadata = Struct(fields={"gsp_id": Value(number_value=1)})
    create_location_request = dp.CreateLocationRequest(
        location_name="gsp1",
        energy_source=dp.EnergySource.SOLAR,
        geometry_wkt="POINT(0 0)",
        location_type=dp.LocationType.GSP,
        effective_capacity_watts=1_000_000,
        metadata=metadata,
        valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
    )
    create_location_response = await client.create_location(create_location_request)
    location_uuid = create_location_response.location_uuid

    # setup: make fake data
    fake_data = pd.DataFrame(
        {
            "solar_generation_mw": [0.5] * 24,
            "target_datetime_utc": pd.Timestamp("2025-01-01")
            + pd.timedelta_range(
                start=0,
                periods=24,
                freq="30min",
            ),
        },
    )
    fake_data["gsp_id"] = 1
    fake_data["output_label"] = "forecast_fraction"

    fake_data_p10 = fake_data.copy()
    fake_data_p10["solar_generation_mw"] = [0.3] * 24
    fake_data_p10["output_label"] = "forecast_fraction_plevel_10"

    fake_data_p90 = fake_data.copy()
    fake_data_p90["solar_generation_mw"] = [0.7] * 24
    fake_data_p90["output_label"] = "forecast_fraction_plevel_90"

    fake_data = pd.concat([fake_data, fake_data_p10, fake_data_p90], ignore_index=True)

    fake_data = fake_data.set_index(["target_datetime_utc", "gsp_id", "output_label"])
    fake_data = fake_data.to_xarray().to_dataarray()

    # Test the functyion
    _ = await save_forecast_to_data_platform(
        forecast_normed_da=fake_data,
        locations_gsp_uuid_map={1: location_uuid},
        client=client,
        model_tag="test_model",
        init_time_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
    )

    # check: read from the data platform to check it was saved
    list_forecasters_response = await client.list_forecasters(dp.ListForecastersRequest())
    assert len(list_forecasters_response.forecasters) == 1

    # check: There is a forecast object
    get_latest_forecasts_request = dp.GetLatestForecastsRequest(
        energy_source=dp.EnergySource.SOLAR,
        pivot_timestamp_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
        location_uuid=location_uuid,
    )
    get_latest_forecasts_response = await client.get_latest_forecasts(
        get_latest_forecasts_request,
    )
    assert len(get_latest_forecasts_response.forecasts) == 1
    forecast = get_latest_forecasts_response.forecasts[0]
    assert forecast.forecaster.forecaster_name == "test_model"

    # check: the number of forecast values
    stream_forecast_data_request = dp.StreamForecastDataRequest(
        energy_source=dp.EnergySource.SOLAR,
        location_uuid=location_uuid,
        forecasters=forecast.forecaster,
        time_window=dp.TimeWindow(
            start_timestamp_utc=datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
            end_timestamp_utc=datetime.datetime(2025, 1, 2, tzinfo=datetime.UTC),
        ),
    )
    stream_forecast_data_response = client.stream_forecast_data(
        stream_forecast_data_request,
    )
    count = 0
    async for d in stream_forecast_data_response:
        assert d.p50_fraction == 0.5
        count += 1
    assert count == 24
