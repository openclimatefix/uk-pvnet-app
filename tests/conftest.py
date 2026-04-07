import asyncio
import datetime
import os
import time
from datetime import UTC, timedelta
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from dp_sdk.ocf import dp
from grpclib.client import Channel
from grpclib.exceptions import GRPCError, StreamTerminatedError
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.fake import make_fake_me_latest
from nowcasting_datamodel.models import GSPYield, LocationSQL
from nowcasting_datamodel.models.base import Base_Forecast
from nowcasting_datamodel.models.forecast import (
    ForecastSQL,
    ForecastValueLatestSQL,
    ForecastValueSevenDaysSQL,
    ForecastValueSQL,
)
from nowcasting_datamodel.read.read import get_location
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import PortWaitStrategy
from testcontainers.postgres import PostgresContainer

test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/test_data"

DATA_PLATFORM_GRPC_PORT = 50051
DATA_PLATFORM_STARTUP_TIMEOUT_SECONDS = 60

xr.set_options(keep_attrs=True)


@pytest.fixture(scope="session")
def test_t0():
    return pd.Timestamp.now(tz=None).floor(timedelta(minutes=30))


async def wait_for_data_platform_ready(host: str, port: int) -> None:
    """Wait until the Data Platform gRPC endpoint answers a lightweight RPC."""
    deadline = time.time() + DATA_PLATFORM_STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None

    while time.time() < deadline:
        channel = Channel(host=host, port=port)
        client = dp.DataPlatformDataServiceStub(channel)

        try:
            await client.list_forecasters(dp.ListForecastersRequest(), timeout=2.0)
            return
        except (GRPCError, StreamTerminatedError, OSError, TimeoutError) as exc:
            last_error = exc
            await asyncio.sleep(1)
        finally:
            channel.close()

    raise TimeoutError(
        f"Data Platform gRPC endpoint was not ready at {host}:{port} within "
        f"{DATA_PLATFORM_STARTUP_TIMEOUT_SECONDS} seconds"
    ) from last_error


@pytest.fixture(scope="session")
def dp_client():
    """Spin up a single shared Data Platform gRPC server for the entire test session.

    Yields (host, port) only. Callers must create their own Channel+stub within
    their own async event loop to avoid 'Future attached to a different loop' errors.
    Sets DATA_PLATFORM_HOST and DATA_PLATFORM_PORT env vars so app.py connects to it.
    """
    with PostgresContainer(
        f"ghcr.io/openclimatefix/data-platform-pgdb:{version('dp_sdk')}",
        username="postgres",
        password="postgres",  # noqa: S106
        dbname="postgres",
        env={"POSTGRES_HOST": "db"},
    ) as postgres:
        postgres_container = postgres.get_wrapped_container()
        assert postgres_container is not None

        docker_client = postgres.get_docker_client()
        postgres_network = docker_client.network_name(postgres_container.id)
        postgres_ip = docker_client.bridge_ip(postgres_container.id)
        database_url = (
            f"postgres://{postgres.username}:{postgres.password}@"
            f"{postgres_ip}:{postgres.port}/{postgres.dbname}"
        )

        with (
            DockerContainer(
            image=f"ghcr.io/openclimatefix/data-platform:{version('dp_sdk')}",
            env={"DATABASE_URL": database_url},
            ports=[DATA_PLATFORM_GRPC_PORT],
            platform="linux/amd64",
            )
            .with_kwargs(network=postgres_network)
            .waiting_for(
                PortWaitStrategy(DATA_PLATFORM_GRPC_PORT).with_startup_timeout(
                    DATA_PLATFORM_STARTUP_TIMEOUT_SECONDS
                )
            )
        ) as data_platform_server:

            port = data_platform_server.get_exposed_port(DATA_PLATFORM_GRPC_PORT)
            host = data_platform_server.get_container_host_ip()

            asyncio.run(wait_for_data_platform_ready(host=host, port=port))

            # Set env vars so app.py connects to the test container
            os.environ["DATA_PLATFORM_HOST"] = host
            os.environ["DATA_PLATFORM_PORT"] = str(port)

            yield host, port


@pytest_asyncio.fixture(scope="session")
async def setup_dp_locations(dp_client):
    """Set up GSP locations and observer in the shared Data Platform for integration tests."""
    host, port = dp_client
    channel = Channel(host=host, port=port)
    client = dp.DataPlatformDataServiceStub(channel)

    total_gsps = 342
    for i in range(total_gsps + 1):
        metadata = Struct(fields={"gsp_id": Value(number_value=i)})
        location_type = dp.LocationType.NATION if i == 0 else dp.LocationType.GSP

        req = dp.CreateLocationRequest(
            location_name=f"gsp{i}",
            energy_source=dp.EnergySource.SOLAR,
            geometry_wkt="POINT(0 0)",
            location_type=location_type,
            effective_capacity_watts=1_000_000,
            metadata=metadata,
            valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
        )
        await client.create_location(req)

    # Setup observer
    await client.create_observer(dp.CreateObserverRequest(name="pvlive_day_after"))

    channel.close()


@pytest.fixture()
def db_url():
    with PostgresContainer("postgres:16.1") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture()
def db_connection(test_t0, db_url):
    """Database engine, this includes the table creation."""

    database_connection = DatabaseConnection(db_url, echo=False)
    engine = database_connection.engine

    database_connection.create_all()

    with database_connection.get_session() as s:
        populate_db_session_with_input_data(s, test_t0)

    yield database_connection

    # Tear down
    Base_Forecast.metadata.drop_all(engine)

    engine.dispose()


@pytest.fixture()
def db_session(db_connection):
    """Return a sqlalchemy session, which tears down everything properly post-test."""

    with db_connection.get_session() as s:
        yield s

        # Remove forecasts made in the test
        s.query(ForecastValueSevenDaysSQL).delete()
        s.query(ForecastValueLatestSQL).delete()
        s.query(ForecastValueSQL).delete()
        s.query(ForecastSQL).delete()
        s.commit()


def populate_db_session_with_input_data(session, test_t0):
    """Populate a session with input data for testing"""

    num_gsps = 342
    total_capacity_mw = 17_000

    gsp_yields = []
    for i in range(0, num_gsps + 1):
        # Capacity is total capacity for GSP 0. The rest of the GSPs share the capacity evenly
        installed_capacity_mw = total_capacity_mw if i == 0 else total_capacity_mw / num_gsps

        location_sql: LocationSQL = get_location(
            session=session,
            gsp_id=i,
            installed_capacity_mw=installed_capacity_mw,
        )

        # GSP data is mostly up to date, but a bit delayed
        for date in pd.date_range(
            test_t0 - pd.Timedelta("18h"),
            test_t0 - pd.Timedelta("6.5h"),
            freq="30min",
        ):
            gsp_yield_sql = GSPYield(
                datetime_utc=date.to_pydatetime().replace(tzinfo=UTC),
                solar_generation_kw=np.random.randint(low=0, high=installed_capacity_mw * 1000),
                capacity_mwp=installed_capacity_mw,
            ).to_orm()
            gsp_yield_sql.location = location_sql
            gsp_yields.append(gsp_yield_sql)

    # Add recent GSP data to the database
    session.add_all(gsp_yields)

    # Add recent fake forecast error data to the database for the pvnet_v2 model
    metric_values = make_fake_me_latest(session=session, model_name="pvnet_v2")
    session.add_all(metric_values)

    session.commit()


def make_nwp_data(shell_path, varname, init_time):
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(shell_path).compute()

    ds.init_time.values[:] = init_time

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
            ds[v].encoding.clear()

    for v in list(ds.variables.keys()):
        if ds[v].dtype == object:
            ds[v].encoding.clear()

    # Add data to dataset
    ds[varname] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # Add stored attributes to DataArray
    ds[varname].attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture(scope="session")
def nwp_ukv_data(test_t0):
    # The init time was at least 8 hours ago and floor to 3-hour interval
    init_time = (test_t0 - timedelta(hours=8)).floor(timedelta(hours=3))
    return make_nwp_data(
        shell_path=f"{test_data_dir}/nwp_ukv_shell.zarr",
        varname="um-ukv",
        init_time=init_time,
    )


@pytest.fixture(scope="session")
def nwp_ecmwf_data(test_t0):
    # The init time was at least 8 hours ago and floor to 3-hour interval
    init_time = (test_t0 - timedelta(hours=8)).floor(timedelta(hours=3))
    return make_nwp_data(
        shell_path=f"{test_data_dir}/nwp_ecmwf_shell.zarr",
        varname="hres-ifs_uk",
        init_time=init_time,
    )


@pytest.fixture(scope="session")
def cloudcasting_data(test_t0):
    # The init time is the same as test_t0
    return make_nwp_data(
        shell_path=f"{test_data_dir}/nwp_cloudcasting_shell.zarr",
        varname="sat_pred",
        init_time=test_t0,
    )


@pytest.fixture(scope="session")
def config_filename():
    return f"{test_data_dir}/test.yaml"


def make_sat_data(test_t0, delay_mins, freq_mins):
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(f"{test_data_dir}/non_hrv_shell.zarr").compute()

    # Expand time dim to be len 36 = 3 hours of 5 minute data
    n_hours = 3

    # Add times so they lead up to present
    t0_datetime_utc = test_t0 - timedelta(minutes=delay_mins)
    times = pd.date_range(
        t0_datetime_utc - timedelta(hours=n_hours),
        t0_datetime_utc,
        freq=timedelta(minutes=freq_mins),
    )
    ds = ds.expand_dims(time=times)

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.ones([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture(scope="session")
def sat_5_data(test_t0):
    return make_sat_data(test_t0, delay_mins=10, freq_mins=5)


@pytest.fixture(scope="session")
def sat_5_data_zero_delay(test_t0):
    return make_sat_data(test_t0, delay_mins=0, freq_mins=5)


@pytest.fixture(scope="session")
def sat_5_data_delayed(test_t0):
    return make_sat_data(test_t0, delay_mins=120, freq_mins=5)


@pytest.fixture(scope="session")
def sat_15_data(test_t0):
    return make_sat_data(test_t0, delay_mins=0, freq_mins=15)
