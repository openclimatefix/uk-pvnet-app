import datetime
import os
import time
from collections.abc import AsyncIterator, Iterator
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
import xarray as xr
from betterproto.lib.google.protobuf import Struct, Value
from grpclib.client import Channel
from ocf import dp
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import PortWaitStrategy
from testcontainers.postgres import PostgresContainer

from pvnet_app.data.gsp import get_gsp_locations

test_data_dir = os.path.dirname(os.path.abspath(__file__)) + "/test_data"

DATA_PLATFORM_GRPC_PORT = 50051
DATA_PLATFORM_STARTUP_TIMEOUT_SECONDS = 60


@pytest.fixture(scope="session")
def test_t0() -> pd.Timestamp:
    return pd.Timestamp.now(tz=None).floor("30min")


@pytest.fixture(scope="session")
def location_ids() -> list[int]:
    return get_gsp_locations().index.tolist()


@pytest.fixture(scope="session")
def dp_host_and_port() -> Iterator[tuple[str, int]]:
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
                    DATA_PLATFORM_STARTUP_TIMEOUT_SECONDS,
                ),
            )
        ) as data_platform_server:
            time.sleep(2)  # Give extra time for the server to start

            port = data_platform_server.get_exposed_port(DATA_PLATFORM_GRPC_PORT)
            host = data_platform_server.get_container_host_ip()
            yield host, port


@pytest_asyncio.fixture(scope="session")
async def dp_client(
    dp_host_and_port: tuple[str, int],
) -> AsyncIterator[dp.DataPlatformDataServiceStub]:
    """Create a gRPC client connected to the shared Data Platform server."""
    host, port = dp_host_and_port
    async with Channel(host=host, port=port) as channel:
        yield dp.DataPlatformDataServiceStub(channel)


@pytest_asyncio.fixture(scope="session")
async def dp_client_with_locations(
    dp_client: dp.DataPlatformDataServiceStub,
    location_ids: list[int],
) -> dp.DataPlatformDataServiceStub:
    """Set up GSP locations and observer in the shared Data Platform for integration tests."""

    for location_id in location_ids:
        metadata = Struct(fields={"gsp_id": Value(number_value=location_id)})
        location_type = dp.LocationType.NATION if location_id == 0 else dp.LocationType.GSP
        effective_capacity_watts = 15_000_000_000 if location_id == 0 else 1_000_000
        location_name = "uk" if location_id == 0 else f"gsp{location_id}"

        req = dp.CreateLocationRequest(
            location_name=location_name,
            energy_source=dp.EnergySource.SOLAR,
            geometry_wkt="POINT(0 0)",
            location_type=location_type,
            effective_capacity_watts=effective_capacity_watts,
            metadata=metadata,
            valid_from_utc=datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
        )
        await dp_client.create_location(req)

    # Setup observer
    await dp_client.create_observer(dp.CreateObserverRequest(name="pvlive_day_after"))

    return dp_client


def make_nwp_data(shell_path: str, varname: str, init_time: pd.Timestamp) -> xr.Dataset:
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(shell_path).compute()

    ds = ds.assign_coords(init_time=[init_time])

    # This is important to avoid saving errors
    for v in list(ds.coords.keys()):
        if ds.coords[v].dtype == object:
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
def nwp_ukv_data(test_t0: pd.Timestamp) -> xr.Dataset:
    # The init time was at least 8 hours ago and floor to 3-hour interval
    init_time = (test_t0 - pd.Timedelta("8h")).floor("3h")
    return make_nwp_data(
        shell_path=f"{test_data_dir}/nwp_ukv_shell.zarr",
        varname="um-ukv",
        init_time=init_time,
    )


@pytest.fixture(scope="session")
def nwp_ecmwf_data(test_t0: pd.Timestamp) -> xr.Dataset:
    # The init time was at least 8 hours ago and floor to 3-hour interval
    init_time = (test_t0 - pd.Timedelta("8h")).floor("3h")
    return make_nwp_data(
        shell_path=f"{test_data_dir}/nwp_ecmwf_shell.zarr",
        varname="hres-ifs_uk",
        init_time=init_time,
    )


@pytest.fixture(scope="session")
def cloudcasting_data(test_t0: pd.Timestamp) -> xr.Dataset:
    # The init time is the same as test_t0
    return make_nwp_data(
        shell_path=f"{test_data_dir}/nwp_cloudcasting_shell.zarr",
        varname="sat_pred",
        init_time=test_t0,
    )


@pytest.fixture(scope="session")
def config_filename() -> str:
    return f"{test_data_dir}/test.yaml"


def make_sat_data(test_t0: pd.Timestamp, delay_mins: int, freq_mins: int) -> xr.Dataset:
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(f"{test_data_dir}/non_hrv_shell.zarr").compute()

    times = pd.date_range(
        test_t0 - pd.Timedelta(hours=3),
        test_t0 - pd.Timedelta(minutes=delay_mins),
        freq=f"{freq_mins}min",
    )
    ds = ds.expand_dims(time=times)

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.ones([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    return ds


@pytest.fixture(scope="session")
def sat_5_data(test_t0: pd.Timestamp) -> xr.Dataset:
    return make_sat_data(test_t0, delay_mins=10, freq_mins=5)


@pytest.fixture(scope="session")
def sat_5_data_zero_delay(test_t0: pd.Timestamp) -> xr.Dataset:
    return make_sat_data(test_t0, delay_mins=0, freq_mins=5)


@pytest.fixture(scope="session")
def sat_5_data_delayed(test_t0: pd.Timestamp) -> xr.Dataset:
    return make_sat_data(test_t0, delay_mins=120, freq_mins=5)


@pytest.fixture(scope="session")
def sat_15_data(test_t0: pd.Timestamp) -> xr.Dataset:
    return make_sat_data(test_t0, delay_mins=0, freq_mins=15)
