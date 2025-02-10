import os

import pytest
import pandas as pd
import numpy as np
import xarray as xr
import torch
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast, Base_PV
from nowcasting_datamodel.read.read import get_location
from nowcasting_datamodel.fake import make_fake_me_latest
from nowcasting_datamodel.models import (
    GSPYield,
    LocationSQL,
)

from testcontainers.postgres import PostgresContainer
from datetime import timedelta, timezone


xr.set_options(keep_attrs=True)

@pytest.fixture()
def test_t0():
    return pd.Timestamp.now(tz=None).floor(timedelta(minutes=30))


@pytest.fixture(scope="session")
def engine_url():
    """Database engine, this includes the table creation."""
    with PostgresContainer("postgres:16.1") as postgres:
        url = postgres.get_connection_url()
        os.environ["DB_URL"] = url

        database_connection = DatabaseConnection(url, echo=False)

        engine = database_connection.engine

        # Would like to do this here but found the data
        # was not being deleted when using 'db_connection'
        # database_connection.create_all()
        # Base_PV.metadata.create_all(engine)

        yield url

        # Base_PV.metadata.drop_all(engine)
        # Base_Forecast.metadata.drop_all(engine)

        engine.dispose()


@pytest.fixture()
def db_connection(engine_url):
    database_connection = DatabaseConnection(engine_url, echo=False)

    engine = database_connection.engine
    # connection = engine.connect()
    # transaction = connection.begin()

    # There should be a way to only make the tables once
    # but make sure we remove the data
    database_connection.create_all()
    Base_PV.metadata.create_all(engine)

    yield database_connection

    # transaction.rollback()
    # connection.close()

    Base_PV.metadata.drop_all(engine)
    Base_Forecast.metadata.drop_all(engine)


@pytest.fixture()
def db_session(db_connection, engine_url):
    """Return a sqlalchemy session, which tears down everything properly post-test."""

    with db_connection.get_session() as s:
        s.begin()
        yield s
        s.rollback()


def make_nwp_data(shell_path, varname, test_t0):
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(shell_path)

    # Last init time was at least 8 hours ago and floor to 3-hour interval
    t0_datetime_utc = (test_t0 - timedelta(hours=8)).floor(timedelta(hours=3))
    ds.init_time.values[:] = pd.date_range(
        t0_datetime_utc - timedelta(hours=3 * (len(ds.init_time) - 1)),
        t0_datetime_utc,
        freq=timedelta(hours=3),
    )

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
        

@pytest.fixture
def nwp_ukv_data(test_t0):
    return make_nwp_data(
        shell_path=f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp_ukv_shell.zarr",
        varname="UKV",
        test_t0=test_t0,
    )


@pytest.fixture
def nwp_ecmwf_data(test_t0):
    return make_nwp_data(
        shell_path=f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp_ecmwf_shell.zarr",
        varname="ECMWF_UK",
        test_t0=test_t0,
    )

@pytest.fixture
def config_filename():
    return f"{os.path.dirname(os.path.abspath(__file__))}/test_data/test.yaml"


def make_sat_data(test_t0, delay_mins, freq_mins, small=False):
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.zarr"
    )

    if small:
        # only select 10 by 10
        ds = ds.isel(x_geostationary=slice(0, 10), y_geostationary=slice(0, 10))

    # remove tim dim and expand time dim to be len 36 = 3 hours of 5 minute data
    ds = ds.drop_vars("time")
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

@pytest.fixture()
def sat_5_data(test_t0):
    return make_sat_data(test_t0, delay_mins=10, freq_mins=5)

@pytest.fixture()
def sat_5_data_zero_delay(test_t0):
    return make_sat_data(test_t0, delay_mins=0, freq_mins=5)

@pytest.fixture()
def sat_5_data_delayed(test_t0):
    return make_sat_data(test_t0, delay_mins=120, freq_mins=5)


@pytest.fixture()
def sat_15_data(test_t0):
    return make_sat_data(test_t0, delay_mins=0, freq_mins=15)

@pytest.fixture()
def sat_15_data_small(test_t0):
    return make_sat_data(test_t0, delay_mins=0, freq_mins=15,small=True)


@pytest.fixture()
def gsp_yields_and_systems(db_session, test_t0):
    """Create gsp yields and systems"""

    # GSP data is mostly up to date, but 10 hours delayed
    t0_datetime_utc = test_t0 - timedelta(hours=10)

    # this pv systems has same coordiantes as the first gsp
    gsp_yields = []
    locations = []
    for i in range(0, 318):

        if i == 0:
            installed_capacity_mw = 17000
        else:
            installed_capacity_mw = 17000/318
        
        location_sql: LocationSQL = get_location(
            session=db_session,
            gsp_id=i,
            installed_capacity_mw = installed_capacity_mw,
        )

        # From 3 hours ago to 8.5 hours into future
        for minute in range(-3 * 60, 9 * 60, 30):
            gsp_yield_sql = GSPYield(
                datetime_utc=(t0_datetime_utc + timedelta(minutes=minute)).replace(tzinfo=timezone.utc),
                solar_generation_kw=np.random.randint(low=0, high=1000),
                capacity_mwp=installed_capacity_mw,
            ).to_orm()
            gsp_yield_sql.location = location_sql
            gsp_yields.append(gsp_yield_sql)
            locations.append(location_sql)

    # add to database
    db_session.add_all(gsp_yields)

    db_session.commit()

    return {
        "gsp_yields": gsp_yields,
        "gs_systems": locations,
    }


@pytest.fixture()
def me_latest(db_session):
    metric_values = make_fake_me_latest(session=db_session, model_name="pvnet_v2")
    db_session.add_all(metric_values)
    db_session.commit()
