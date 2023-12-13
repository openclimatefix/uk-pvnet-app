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
from datetime import timedelta


xr.set_options(keep_attrs=True)


def time_before_present(dt: timedelta):
    return pd.Timestamp.now(tz=None) - dt


@pytest.fixture(scope="session")
def engine_url():
    """Database engine, this includes the table creation."""
    with PostgresContainer("postgres:14.5") as postgres:
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
    database_connection = DatabaseConnection(engine_url, echo=True)

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


@pytest.fixture
def nwp_data():
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/nwp_shell.zarr"
    )

    # Last init time was at least 2 hours ago and floor to 3-hour interval
    t0_datetime_utc = time_before_present(timedelta(hours=2)).floor(timedelta(hours=3))
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
    ds["UKV"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.xindexes]),
        coords=[ds[c] for c in ds.xindexes],
    )

    # Add stored attributes to DataArray
    ds.UKV.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]
    return ds


@pytest.fixture()
def sat_5_data():
    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.zarr"
    )

    # Change times so they lead up to present. Delayed by 30-60 mins
    t0_datetime_utc = time_before_present(timedelta(minutes=30)).floor(timedelta(minutes=30))
    ds.time.values[:] = pd.date_range(
        t0_datetime_utc - timedelta(minutes=5 * (len(ds.time) - 1)),
        t0_datetime_utc,
        freq=timedelta(minutes=5),
    )

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords]),
        coords=ds.coords,
    )

    # Add stored attributes to DataArray
    ds.data.attrs = ds.attrs["_data_attrs"]
    del ds.attrs["_data_attrs"]

    return ds


@pytest.fixture()
def sat_5_data_delayed(sat_5_data):
    # Set the most recent timestamp to 2 - 2.5 hours ago
    t_most_recent = time_before_present(timedelta(hours=2)).floor(timedelta(minutes=30))
    offset = sat_5_data.time.max().values - t_most_recent
    sat_5_delayed = sat_5_data.copy(deep=True)
    sat_5_delayed["time"] = sat_5_data.time.values - offset
    return sat_5_delayed


@pytest.fixture()
def sat_15_data(sat_5_data):
    freq = timedelta(minutes=15)
    times_15 = pd.date_range(
        pd.to_datetime(sat_5_data.time.min().values).ceil(freq),
        pd.to_datetime(sat_5_data.time.max().values).floor(freq),
        freq=freq,
    )
    return sat_5_data.sel(time=times_15)


@pytest.fixture()
def gsp_yields_and_systems(db_session):
    """Create gsp yields and systems"""

    # GSP data is mostly up to date
    t0_datetime_utc = time_before_present(timedelta(minutes=0)).floor(timedelta(minutes=30))

    # this pv systems has same coordiantes as the first gsp
    gsp_yields = []
    locations = []
    for i in range(0, 318):
        location_sql: LocationSQL = get_location(
            session=db_session,
            gsp_id=i,
            installed_capacity_mw=123.0,
        )

        gsp_yield_sqls = []
        # From 3 hours ago to 8.5 hours into future
        for minute in range(-3 * 60, 9 * 60, 30):
            gsp_yield_sql = GSPYield(
                datetime_utc=t0_datetime_utc + timedelta(minutes=minute),
                solar_generation_kw=np.random.randint(low=0, high=1000),
                capacity_mwp=100,
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
