"""This script downloads an archive data from the live production system.

The data is processed equivalently to how it is processed in the uk-pvnet-app so that a recent
live backtest can be performed.
"""

import os
from datetime import UTC

import dask
import fsspec
import icechunk
import pandas as pd
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from pvlive_api import PVLive
from s3fs import S3FileSystem
from sqlalchemy import text
from tqdm import tqdm

from pvnet_app.data.nwp import ECMWFDownloader, UKVDownloader

# -------------------------------------------------------------
# USER VARIABLES

start_date = "2025-10-14 00:00"
end_date = "2025-11-10 00:00"

start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

ecmwf_s3_dir = "s3://nowcasting-nwp-development/ecmwf/data"
ukv_s3_dir = "s3://nowcasting-nwp-development/data-metoffice"
cloud_s3_dir = "s3://nowcasting-sat-development/cloudcasting_forecast"
sat_s3_icechunk = "s3://nowcasting-sat-development/rss/data/rss_uk3000m.icechunk"

local_raw_download_dir = "/home/james/tmp/live_inputs_raw"
local_output_dir = "/mnt/storage_u2_30tb_a/uk_live_data"

# -------------------------------------------------------------
# FUNCTIONS

@dask.delayed
def download_zarr(fs: S3FileSystem, s3_path: str, local_dir: str) -> None:
    """Lazily download a file from s3.

    Args:
        fs: The fsspec filesystem
        s3_path: The s3 path of the zarr
        local_dir: The local root directory to save the zarr to
    """
    zarr_name = os.path.basename(s3_path)
    fs.get(s3_path, f"{local_dir}/{zarr_name}", recursive=True)


def get_nwp_dataset(
    s3_dir: str,
    datetime_fmt: str,
    local_dir: str,
    start_date: pd.Timestamp,
    end_date:pd.Timestamp,
) -> xr.Dataset:
    """Download a collection of zarrs from s3 and open as a xarray dataset.

    The files are downloaded to a supplied location on local disk before being opened. Files
    that already exist in this location will not be redownloaded. We filter the zarrs to ones which
    have a filename which is between the supplied start and end date.

    Args:
        s3_dir: Where to search for the zarrs
        datetime_fmt: The datetime format that zarrs are saved under. e.g. r"%Y%m%d%H" for zarr
            which is named like 2025120100.zarr
        local_dir: Where to save the zarrs to in the local filesystem
        start_date: The start datetime used to filter the zarrs
        end_date: The end datetime used to filter the zarrs
    """
    if datetime_fmt not in [r"%Y%m%d%H", r"%Y-%m-%dT%H:%M"]:
        raise ValueError("This function has not been tested with the datetime format provided")

    # Convert the datetime format into a glob pattern
    glob_pattern = datetime_fmt
    for s, n in [("Y", 4), ("m", 2), ("d", 2), ("H", 2), ("M", 2)]:
        glob_pattern = glob_pattern.replace(r"%"+s, "[0-9]"*n)

    # Find the zarr files
    fs = fsspec.filesystem("s3")
    zarr_paths = fs.glob(f"{s3_dir}/{glob_pattern}.zarr")

    # Extract the timestamp from each path
    zarr_datetimes = [os.path.basename(p).removesuffix(".zarr") for p in zarr_paths]
    zarr_datetimes = pd.to_datetime(zarr_datetimes, format=datetime_fmt)

    # Filter to zarrs with datetime between the start and end dates
    zarr_paths = [
        path for path, dt in zip(zarr_paths, zarr_datetimes, strict=False)
        if (start_date <= dt <= end_date)
    ]

    # Filter out files already downloaded
    required_zarr_paths = [
        p for p in zarr_paths if not os.path.exists(f"{local_dir}/{os.path.basename(p)}")
    ]

    # Download the files if required
    if len(required_zarr_paths)>0:

        # Download in parallel
        tasks = [download_zarr(fs, s3_path, local_dir) for s3_path in required_zarr_paths]
        with ProgressBar():
            _ = dask.compute(*tasks, scheduler="threads")

    # Open the datasets
    local_zarr_paths = [f"{local_dir}/{os.path.basename(p)}" for p in zarr_paths]

    return xr.open_mfdataset(local_zarr_paths, engine="zarr", decode_timedelta=True)


def filter_sat_times_to_available(ds_sat: xr.Dataset) -> xr.Dataset:
    """Filters the partially corrupt satellite dataset to timestamps which are available.

    Due to life-cycling in the s3 bucket, older parts of the icechunk data are deleted but without
    the timestamps being updated. This function filters the timestamps to find the minimum timestamp
    for which the corresponding data has not been deleted by lifecycling.

    Args:
        ds_sat: The satellite dataset opened from icechunk
    """

    def check_time_available(i: int) -> bool:
        try:
            ds_sat.isel(time=i, variable=0).compute()
            return True
        except icechunk.IcechunkError:
            return False

    N = len(ds_sat.time)

    # If the first timestamp is avaialble we assume all timestamps are available
    if check_time_available(0):
        return ds_sat

    # If the last timestamp is available we assume none of the timestamps are available
    elif not check_time_available(N-1):
        raise Exception("No sat data available")

    # Otherwise we do an binary search for the first timestamp for which data is available
    else:
        lowest_available_index = N-1
        highest_unavailable_index = 0

        first_available_index = None
        while first_available_index is None:

            i = (lowest_available_index + highest_unavailable_index)//2

            if check_time_available(i):
                lowest_available_index = i
                if i-1==highest_unavailable_index:
                    first_available_index = i

            else:
                highest_unavailable_index = i
                if i+1 == lowest_available_index:
                    first_available_index = i+1

        return ds_sat.isel(time=slice(first_available_index, None))


def get_sat_dataset(
    s3_path: str,
    start_date: pd.Timestamp,
    end_date:pd.Timestamp,
) -> xr.Dataset:
    """Open the satellite data from s3 incechunk and filter to date period.

    Args:
        s3_path: The path on s3 to the icechunk
        start_date: The start datetime of the period
        end_date: The end datetime of the period
    """
    bucket, _, path = s3_path.removeprefix("s3://").partition("/")
    store = icechunk.s3_storage(
        bucket=bucket,
        prefix=path,
        from_env=True,
        region="eu-west-1",
    )

    repo = icechunk.Repository.open(store)
    session = repo.readonly_session("main")

    ds = xr.open_zarr(session.store).sel(time=slice(start_date, end_date))

    return filter_sat_times_to_available(ds)


def get_api_pvlive_dataset(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> xr.Dataset:
    """Get data from PVLive for the given period from the PVLive API.

    Args:
        start_date: The start datetime of the period
        end_date: The end datetime of the period
    """
    py_start_date = start_date.to_pydatetime().replace(tzinfo=UTC)
    py_end_date = end_date.to_pydatetime().replace(tzinfo=UTC)

    pvl = PVLive()

    ds_gsp_list = []
    for gsp_id in tqdm(pvl.gsp_ids):
        df = pvl.between(
            start=py_start_date,
            end=py_end_date,
            entity_type="gsp",
            entity_id=gsp_id,
            extra_fields="capacity_mwp",
            dataframe=True,
        )

        df["datetime_gmt"] = df["datetime_gmt"].dt.tz_localize(None)

        ds = (
            df.set_index("datetime_gmt")
            .drop("gsp_id", axis=1)
            .to_xarray()
            .expand_dims(gsp_id=[gsp_id])
        )

        ds_gsp_list.append(ds)

    return xr.concat(ds_gsp_list, dim="gsp_id")


def get_db_pvlive_dataset(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    updated: bool,
) -> xr.Dataset:
    """Get data from PVLive for the given period from the OCF database.

    Args:
        start_date: The start datetime of the period
        end_date: The end datetime of the period
        updated: Whether to get the in-day initial estimates or the updated values which are created
            the day after.
    """
    if "OCF_DB_URL" not in os.environ:
        raise Exception(
            "To get the PVLive values from the database the environmental variable `OCF_DB_URL` "
            "must be set",
        )

    db_connection = DatabaseConnection(url=os.environ["OCF_DB_URL"], base=Base_Forecast)

    with db_connection.engine.connect() as conn:

        pvlive_cols = """
            g.datetime_utc,
            g.solar_generation_kw,
            g.capacity_mwp,
            l.gsp_id
        """

        regime = "day-after" if updated else "in-day"

        query = text(
            f"""
            SELECT {pvlive_cols}
            FROM gsp_yield AS g
            JOIN location AS l ON g.location_id = l.id
            WHERE g.regime='{regime}'
            AND g.datetime_utc>='{start_date}'
            AND g.datetime_utc<='{end_date}'
            """, # noqa: S608
        )

        df_pvlive = pd.DataFrame(
            conn.execute(query).fetchall(),
            columns=["datetime_utc", "solar_generation_kw", "capacity_mwp", "gsp_id"],
        )

    df_pvlive["generation_mw"] = df_pvlive.solar_generation_kw * 1e-3
    del df_pvlive["solar_generation_kw"]

    return df_pvlive.set_index(["datetime_gmt", "gsp_id"]).to_xarray()


def save(ds: xr.Dataset, chunk_dict: dict[str, int], save_path: str) -> None:
    """Rechunk and save the xarray dataset to zarr storage.

    Args:
        ds: The xarray Dataset to save
        chunk_dict: Dictionary of chunk sizes for all dimensions in the data
        save_path: Path to save the zarr under
    """
    if "variable" in ds:
        ds["variable"] = ds["variable"].astype(str)

    for v in list(ds.variables.keys()):
        ds[v].encoding.clear()

    ds.chunk(chunk_dict).to_zarr(save_path)


if __name__=="__main__":

    # Set up local output directories
    os.makedirs(local_output_dir, exist_ok=False)

    ecmwf_local_dir = f"{local_raw_download_dir}/ecmwf"
    ukv_local_dir = f"{local_raw_download_dir}/ukv"
    cloud_local_dir = f"{local_raw_download_dir}/cloudcasting"

    for local_dir in [ecmwf_local_dir, ukv_local_dir, cloud_local_dir]:
        os.makedirs(local_dir, exist_ok=True)

    # Download the ECMWF IFS data
    ds_ecmwf = get_nwp_dataset(
        s3_dir=ecmwf_s3_dir,
        datetime_fmt=r"%Y%m%d%H",  # IFS zarr files in form YYYYmmddHH.zarr
        start_date=start_date,
        end_date=end_date,
        local_dir=ecmwf_local_dir,
    )

    # Process it to match training data
    ds_ecmwf = ECMWFDownloader.rename_variables(ds_ecmwf)

    # Save
    chunk_dict = {
        "init_time": 1,
        "step": 10,
        "latitude": 50,
        "longitude": 50,
        "variable": len(ds_ecmwf.variable),
    }

    save(ds_ecmwf, chunk_dict, f"{local_output_dir}/ecmwf.zarr")

    # Download the Met Office UKV data
    ds_ukv = get_nwp_dataset(
        s3_dir=ukv_s3_dir,
        datetime_fmt=r"%Y%m%d%H",  # UKV zarr files in form YYYYmmddHH.zarr
        start_date=start_date,
        end_date=end_date,
        local_dir=ukv_local_dir,
    )

    # Process it to match training data
    ds_ukv = ds_ukv.chunk({"step": -1, "variable": -1})
    ds_ukv = UKVDownloader.rename_variables(ds_ukv)
    ds_ukv = UKVDownloader.add_lon_lat_coords(ds_ukv)
    ds_ukv = UKVDownloader.regrid(ds_ukv, split_by_step=False)
    ds_ukv = UKVDownloader.fix_dtype(ds_ukv)

    # Save
    chunk_dict = {
        "init_time": 1,
        "step": 10,
        "x": 100,
        "y": 100,
        "variable": len(ds_ukv.variable),
    }

    save(ds_ukv, chunk_dict, f"{local_output_dir}/ukv.zarr")

    # Download the cloudcasting data
    ds_cloud = get_nwp_dataset(
        s3_dir=cloud_s3_dir,
        datetime_fmt=r"%Y-%m-%dT%H:%M",  # Cloudcasting zarr files in form YYYY-mm-ddTHH:MM.zarr
        start_date=start_date,
        end_date=end_date,
        local_dir=cloud_local_dir,
    )

    # Save - no processing is required since it already matches the training data
    chunk_dict = {
        "init_time": 1,
        "step": 10,
        "x_geostationary": 100,
        "y_geostationary": 100,
        "variable": len(ds_cloud.variable),
    }

    save(ds_cloud, chunk_dict, f"{local_output_dir}/cloudcasting.zarr")

    # Get the satellite data
    ds_sat = get_sat_dataset(sat_s3_icechunk, start_date, end_date)

    # Process it to match training data
    area_sring = yaml.dump(ds_sat.attrs["area"])
    ds_sat.attrs["area"] = ds_sat.data.attrs["area"] = area_sring

    # Save
    chunk_dict = {
        "time": 12,
        "x_geostationary": 100,
        "y_geostationary": 100,
        "variable": len(ds_sat.variable),
    }

    save(ds_sat, chunk_dict, f"{local_output_dir}/rss.zarr")

    # Get new PVLive data from the PVLive API
    ds_pvlive_api = get_api_pvlive_dataset(start_date, end_date)

    chunk_dict = {"gsp_id": len(ds_pvlive_api.gsp_id), "datetime_gmt": 200}
    save(ds_pvlive_api, chunk_dict, f"{local_output_dir}/pvlive_api_data.zarr")

    # Get PVLive data from the OCF database
    ds_pvlive_db = get_db_pvlive_dataset(start_date, end_date, updated=False)

    chunk_dict = {"gsp_id": len(ds_pvlive_db.gsp_id), "datetime_gmt": 200}
    save(ds_pvlive_api, chunk_dict, f"{local_output_dir}/pvlive_db_data.zarr")
