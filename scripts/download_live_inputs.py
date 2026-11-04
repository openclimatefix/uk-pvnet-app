"""This script downloads archive input data from the live production system.

The data is processed equivalently to how it is processed in the uk-pvnet-app so that a recent
live backtest can be performed.

The NWP and satellite data are pulled from OCF's s3 buckets, so AWS credentials which can read
those buckets must be available in the environment.

Use from the uk-pvnet-app root directory:
```
# Create environment and install dependencies with make
make sync

# Run the script to download and process the data
conda run --no-capture-output -p .venv uv run --with icechunk python scripts/download_live_inputs.py
```
"""

import logging
import os
import shutil
import warnings
from functools import partial
from glob import glob
from typing import Any, Literal

import dask
import fsspec
import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr
from dask.diagnostics import ProgressBar
from ocf_data_sampler.load.open_xarray_tensorstore import open_zarrs
from s3fs import S3FileSystem
from tqdm import tqdm
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec

from pvnet_app.data.nwp import ECMWFDownloader, UKVDownloader

# -------------------------------------------------------------
# USER VARIABLES

START_DATE: pd.Timestamp = pd.Timestamp("2026-08-04 00:00")
END_DATE: pd.Timestamp = pd.Timestamp("2026-08-06 23:30")

# "w" to overwrite existing data, "a-" to append to existing data without overwriting
WRITE_MODE: Literal["w", "a-"] = "a-"

RAW_DOWNLOAD_DIR: str = "/mnt/storage_u2_30tb_a/source_data/uk_eclipse_inputs/raw"
PROCESSED_ARCHIVE_DIR: str = "/mnt/storage_u2_30tb_a/source_data/uk_eclipse_inputs/processed"

ECMWF_S3_DIR: str = "s3://nowcasting-nwp-development/ecmwf/data"
UKV_S3_DIR: str = "s3://nowcasting-nwp-development/data-metoffice"
SAT_S3_ICECHUNK: str = "s3://nowcasting-sat-development/rss_v1/data/rss_uk3000m.icechunk"

# The ECMWF and UKV zarrs are saved on s3 with filenames like 2025120100.zarr
NWP_DATETIME_FMT: str = r"%Y%m%d%H"

# Number of UKV init-times to process and save at once
UKV_BATCH_SIZE: int = 4

# Compressor used for all saved data
COMPRESSOR: BloscCodec = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle", blocksize=0)

# The size of the window to download around the UKV and ECMWF data
# This slices the data to just the area used by PVNet. Set to None for the whole area
WINDOW_SIZE_PIXELS: int | None = 24  

# -------------------------------------------------------------

if WRITE_MODE not in ["w", "a-"]:
    raise ValueError("WRITE_MODE must be either 'w' or 'a-'")
if START_DATE >= END_DATE:
    raise ValueError("START_DATE must be before END_DATE")

logger = logging.getLogger("live_data_downloader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logging.getLogger("pvnet_app").setLevel(logging.WARNING)
logging.getLogger("aiobotocore.credentials").setLevel(logging.WARNING)

# Suppress the noisy warnings from the zarr/xarray/pyproj/h5py stack. Note that ignoring all
# FutureWarnings also covers the pyproj "+init=<authority>:<code> is deprecated" warning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Consolidated metadata is currently not part in the Zarr format 3 specification",
)
warnings.filterwarnings("ignore", category=UserWarning, message="h5py is running against HDF5.*")


# -------------------------------------------------------------
# FUNCTIONS


class DummyUKVDownloader(UKVDownloader):
    """A dummy UKVDownloader which does not require a config file or model."""

    def __init__(self) -> None:
        self.window_size_pixels = WINDOW_SIZE_PIXELS

class DummyECMWFDownloader(ECMWFDownloader):
    """A dummy ECMWFDownloader which does not require a config file or model."""

    def __init__(self) -> None:
        self.window_size_pixels = WINDOW_SIZE_PIXELS


@dask.delayed
def download_zarr(fs: S3FileSystem, s3_path: str, download_dir: str, retries: int = 3) -> None:
    """Download a single zarr from s3, retrying if the download is incomplete.

    Args:
        fs: The s3 filesystem to download through
        s3_path: The path of the zarr on s3
        download_dir: The local directory to download the zarr into
        retries: Number of times to attempt the download before raising
    """
    zarr_name = os.path.basename(s3_path)
    local_path = f"{download_dir}/{zarr_name}"

    s3_files = fs.find(s3_path)
    local_files = [os.path.join(local_path, f.replace(s3_path + "/", "")) for f in s3_files]

    for local_file in local_files:
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

    for attempt in range(retries):
        if attempt > 0 and os.path.exists(local_path):
            shutil.rmtree(local_path)
            for local_file in local_files:
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

        fs.get(s3_files, local_files, batch_size=32)

        downloaded = [f for f in local_files if os.path.isfile(f)]
        if len(downloaded) >= len(s3_files):
            return

        missing = set(local_files) - set(downloaded)
        logger.warning(
            f"Attempt {attempt + 1}: incomplete download for {zarr_name} "
            f"({len(downloaded)}/{len(s3_files)}) Missing: {missing}",
        )

    shutil.rmtree(local_path, ignore_errors=True)
    raise RuntimeError(f"Failed to fully download {zarr_name} after {retries} attempts")


def extract_timestamp_from_path(path: str, datetime_fmt: str) -> pd.Timestamp:
    """Extract the timestamp which a zarr is named after from its path.

    Args:
        path: The path of the zarr
        datetime_fmt: The datetime format that the zarr is named under
    """
    filename = os.path.basename(path)
    timestamp_str = filename.removesuffix(".zarr")
    return pd.to_datetime(timestamp_str, format=datetime_fmt)


def archive_write_kwargs(
    archive_zarr: str,
    write_mode: Literal["w", "a-"],
    append_dim: str,
) -> dict[str, Any]:
    """Construct the `to_zarr()` keyword arguments used to write to an archive.

    We only append if we have been asked to and the archive already exists, else we overwrite.

    Args:
        archive_zarr: Path of the archive zarr
        write_mode: Whether to overwrite existing data ("w") or append to it ("a-")
        append_dim: The dimension to append along if appending
    """
    if write_mode == "a-" and os.path.exists(archive_zarr):
        return {"mode": "a-", "append_dim": append_dim}
    return {"mode": "w"}


def round_up(x: int, multiple: int) -> int:
    """Round a size up to the nearest whole number of `multiple`.

    Args:
        x: The size to round up
        multiple: The number to round up to a whole number of
    """
    return int(np.ceil(x / multiple) * multiple)


def rechunk_and_save(
    ds: xr.Dataset,
    save_path: str,
    chunk_dict: dict[str, int],
    shard_dict: dict[str, int],
    coord_chunk_dict: dict[str, int] | None = None,
    compressor: BytesBytesCodec = COMPRESSOR,
    **kwargs: Any,
) -> None:
    """Rechunk and save the xarray dataset to sharded zarr storage.

    A chunk or shard size of -1 means the whole dimension. Shard sizes are rounded up to a whole
    number of chunks. The encoding is only applied when the zarr is created - when appending, zarr
    keeps the chunk and shard scheme the store was created with.

    Args:
        ds: The xarray Dataset to save
        save_path: Path to save the zarr under
        chunk_dict: Dictionary of chunk sizes for the dimensions in the data
        shard_dict: Dictionary of shard sizes for the dimensions in the data
        coord_chunk_dict: Dictionary of chunk sizes to use for the coordinates. Coordinate
            dimensions not included here are stored in a single chunk. Mostly useful for the
            dimension which is appended along, which would otherwise be chunked by append size
        compressor: The compressor used for all variables
        **kwargs: Extra keyword arguments passed to `Dataset.to_zarr()`
    """
    chunk_dict = chunk_dict.copy()
    shard_dict = shard_dict.copy()
    coord_chunk_dict = {} if coord_chunk_dict is None else coord_chunk_dict

    if set(chunk_dict) != set(shard_dict):
        raise ValueError("chunk_dict and shard_dict must cover the same dimensions")
    if not set(chunk_dict).issubset(ds.dims):
        raise ValueError(
            f"chunk_dict has dimensions not in the data: {set(chunk_dict) - set(ds.dims)}",
        )

    # Resolve the -1 values into sizes and check the shards divide into whole chunks
    for dim in chunk_dict:
        if chunk_dict[dim] == -1:
            chunk_dict[dim] = ds.sizes[dim]
        if shard_dict[dim] == -1:
            shard_dict[dim] = round_up(ds.sizes[dim], chunk_dict[dim])

        if shard_dict[dim] % chunk_dict[dim] != 0:
            raise ValueError(
                f"Shard size {shard_dict[dim]} for dimension '{dim}' is not a whole number of "
                f"chunks of size {chunk_dict[dim]}",
            )

    if "variable" in ds:
        ds["variable"] = ds["variable"].astype(str)

    # Clear old encoding
    for v in list(ds.variables.keys()):
        ds[v].encoding.clear()

    # This rechunk makes saving more efficient since each shard is written by one worker
    # It will also wrap the tensorstore backend with dask
    ds = ds.chunk({dim: min(ds.sizes[dim], shard_dict[dim]) for dim in shard_dict})

    # Same compressor for all variables
    encoding: dict[str, dict[str, Any]] = {v: {"compressors": compressor} for v in ds.variables}

    for coord in ds.coords:
        if ds[coord].ndim == 0:
            continue
        # Coords are stored in a single chunk unless a chunk size has been given for them
        encoding[coord]["chunks"] = tuple(
            coord_chunk_dict.get(dim, ds.sizes[dim]) for dim in ds[coord].dims
        )
        # Pin the datetime encoding so that appended data cannot be rounded to coarser units
        if np.issubdtype(ds[coord].dtype, np.datetime64):
            encoding[coord].update(
                {
                    "dtype": "int",
                    "units": "nanoseconds since 1970-01-01",
                    "calendar": "proleptic_gregorian",
                },
            )

    # Set the chunk and shard scheme for the data variables which the dicts cover
    for data_var in ds.data_vars:
        if set(ds[data_var].dims).issubset(chunk_dict):
            encoding[data_var]["chunks"] = tuple(chunk_dict[d] for d in ds[data_var].dims)
            encoding[data_var]["shards"] = tuple(shard_dict[d] for d in ds[data_var].dims)

    ds.to_zarr(
        save_path,
        zarr_format=3,
        consolidated=False,
        # xarray takes the encoding from the store for variables which already exist, so we only
        # pass it when creating the zarr
        encoding=None if "append_dim" in kwargs else encoding,
        **kwargs,
    )


# -------------------------------------------------------------
# NWP


def download_nwp_dataset(
    s3_dir: str,
    datetime_fmt: str,
    download_dir: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Download a collection of zarrs from s3 to local disk.

    Files that already exist in the download location will not be redownloaded. We filter the
    zarrs to ones which have a filename which is between the supplied start and end date.

    Args:
        s3_dir: Where to search for the zarrs
        datetime_fmt: The datetime format that zarrs are saved under. e.g. r"%Y%m%d%H" for zarr
            which is named like 2025120100.zarr
        download_dir: Where to save the zarrs to in the local filesystem
        start_date: The start datetime used to filter the zarrs
        end_date: The end datetime used to filter the zarrs
    """
    if datetime_fmt not in [r"%Y%m%d%H", r"%Y-%m-%dT%H:%M"]:
        raise ValueError("This function has not been tested with the datetime format provided")

    # Convert the datetime format into a glob pattern
    glob_pattern = datetime_fmt
    for s, n in [("Y", 4), ("m", 2), ("d", 2), ("H", 2), ("M", 2)]:
        glob_pattern = glob_pattern.replace(r"%" + s, "[0-9]" * n)

    # Find the zarr files
    fs = fsspec.filesystem("s3")
    zarr_paths = sorted(fs.glob(f"{s3_dir}/{glob_pattern}.zarr"))

    # Extract the timestamp from each path
    get_path_timestamp = partial(extract_timestamp_from_path, datetime_fmt=datetime_fmt)
    zarr_datetimes = [get_path_timestamp(p) for p in zarr_paths]

    logger.info(
        f"Found {len(zarr_datetimes)} zarr files in s3 directory {s3_dir} (before date filtering), "
        f"spanning init-times {min(zarr_datetimes)} to {max(zarr_datetimes)}."
    )

    # Filter to zarrs with datetime between the start and end dates
    zarr_paths = [
        path for path in zarr_paths if (start_date <= get_path_timestamp(path) <= end_date)
    ]

    # Filter out files already downloaded
    required_zarr_paths = [
        p for p in zarr_paths if not os.path.exists(f"{download_dir}/{os.path.basename(p)}")
    ]

    if len(required_zarr_paths) == 0:
        logger.info("All zarr files in the specified date range are already downloaded locally.")
        return

    required_zarr_datetimes = [get_path_timestamp(p) for p in required_zarr_paths]

    logger.info(
        f"Downloading the {len(required_zarr_paths)} of these zarr files which are not already "
        f"downloaded locally, between {min(required_zarr_datetimes)} and "
        f"{max(required_zarr_datetimes)}.",
    )

    # Download in parallel
    tasks = [download_zarr(fs, s3_path, download_dir) for s3_path in required_zarr_paths]
    with ProgressBar():
        dask.compute(*tasks, scheduler="threads", num_workers=2)


def get_nwp_dataset(
    datetime_fmt: str,
    raw_download_dir: str,
    archive_zarr: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    write_mode: Literal["w", "a-"] = "a-",
) -> xr.Dataset | None:
    """Open the downloaded NWP zarrs which still need to be processed into the archive.

    Args:
        datetime_fmt: The datetime format that the downloaded zarrs are named under
        raw_download_dir: The local directory the zarrs were downloaded to
        archive_zarr: Path of the processed archive zarr
        start_date: The start datetime used to filter the zarrs
        end_date: The end datetime used to filter the zarrs
        write_mode: Whether the archive will be overwritten ("w") or appended to ("a-"). If
            appending, init-times already in the archive are excluded.

    Returns:
        The dataset of init-times still to be archived, or None if there are none
    """
    if os.path.exists(archive_zarr):
        existing_inits = pd.to_datetime(xr.open_zarr(archive_zarr, consolidated=False).init_time)
    else:
        existing_inits = pd.to_datetime([])

    def init_time_filter(path: str) -> bool:
        timestamp = extract_timestamp_from_path(path, datetime_fmt)
        in_bounds = start_date <= timestamp <= end_date
        if write_mode == "w":
            return in_bounds
        return in_bounds and timestamp not in existing_inits

    # Filter to zarrs which are in the date range and not already in the archive
    zarr_paths = [p for p in sorted(glob(f"{raw_download_dir}/*.zarr")) if init_time_filter(p)]

    if len(zarr_paths) == 0:
        return None
    return open_zarrs(zarr_paths, concat_dim="init_time").sortby("init_time")


# -------------------------------------------------------------
# SATELLITE


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
            ds_sat["data"].isel(time=i, channel=0).compute()
            return True
        except icechunk.IcechunkError:
            return False

    n_times = len(ds_sat.time)

    # If the first timestamp is available we assume all timestamps are available
    if check_time_available(0):
        return ds_sat

    # If the last timestamp is not available we assume none of the timestamps are available
    if not check_time_available(n_times - 1):
        raise Exception("No sat data available")

    # Otherwise we do a binary search for the first timestamp for which data is available
    lowest_available_index = n_times - 1
    highest_unavailable_index = 0

    first_available_index = None
    while first_available_index is None:
        i = (lowest_available_index + highest_unavailable_index) // 2

        if check_time_available(i):
            lowest_available_index = i
            if i - 1 == highest_unavailable_index:
                first_available_index = i

        else:
            highest_unavailable_index = i
            if i + 1 == lowest_available_index:
                first_available_index = i + 1

    return ds_sat.isel(time=slice(first_available_index, None))


def download_sat_dataset(
    s3_icechunk_path: str,
    raw_download_zarr: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Open the satellite data from s3 icechunk, filter to the date period, and save it locally.

    Args:
        s3_icechunk_path: The path on s3 to the icechunk
        raw_download_zarr: The local zarr to download the satellite data into
        start_date: The start datetime of the period
        end_date: The end datetime of the period
    """
    bucket, _, path = s3_icechunk_path.removeprefix("s3://").partition("/")
    store = icechunk.s3_storage(
        bucket=bucket,
        prefix=path,
        from_env=True,
        region="eu-west-1",
    )

    repo = icechunk.Repository.open(store)
    session = repo.readonly_session("main")

    ds_s3 = xr.open_zarr(session.store)

    # The live data can contain duplicate and out-of-order timestamps, which we cannot slice by
    # time. Keep the first occurrence of each timestamp and sort into time order
    _, unique_indexes = np.unique(ds_s3.time.values, return_index=True)
    if len(unique_indexes) != len(ds_s3.time):
        logger.warning(
            f"Dropping {len(ds_s3.time) - len(unique_indexes)} duplicate satellite timestamps.",
        )
    ds_s3 = ds_s3.isel(time=unique_indexes)

    # Find the times that are actually available on s3
    ds_s3 = filter_sat_times_to_available(ds_s3)

    logger.info(
        f"Satellite data on s3 has {len(ds_s3.time)} timestamps between {ds_s3.time.min().values} "
        f"and {ds_s3.time.max().values}.",
    )
    ds_s3 = ds_s3.sel(time=slice(start_date, end_date))

    if os.path.exists(raw_download_zarr):
        ds_local = xr.open_zarr(raw_download_zarr, consolidated=False)

        # Remove already downloaded times
        already_downloaded = np.isin(ds_s3.time.values, ds_local.time.values)
        ds_s3 = ds_s3.sel(time=ds_s3.time.values[~already_downloaded])

    if len(ds_s3.time) == 0:
        logger.info("All satellite data in the specified date range is already downloaded locally.")
        return

    logger.info(
        f"Downloading {len(ds_s3.time)} satellite images between {ds_s3.time.min().values} "
        f"and {ds_s3.time.max().values}.",
    )

    # Chunk the data for storage and save. One shard per timestamp
    chunk_dict = {
        "time": 1,
        "x_geostationary": -1,
        "y_geostationary": -1,
        "channel": -1,
    }
    rechunk_and_save(
        ds_s3.compute(),
        raw_download_zarr,
        chunk_dict=chunk_dict,
        shard_dict=chunk_dict,
        coord_chunk_dict={"time": 300},
        **archive_write_kwargs(raw_download_zarr, "a-", append_dim="time"),
    )


def get_sat_dataset(
    raw_download_zarr: str,
    archive_zarr: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    write_mode: Literal["w", "a-"] = "a-",
) -> xr.Dataset | None:
    """Open the downloaded satellite data which still needs to be processed into the archive.

    Args:
        raw_download_zarr: The local zarr the satellite data was downloaded into
        archive_zarr: Path of the processed archive zarr
        start_date: The start datetime used to filter the timestamps
        end_date: The end datetime used to filter the timestamps
        write_mode: Whether the archive will be overwritten ("w") or appended to ("a-"). If
            appending, timestamps already in the archive are excluded.

    Returns:
        The dataset of timestamps still to be archived, or None if there are none
    """
    if os.path.exists(archive_zarr):
        existing_times = xr.open_zarr(archive_zarr, consolidated=False).time.values
    else:
        existing_times = []

    def time_filter(dt: np.datetime64) -> bool:
        in_bounds = start_date <= dt <= end_date
        if write_mode == "w":
            return in_bounds
        return in_bounds and dt not in existing_times

    ds_local = xr.open_zarr(raw_download_zarr, consolidated=False)
    ds_local = ds_local.sel(time=[t for t in ds_local.time.values if time_filter(t)])

    if len(ds_local.time) == 0:
        return None
    return ds_local.sortby("time")


# -------------------------------------------------------------
# DOWNLOAD, PROCESS AND ARCHIVE EACH SOURCE


def archive_ecmwf(
    raw_download_dir: str,
    archive_zarr: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    write_mode: Literal["w", "a-"],
) -> None:
    """Download the ECMWF IFS data and process it to match the training data.

    Args:
        raw_download_dir: Where to download the raw zarrs to in the local filesystem
        archive_zarr: Path to save the processed archive zarr under
        start_date: The start datetime of the period
        end_date: The end datetime of the period
        write_mode: Whether to overwrite existing archive data ("w") or append to it ("a-")
    """
    logger.info("Downloading ECMWF data")

    download_nwp_dataset(
        s3_dir=ECMWF_S3_DIR,
        datetime_fmt=NWP_DATETIME_FMT,
        start_date=start_date,
        end_date=end_date,
        download_dir=raw_download_dir,
    )

    ds_ecmwf = get_nwp_dataset(
        datetime_fmt=NWP_DATETIME_FMT,
        raw_download_dir=raw_download_dir,
        archive_zarr=archive_zarr,
        start_date=start_date,
        end_date=end_date,
        write_mode=write_mode,
    )

    if ds_ecmwf is None:
        logger.info("No new ECMWF data to process and save")
        return

    logger.info("Processing and saving ECMWF data")

    ds_ecmwf = ds_ecmwf.chunk({"init_time": 1})

    # Process it to match training data
    ds_ecmwf = DummyECMWFDownloader().process(ds_ecmwf)

    # Save. One shard per init-time
    chunk_dict = {
        "init_time": 1,
        "step": 10,
        "latitude": 50,
        "longitude": 50,
        "variable": -1,
    }
    shard_dict = {
        "init_time": 1,
        "step": -1,
        "latitude": -1,
        "longitude": -1,
        "variable": -1,
    }
    with ProgressBar():
        rechunk_and_save(
            ds_ecmwf,
            archive_zarr,
            chunk_dict=chunk_dict,
            shard_dict=shard_dict,
            coord_chunk_dict={"init_time": 300},
            **archive_write_kwargs(archive_zarr, write_mode, append_dim="init_time"),
        )


def archive_ukv(
    raw_download_dir: str,
    archive_zarr: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    write_mode: Literal["w", "a-"],
) -> None:
    """Download the Met Office UKV data and process it to match the training data.

    The data is processed and saved in batches of init-times to limit memory use.

    Args:
        raw_download_dir: Where to download the raw zarrs to in the local filesystem
        archive_zarr: Path to save the processed archive zarr under
        start_date: The start datetime of the period
        end_date: The end datetime of the period
        write_mode: Whether to overwrite existing archive data ("w") or append to it ("a-")
    """
    logger.info("Downloading UKV data")

    download_nwp_dataset(
        s3_dir=UKV_S3_DIR,
        datetime_fmt=NWP_DATETIME_FMT,
        start_date=start_date,
        end_date=end_date,
        download_dir=raw_download_dir,
    )

    ds_ukv = get_nwp_dataset(
        datetime_fmt=NWP_DATETIME_FMT,
        raw_download_dir=raw_download_dir,
        archive_zarr=archive_zarr,
        start_date=start_date,
        end_date=end_date,
        write_mode=write_mode,
    )

    if ds_ukv is None:
        logger.info("No new UKV data to process and save")
        return

    logger.info(f"Processing and saving UKV data in chunks of {UKV_BATCH_SIZE} init-times")

    # One shard per init-time
    chunk_dict = {
        "init_time": 1,
        "step": 10,
        "x_osgb": 100,
        "y_osgb": 100,
        "variable": -1,
    }
    shard_dict = {
        "init_time": 1,
        "step": -1,
        "x_osgb": -1,
        "y_osgb": -1,
        "variable": -1,
    }

    ukv_downloader = DummyUKVDownloader()

    pbar = tqdm(range(0, len(ds_ukv.init_time), UKV_BATCH_SIZE), desc="UKV batches")

    for i in pbar:
        pbar.set_postfix_str("processing")
        ds_ukv_part = ds_ukv.isel(init_time=slice(i, i + UKV_BATCH_SIZE)).load()
        ds_ukv_part = ukv_downloader.process(ds_ukv_part).compute()

        if i == 0:
            kwargs = archive_write_kwargs(archive_zarr, write_mode, append_dim="init_time")
        else:
            kwargs = {"mode": "a-", "append_dim": "init_time"}

        pbar.set_postfix_str("saving")
        with dask.config.set(scheduler="threads"):
            rechunk_and_save(
                ds_ukv_part,
                archive_zarr,
                chunk_dict=chunk_dict,
                shard_dict=shard_dict,
                coord_chunk_dict={"init_time": 300},
                **kwargs,
            )


def archive_sat(
    raw_download_zarr: str,
    archive_zarr: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    write_mode: Literal["w", "a-"],
) -> None:
    """Download the satellite data and process it to match the training data.

    Args:
        raw_download_zarr: The local zarr to download the raw satellite data into
        archive_zarr: Path to save the processed archive zarr under
        start_date: The start datetime of the period
        end_date: The end datetime of the period
        write_mode: Whether to overwrite existing archive data ("w") or append to it ("a-")
    """
    logger.info("Downloading satellite data")

    download_sat_dataset(
        s3_icechunk_path=SAT_S3_ICECHUNK,
        raw_download_zarr=raw_download_zarr,
        start_date=start_date,
        end_date=end_date,
    )

    ds_sat = get_sat_dataset(
        raw_download_zarr=raw_download_zarr,
        archive_zarr=archive_zarr,
        start_date=start_date,
        end_date=end_date,
        write_mode=write_mode,
    )

    if ds_sat is None:
        logger.info("No new satellite data to process and save")
        return

    logger.info("Processing and saving satellite data")

    # Process it to match training data - the area attribute must be a string to save to zarr
    area_string = yaml.dump(ds_sat.attrs["area"])
    ds_sat.attrs["area"] = ds_sat.data.attrs["area"] = area_string

    # Save. One shard per hour of images
    chunk_dict = {
        "time": 12,
        "x_geostationary": 100,
        "y_geostationary": 100,
        "channel": -1,
    }
    shard_dict = {
        "time": 12,
        "x_geostationary": -1,
        "y_geostationary": -1,
        "channel": -1,
    }
    rechunk_and_save(
        ds_sat[["data"]],
        archive_zarr,
        chunk_dict=chunk_dict,
        shard_dict=shard_dict,
        coord_chunk_dict={"time": 300},
        **archive_write_kwargs(archive_zarr, write_mode, append_dim="time"),
    )


def main() -> None:
    """Download, process and archive the ECMWF, UKV and satellite data for the given period."""
    # Paths to save the raw downloads
    ecmwf_raw_download_dir = f"{RAW_DOWNLOAD_DIR}/ecmwf"
    ukv_raw_download_dir = f"{RAW_DOWNLOAD_DIR}/ukv"
    sat_raw_download_zarr = f"{RAW_DOWNLOAD_DIR}/rss.zarr"

    # Paths to save the processed archives
    ecmwf_archive_zarr = f"{PROCESSED_ARCHIVE_DIR}/ecmwf.zarr"
    ukv_archive_zarr = f"{PROCESSED_ARCHIVE_DIR}/ukv.zarr"
    sat_archive_zarr = f"{PROCESSED_ARCHIVE_DIR}/rss.zarr"

    # Set up local output directories
    for p in [ecmwf_raw_download_dir, ukv_raw_download_dir, PROCESSED_ARCHIVE_DIR]:
        os.makedirs(p, exist_ok=True)

    archive_ecmwf(
        raw_download_dir=ecmwf_raw_download_dir,
        archive_zarr=ecmwf_archive_zarr,
        start_date=START_DATE,
        end_date=END_DATE,
        write_mode=WRITE_MODE,
    )

    archive_ukv(
        raw_download_dir=ukv_raw_download_dir,
        archive_zarr=ukv_archive_zarr,
        start_date=START_DATE,
        end_date=END_DATE,
        write_mode=WRITE_MODE,
    )

    archive_sat(
        raw_download_zarr=sat_raw_download_zarr,
        archive_zarr=sat_archive_zarr,
        start_date=START_DATE,
        end_date=END_DATE,
        write_mode=WRITE_MODE,
    )

    logger.info("Finished!")


if __name__ == "__main__":
    main()
