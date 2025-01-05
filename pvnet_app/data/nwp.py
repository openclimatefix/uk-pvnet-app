import numpy as np
import xarray as xr
import xesmf as xe
import logging
from typing import Optional
import os
import fsspec

from pvnet_app.consts import nwp_ukv_path, nwp_ecmwf_path

logger = logging.getLogger(__name__)

this_dir = os.path.dirname(os.path.abspath(__file__))


def _download_nwp_data(source, destination):

    logger.info(f"Downloading NWP data from {source} to {destination}")

    fs = fsspec.open(source).fs
    fs.get(source, destination, recursive=True)


def download_all_nwp_data(
    download_ukv: Optional[bool] = True, download_ecmwf: Optional[bool] = True
):
    """Download the NWP data"""
    if download_ukv:
        _download_nwp_data(os.environ["NWP_UKV_ZARR_PATH"], nwp_ukv_path)
    else:
        logger.info(f"Skipping download of UKV data")
    if download_ecmwf:
        _download_nwp_data(os.environ["NWP_ECMWF_ZARR_PATH"], nwp_ecmwf_path)
    else:
        logger.info(f"Skipping download of ECMWF data")


def regrid_nwp_data(nwp_zarr, target_coords_path, method):
    """This function loads the  NWP data, then regrids and saves it back out if the data is not
    on the same grid as expected. The data is resaved in-place.
    """

    logger.info(f"Regridding NWP data {nwp_zarr} to expected grid to {target_coords_path}")

    ds_raw = xr.open_zarr(nwp_zarr)

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(target_coords_path)

    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude)
        and ds_raw.longitude.equals(ds_target_coords.longitude)
    )

    if not needs_regridding:
        logger.info(f"No NWP regridding required for {nwp_zarr} - skipping this step")
        return

    logger.info(f"Regridding NWP {nwp_zarr} to expected grid")

    # Pull the raw data into RAM
    ds_raw = ds_raw.compute()

    # Regrid in RAM efficient way by chunking first. Each step is regridded separately
    regrid_chunk_dict = {
        "step": 1,
        "latitude": -1,
        "longitude": -1,
        "x": -1,
        "y": -1,
    }

    regridder = xe.Regridder(ds_raw, ds_target_coords, method=method)
    ds_regridded = regridder(
        ds_raw.chunk(
            {k: regrid_chunk_dict[k] for k in list(ds_raw.xindexes) if k in regrid_chunk_dict}
        )
    ).compute(scheduler="single-threaded")

    # Re-save - including rechunking
    os.system(f"rm -rf {nwp_zarr}")
    ds_regridded["variable"] = ds_regridded["variable"].astype(str)

    # Rechunk to these dimensions when saving
    save_chunk_dict = {
        "step": 5,
        "latitude": 100,
        "longitude": 100,
        "x": 100,
        "y": 100,
    }

    ds_regridded.chunk(
        {k: save_chunk_dict[k] for k in list(ds_raw.xindexes) if k in save_chunk_dict}
    ).to_zarr(nwp_zarr)


def fix_ecmwf_data():

    ds = xr.open_zarr(nwp_ecmwf_path).compute()
    ds["variable"] = ds["variable"].astype(str)

    name_sub = {"t": "t2m", "clt": "tcc"}

    if any(v in name_sub for v in ds["variable"].values):
        logger.info(f"Renaming the ECMWF variables")
        ds["variable"] = np.array(
            [name_sub[v] if v in name_sub else v for v in ds["variable"].values]
        )
    else:
        logger.info(f"No ECMWF renaming required - skipping this step")

    logger.info(f"Extending the ECMWF data to reach the shetlands")
    # Thw data must be extended to reach the shetlands. This will fill missing lats with NaNs
    # and reflects what the model saw in training
    ds = ds.reindex(latitude=np.concatenate([np.arange(62, 60, -0.05), ds.latitude.values]))

    # Re-save inplace
    os.system(f"rm -rf {nwp_ecmwf_path}")
    ds.to_zarr(nwp_ecmwf_path)


def fix_ukv_data():
    """Extra steps to align UKV production data with training

    - In training the UKV data is float16. This causes it to overflow into inf values which are then
      clipped.
    """

    ds = xr.open_zarr(nwp_ukv_path).compute()
    ds = ds.astype(np.float16)

    ds["variable"] = ds["variable"].astype(str)

    # Re-save inplace
    os.system(f"rm -rf {nwp_ukv_path}")
    ds.to_zarr(nwp_ukv_path)


def preprocess_nwp_data(use_ukv: Optional[bool] = True, use_ecmwf: Optional[bool] = True):

    if use_ukv:
        # Regrid the UKV data
        regrid_nwp_data(
            nwp_zarr=nwp_ukv_path,
            target_coords_path=f"{this_dir}/../../data/nwp_ukv_target_coords.nc",
            method="bilinear",
        )

        # UKV data must be float16 to allow overflow to inf like in training
        fix_ukv_data()
    else:
        logger.info(f"Skipping UKV data preprocessing")

    if use_ecmwf:

        # rename dataset variable from  HRES-IFS_uk to ECMWF_UK
        rename_ecmwf_variables()

        # Regrid the ECMWF data
        regrid_nwp_data(
            nwp_zarr=nwp_ecmwf_path,
            target_coords_path=f"{this_dir}/../../data/nwp_ecmwf_target_coords.nc",
            method="conservative",  # this is needed to avoid zeros around edges of ECMWF data
        )

        # Names need to be aligned between training and prod, and we need to infill the shetlands
        fix_ecmwf_data()
    else:
        logger.info(f"Skipping ECMWF data preprocessing")


def rename_ecmwf_variables():
    """ Rename the ECMWF variables to what we use in the ML Model"""
    d = xr.open_zarr(nwp_ecmwf_path)
    # if the variable HRES-IFS_uk is there
    if "HRES-IFS_uk" in d.data_vars:
        logger.info(f"Renaming the ECMWF variables")

        d = d.rename({"HRES-IFS_uk": "ECMWF_UK"})

        # rename variable names in the variable coordinate
        # This is a renaming from ECMWF variables to what we use in the ML Model
        # This change happened in the new nwp-consumer>=1.0.0
        # Ideally we won't need this step in the future
        variable_coords = d.variable.values
        rename = {'cloud_cover_high': 'hcc',
                  'cloud_cover_low': 'lcc',
                  'cloud_cover_medium': 'mcc',
                  'cloud_cover_total': 'tcc',
                  'snow_depth_gl': 'sde',
                  'direct_shortwave_radiation_flux_gl': 'sr',
                  'downward_longwave_radiation_flux_gl': 'dlwrf',
                  'downward_shortwave_radiation_flux_gl': 'dswrf',
                  'downward_ultraviolet_radiation_flux_gl': 'durvs',
                  'temperature_sl': 't',
                  'total_precipitation_rate_gl': 'prate',
                  'visibility_sl': 'vis',
                  'wind_u_component_100m': '100',
                  'wind_u_component_10m': 'u10',
                  'wind_u_component_200m': 'u200',
                  'wind_v_component_100m': 'v100',
                  'wind_v_component_10m': 'v10',
                  'wind_v_component_200m': 'v200'}

        for k, v in rename.items():
            variable_coords[variable_coords == k] = v

        # assign the new variable names
        d = d.assign_coords(variable=variable_coords)

        # save back to path
        os.system(f"rm -rf {nwp_ecmwf_path}")
        d.to_zarr(nwp_ecmwf_path)
