import xarray as xr
import xesmf as xe
import logging
import os
import fsspec

from pvnet_app.consts import sat_path, sat_15_path, nwp_path

logger = logging.getLogger(__name__)

this_dir = os.path.dirname(os.path.abspath(__file__))

def download_sat_data():
    """Download the sat data"""
    fs = fsspec.open(os.environ["SATELLITE_ZARR_PATH"]).fs
    fs.get(os.environ["SATELLITE_ZARR_PATH"], "sat.zarr.zip")
    os.system(f"rm -r {sat_path}")
    os.system(f"unzip sat.zarr.zip -d {sat_path}")
    
    # Also download 15-minute satellite if it exists
    sat_latest_15 = os.environ["SATELLITE_ZARR_PATH"].replace("sat.zarr", "sat_15.zarr")
    if fs.exists(sat_latest_15):
        logger.info("Downloading 15-minute satellite data")
        fs.get(sat_latest_15, "sat_15.zarr")
        os.system(f"unzip sat_15.zarr.zip -d {sat_15_path}")
        

def download_nwp_data():
    """Download the NWP data"""
    fs = fsspec.open(os.environ["NWP_ZARR_PATH"]).fs
    fs.get(os.environ["NWP_ZARR_PATH"], nwp_path, recursive=True)

    
def regrid_nwp_data():
    """This function loads the NWP data, then regrids and saves it back out if the data is not on
    the same grid as expected. The data is resaved in-place.
    """
    
    ds_raw = xr.open_zarr(nwp_path)

    # These are the coords we are aiming for
    ds_target_coords = xr.load_dataset(f"{this_dir}/../data/nwp_target_coords.nc")
    
    # Check if regridding step needs to be done
    needs_regridding = not (
        ds_raw.latitude.equals(ds_target_coords.latitude) and
         ds_raw.longitude.equals(ds_target_coords.longitude)
        
    )
    
    if not needs_regridding:
        logger.info("No NWP regridding required - skipping this step")
        return
    
    logger.info("Regridding NWP to expected grid")
    
    # Pull the raw data into RAM
    ds_raw = ds_raw.compute()
    
    # Regrid in RAM efficient way by chunking first. Each step is regridded separately
    regridder = xe.Regridder(ds_raw, ds_target_coords, method="bilinear")
    ds_regridded = regridder(
        ds_raw.chunk(dict(x=-1, y=-1, step=1))
    ).compute(scheduler="single-threaded")

    # Re-save - including rechunking
    os.system(f"rm -fr {nwp_path}")
    ds_regridded["variable"] = ds_regridded["variable"].astype(str)
    ds_regridded.chunk(dict(step=12, x=100, y=100)).to_zarr(nwp_path)
    
    return