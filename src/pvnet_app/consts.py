from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pvnet-app")
except PackageNotFoundError:
    __version__ = "v?"

sat_path = "sat.zarr"
nwp_ukv_path = "nwp_ukv.zarr"
nwp_ecmwf_path = "nwp_ecmwf.zarr"
nwp_cloudcasting_path = "nwp_cloudcasting.zarr"
