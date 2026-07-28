"""Constants."""

sat_path: str = "sat.zarr"
nwp_ukv_path: str = "nwp_ukv.zarr"
nwp_ecmwf_path: str = "nwp_ecmwf.zarr"
nwp_cloudcasting_path: str = "nwp_cloudcasting.zarr"
generation_path: str = "generation.zarr"

# Our current API requires this exact forecast version to be assigned for all forecasts. This
# will be updated in the future to allow for different versions of forecasts to be assigned.
forecast_version: str = "2.8.0"
