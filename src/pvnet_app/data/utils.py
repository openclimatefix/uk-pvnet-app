"""Utility functions for processing the raw input data."""

import xarray as xr
from ocf_data_sampler.load.utils import make_spatial_coords_increasing
from ocf_data_sampler.select.geospatial import convert_coordinates, find_coord_system
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.select_spatial_slice import select_spatial_slice_pixels_multiple

from pvnet_app.data.gsp import get_gsp_locations


def slice_to_pvnet_spatial_area(
    ds: xr.Dataset,
    width_pixels: int,
    height_pixels: int,
) -> xr.Dataset:
    """Get the spatial extent of the satellite data used in PVNet.

    Args:
        ds: The dataset
        width_pixels: The width of the spatial slice in pixels
        height_pixels: The height of the spatial slice in pixels

    Returns:
        xr.Dataset: The spatial slice of the dataset used by PVNet
    """
    coord_system, x_coord, y_coord = find_coord_system(ds)

    # Cut down the slice for efficiency and reorder the coordinates if needed
    ds = make_spatial_coords_increasing(
        ds,
        x_coord=x_coord,
        y_coord=y_coord,
    )

    # We will loop over all the GSP locations and find the min and max x and y coordinates
    # This gives us a bounding box used by PVNet
    df_locs = get_gsp_locations().loc[1:]

    if coord_system == "lon_lat":
        xs, ys = df_locs.longitude.values, df_locs.latitude.values

    else:
        xs, ys = convert_coordinates(
            x=df_locs.longitude.values,
            y=df_locs.latitude.values,
            from_coords="lon_lat",
            target_coords=coord_system,
            area_string=str(ds.attrs["area"]) if "area" in ds.attrs else None,
        )

    # Add the projection to the locations objects
    locations = []
    for x, y, loc_id in zip(xs, ys, df_locs.index.values, strict=True):
        locations.append(Location(x=x, y=y, coord_system=coord_system, id=loc_id))

    return select_spatial_slice_pixels_multiple(ds, locations, width_pixels, height_pixels)
