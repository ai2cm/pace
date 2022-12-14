import copy
from typing import Tuple, TypeVar, Union

import numpy as np

from pace.util import Quantity


T = TypeVar("T", bound=Union[Quantity, np.ndarray])


def direct_transform(
    *,
    lon: T,
    lat: T,
    stretch_factor: float,
    lon_target: float,
    lat_target: float,
    np,
) -> Tuple[T, T]:
    """
    The direct_transform subroutine from fv_grid_utils.F90.
    Takes in latitude and longitude in radians.
    Shrinks tile 6 by stretch factor in area to increse resolution locally.
    Then performs translation of all tiles so that the now-smaller tile 6 is
    centeres on lon_target, lat_target.

    Args:
        lon (in) in radians
        lat (in) in radians
        stretch_factor (in) stretch_factor (e.g. 3.0 means that the resolution
            on tile 6 becomes 3 times as fine)
        lon_target (in) in degrees (from namelist)
        lat_target (in) in degrees (from namelist)
        np: numpy or cupy module

    Returns:
        lon_transform (out) in radians
        lat_transform (out) in radians
    """

    if isinstance(lon, Quantity):
        lon_data = lon.data
        lat_data = lat.data
    elif isinstance(lon, np.ndarray):
        lon_data = lon
        lat_data = lat
    else:
        raise Exception("Input data type not supported.")

    STRETCH_GRID_ROTATION_LON_OFFSET_DEG = 190
    # this is added to all longitude values to match the SHiELD TC case
    # 180 is to flip the orientation around the center tile (6)
    # 10 is because the tile center is offset from the prime meridian by 10
    lon_data = lon_data + np.deg2rad(STRETCH_GRID_ROTATION_LON_OFFSET_DEG)

    lon_p, lat_p = np.deg2rad(lon_target), np.deg2rad(lat_target)
    sin_p, cos_p = np.sin(lat_p), np.cos(lat_p)
    c2p1 = 1.0 + stretch_factor ** 2
    c2m1 = 1.0 - stretch_factor ** 2

    # first limit longitude so it's between 0 and 2pi
    lon_data[lon_data < 0] += 2 * np.pi
    lon_data[lon_data >= 2 * np.pi] -= 2 * np.pi

    if np.abs(c2m1) > 1e-7:  # do stretching
        lat_t = np.arcsin(
            (c2m1 + c2p1 * np.sin(lat_data)) / (c2p1 + c2m1 * np.sin(lat_data))
        )
    else:  # no stretching
        lat_t = lat_data

    sin_lat = np.sin(lat_t)
    cos_lat = np.cos(lat_t)

    sin_o = -(sin_p * sin_lat + cos_p * cos_lat * np.cos(lon_data))
    tmp = 1 - np.abs(sin_o)

    lon_transformed = np.zeros(lon_data.shape) * np.nan
    lat_transformed = np.zeros(lat_data.shape) * np.nan

    lon_transformed[tmp < 1e-7] = 0.0
    lat_transformed[tmp < 1e-7] = np.abs(np.pi / 2) * np.sign(sin_o[tmp < 1e-7])

    lon_transformed[tmp >= 1e-7] = lon_p + np.arctan2(
        -np.cos(lat_t[tmp >= 1e-7]) * np.sin(lon_data[tmp >= 1e-7]),
        -np.sin(lat_t[tmp >= 1e-7]) * np.cos(lat_p)
        + np.cos(lat_t[tmp >= 1e-7]) * np.sin(lat_p) * np.cos(lon_data[tmp >= 1e-7]),
    )
    lat_transformed[tmp >= 1e-7] = np.arcsin(sin_o[tmp >= 1e-7])

    lon_transformed[lon_transformed < 0] += 2 * np.pi
    lon_transformed[lon_transformed >= 2 * np.pi] -= 2 * np.pi

    if isinstance(lon, Quantity):
        lon_out = copy.deepcopy(lon)
        lat_out = copy.deepcopy(lat)

        lon_out.data[:] = lon_transformed
        lat_out.data[:] = lat_transformed
    else:
        lon_out = lon_transformed
        lat_out = lat_transformed

    return lon_out, lat_out  # type: ignore
