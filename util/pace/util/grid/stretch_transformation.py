import copy
from typing import Tuple, Union

import numpy as np


try:
    import cupy as cp
except ImportError:
    cp = None
from pace.util import Quantity


def direct_transform(
    *,
    lon: Union[Quantity, np.ndarray, cp.ndarray],
    lat: Union[Quantity, np.ndarray, cp.ndarray],
    stretch_factor: np.float_,
    lon_target: np.float_,
    lat_target: np.float_,
    numpy_module,
) -> Tuple[
    Union[np.ndarray, Quantity, cp.ndarray], Union[np.ndarray, Quantity, cp.ndarray]
]:
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

    Returns:
        lon_transform (out) in radians
        lat_transform (out) in radians
    """

    if isinstance(lon, Quantity):
        lon_data = lon.data
        lat_data = lat.data
    elif isinstance(lon, numpy_module.ndarray):
        lon_data = lon
        lat_data = lat
    else:
        raise Exception("Input data type not supported.")

    STRETCH_GRID_ROTATION_LON_OFFSET_DEG = 190
    # this is added to all longitude values to match the SHiELD TC case
    # 180 is to flip the orientation around the center tile (6)
    # 10 is because the tile center is offset from the prime meridian by 10
    lon_data = lon_data + numpy_module.deg2rad(STRETCH_GRID_ROTATION_LON_OFFSET_DEG)

    lon_p, lat_p = numpy_module.deg2rad(lon_target), numpy_module.deg2rad(lat_target)
    sin_p, cos_p = numpy_module.sin(lat_p), numpy_module.cos(lat_p)
    c2p1 = 1.0 + stretch_factor ** 2
    c2m1 = 1.0 - stretch_factor ** 2

    # first limit longitude so it's between 0 and 2pi
    lon_data[lon_data < 0] += 2 * numpy_module.pi
    lon_data[lon_data >= 2 * numpy_module.pi] -= 2 * numpy_module.pi

    if numpy_module.abs(c2m1) > 1e-7:  # do stretching
        lat_t = numpy_module.arcsin(
            (c2m1 + c2p1 * numpy_module.sin(lat_data))
            / (c2p1 + c2m1 * numpy_module.sin(lat_data))
        )
    else:  # no stretching
        lat_t = lat_data

    sin_lat = numpy_module.sin(lat_t)
    cos_lat = numpy_module.cos(lat_t)

    sin_o = -(sin_p * sin_lat + cos_p * cos_lat * numpy_module.cos(lon_data))
    tmp = 1 - numpy_module.abs(sin_o)

    lon_trans = numpy_module.zeros(lon_data.shape) * numpy_module.nan
    lat_trans = numpy_module.zeros(lat_data.shape) * numpy_module.nan

    lon_trans[tmp < 1e-7] = 0.0
    lat_trans[tmp < 1e-7] = _sign(numpy_module.pi / 2, sin_o[tmp < 1e-7], numpy_module)

    lon_trans[tmp >= 1e-7] = lon_p + numpy_module.arctan2(
        -numpy_module.cos(lat_t[tmp >= 1e-7]) * numpy_module.sin(lon_data[tmp >= 1e-7]),
        -numpy_module.sin(lat_t[tmp >= 1e-7]) * numpy_module.cos(lat_p)
        + numpy_module.cos(lat_t[tmp >= 1e-7])
        * numpy_module.sin(lat_p)
        * numpy_module.cos(lon_data[tmp >= 1e-7]),
    )
    lat_trans[tmp >= 1e-7] = numpy_module.arcsin(sin_o[tmp >= 1e-7])

    lon_trans[lon_trans < 0] += 2 * numpy_module.pi
    lon_trans[lon_trans >= 2 * numpy_module.pi] -= 2 * numpy_module.pi

    if isinstance(lon, Quantity):
        lon_transform = copy.deepcopy(lon)
        lat_transform = copy.deepcopy(lat)

        lon_transform.data[:] = lon_trans
        lat_transform.data[:] = lat_trans

    elif isinstance(lon, numpy_module.ndarray):
        lon_transform = lon_trans
        lat_transform = lat_trans

    return lon_transform, lat_transform


def _sign(input, pn, numpy_module):
    """
    Use:
    output = sign(input, pn)

    Takes the sign of pn (positive or negative) and assigns it to value.

    Args:
        input (in): value to be assigned a sign
        pn (in): value whose sign is assigned to input
    Returns:
        output (out): value with assigned sign based on pn
    """
    if isinstance(input, float) and isinstance(pn, float):
        output = numpy_module.nan

        if pn >= 0:
            output = numpy_module.abs(input)
        else:
            output = -numpy_module.abs(input)

    elif isinstance(input, numpy_module.ndarray):
        tmp = numpy_module.abs(input)
        output = numpy_module.zeros(input.shape) * numpy_module.nan
        output[pn >= 0] = tmp[pn >= 0]
        output[pn < 0] = -tmp[pn < 0]

    elif isinstance(pn, numpy_module.ndarray) and isinstance(input, float):
        tmp = numpy_module.abs(input)
        output = numpy_module.zeros(pn.shape) * numpy_module.nan
        output[pn >= 0] = tmp
        output[pn < 0] = -tmp

    return output
