import copy as cp
from typing import Tuple, Union

import numpy as np

from pace.util import Quantity


def direct_transform(
    *,
    lon: Quantity,
    lat: Quantity,
    stretch_factor: float,
    lon_target: float,
    lat_target: float,
) -> Tuple[Union[np.ndarray, Quantity], Union[np.ndarray, Quantity]]:
    """
    The direct_transform subroutine from fv_grid_utils.F90.
    Takes in latitude and longitude in radians.
    Shrinks tile 6 by stretch factor in area to increse resolution locally.
    Then performs translation of all tiles so that the now-smaller tile 6 is
    centeres on lon_target, lat_target.

    Inputs:
    - lon in radians
    - lat in radians
    - stretch_factor (e.g. 3.0 means that the resolution on
        tile 6 becomes 3 times as fine)
    - lon_target in degrees (from namelist)
    - lat_target in degrees (from namelist)

    Outputs:
    - lon_transform in radians
    - lat_transform in radians
    """

    if isinstance(lon, Quantity):
        lon_data = lon.data
        lat_data = lat.data
    elif isinstance(lon, np.ndarray):
        lon_data = lon
        lat_data = lat

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
        lon_t = lon_data
    else:  # no stretching
        lat_t = lat_data
        lon_t = lon_data

    sin_lat = np.sin(lat_t)
    cos_lat = np.cos(lat_t)

    sin_o = -(sin_p * sin_lat + cos_p * cos_lat * np.cos(lon_data))
    tmp = 1 - np.abs(sin_o)

    lon_trans = np.zeros(lon_data.shape) * np.nan
    lat_trans = np.zeros(lat_data.shape) * np.nan

    lon_trans[tmp < 1e-7] = 0.0
    lat_trans[tmp < 1e-7] = _sign(np.pi / 2, sin_o[tmp < 1e-7])

    lon_trans[tmp >= 1e-7] = lon_p + np.arctan2(
        -np.cos(lat_t[tmp >= 1e-7]) * np.sin(lon_data[tmp >= 1e-7]),
        -np.sin(lat_t[tmp >= 1e-7]) * np.cos(lat_p)
        + np.cos(lat_t[tmp >= 1e-7]) * np.sin(lat_p) * np.cos(lon_data[tmp >= 1e-7]),
    )
    lat_trans[tmp >= 1e-7] = np.arcsin(sin_o[tmp >= 1e-7])

    lon_trans[lon_trans < 0] += 2 * np.pi
    lon_trans[lon_trans >= 2 * np.pi] -= 2 * np.pi

    if isinstance(lon, Quantity):
        lon_transform = cp.deepcopy(lon)
        lat_transform = cp.deepcopy(lat)

        lon_transform.data[:] = lon_trans
        lat_transform.data[:] = lat_trans

    elif isinstance(lon, np.ndarray):
        lon_transform = lon_trans
        lat_transform = lat_trans

    return lon_transform, lat_transform


def _sign(input, pn):
    """
    Use:
    output = sign(input, pn)

    Takes the sign of pn (positive or negative) and assigns it to value.

    Inputs:
    - input: value to be assigned a sign
    - pn: value whose sign is assigned to input
    """
    if isinstance(input, float) and isinstance(pn, float):
        output = np.nan

        if pn >= 0:
            output = np.abs(input)
        else:
            output = -np.abs(input)

    elif isinstance(input, np.ndarray):
        tmp = np.abs(input)
        output = np.zeros(input.shape) * np.nan
        output[pn >= 0] = tmp[pn >= 0]
        output[pn < 0] = -tmp[pn < 0]

    elif isinstance(pn, np.ndarray) and isinstance(input, float):
        tmp = np.abs(input)
        output = np.zeros(pn.shape) * np.nan
        output[pn >= 0] = tmp
        output[pn < 0] = -tmp

    return output
