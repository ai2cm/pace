import functools
from typing import Union

from . import constants
from ._exceptions import OutOfBoundsError


def shift_boundary_slice_tuple(dims, origin, extent, boundary_type, slice_tuple):
    slice_list = []
    for dim, entry, origin_1d, extent_1d in zip(dims, slice_tuple, origin, extent):
        slice_list.append(
            _shift_boundary_slice(dim, origin_1d, extent_1d, boundary_type, entry)
        )
    return tuple(slice_list)


def bound_default_slice(slice_in, start=None, stop=None):
    if slice_in.start is not None:
        start = slice_in.start
    if slice_in.stop is not None:
        stop = slice_in.stop
    return slice(start, stop, slice_in.step)


def _shift_boundary_slice(dim, origin, extent, boundary_type, slice_object):
    """_get_boundary_slice for corner views"""
    start_offset, stop_offset = _get_offset(boundary_type, dim, origin, extent)
    if isinstance(slice_object, slice):
        if slice_object.start is not None:
            start = slice_object.start + start_offset
        else:
            start = slice_object.start
        if slice_object.stop is not None:
            stop = slice_object.stop + stop_offset
        else:
            stop = slice_object.stop
        return bound_default_slice(
            slice(start, stop, slice_object.step), origin, origin + extent
        )
    else:
        return slice_object + start_offset  # usually an integer


def _get_offset(boundary_type, dim, origin, extent):
    if boundary_type is constants.INTERIOR:
        return origin, origin + extent
    else:
        boundary_at_start = boundary_at_start_of_dim(boundary_type, dim)
        if boundary_at_start is None:  # default is to index within compute domain
            return origin, origin
        elif boundary_at_start:
            return origin, origin
        else:
            return origin + extent, origin + extent


@functools.lru_cache(maxsize=None)
def get_boundary_slice(dims, origin, extent, shape, boundary_type, n_halo, interior):
    boundary_slice = []
    for dim, origin_1d, extent_1d, shape_1d in zip(dims, origin, extent, shape):
        if dim in constants.INTERFACE_DIMS:
            n_overlap = 1
        else:
            n_overlap = 0
        n_points = n_halo
        at_start = boundary_at_start_of_dim(boundary_type, dim)
        if dim not in constants.HORIZONTAL_DIMS:
            start, stop = origin_1d, origin_1d + extent_1d
        elif at_start is None:
            start, stop = origin_1d, origin_1d + extent_1d
        elif at_start:
            edge_index = origin_1d
            if interior:
                edge_index += n_overlap
                start, stop = edge_index, edge_index + n_points
            else:
                start, stop = edge_index - n_points, edge_index
        else:
            edge_index = origin_1d + extent_1d
            if interior:
                edge_index -= n_overlap
                start, stop = edge_index - n_points, edge_index
            else:
                start, stop = edge_index, edge_index + n_points
        if start < 0:
            raise OutOfBoundsError(
                f"boundary slice extends past start of domain on dimension {dim}"
            )
        elif stop > shape_1d:
            raise OutOfBoundsError(
                f"boundary slice extends past end of domain on dimension {dim}"
            )
        else:
            boundary_slice.append(slice(start, stop))
    return tuple(boundary_slice)


def boundary_at_start_of_dim(boundary: int, dim: str) -> Union[bool, None]:
    """
    Return True if boundary is at the start of the dimension,
    False if at the end, None if the boundary does not align with the dimension.
    """
    return BOUNDARY_AT_START_OF_DIM_MAPPING[boundary].get(dim, None)


BOUNDARY_AT_START_OF_DIM_MAPPING = {
    constants.WEST: {constants.X_DIM: True, constants.X_INTERFACE_DIM: True},
    constants.EAST: {constants.X_DIM: False, constants.X_INTERFACE_DIM: False},
    constants.SOUTH: {constants.Y_DIM: True, constants.Y_INTERFACE_DIM: True},
    constants.NORTH: {constants.Y_DIM: False, constants.Y_INTERFACE_DIM: False},
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTHWEST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.WEST],
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTHEAST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.NORTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.EAST],
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTHWEST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.WEST],
}
BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTHEAST] = {
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.SOUTH],
    **BOUNDARY_AT_START_OF_DIM_MAPPING[constants.EAST],
}
