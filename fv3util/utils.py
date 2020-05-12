from . import constants
import numpy as np


def list_by_dims(dims, horizontal_list, non_horizontal_value):
    """Take in a list of dimensions, a (y, x) set of values, and a value for any
    non-horizontal dimensions. Return a list of length len(dims) with the value for
    each dimension.
    """
    return_list = []
    for dim in dims:
        if dim in constants.Y_DIMS:
            return_list.append(horizontal_list[0])
        elif dim in constants.X_DIMS:
            return_list.append(horizontal_list[1])
        else:
            return_list.append(non_horizontal_value)
    return tuple(return_list)


def is_contiguous(array):
    try:
        return array.data.contiguous
    except AttributeError:
        # gt4py storages use numpy arrays for .data attribute instead of memoryview
        return array.data.data.contiguous


def is_c_contiguous(array):
    try:
        return array.data.c_contiguous
    except AttributeError:
        # gt4py storages use numpy arrays for .data attribute instead of memoryview
        return array.data.data.c_contiguous


def ensure_contiguous(maybe_array):
    if isinstance(maybe_array, np.ndarray) and not is_contiguous(maybe_array):
        raise ValueError("ndarray is not contiguous")
