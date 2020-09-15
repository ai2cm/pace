from . import constants
import numpy as np

try:
    from gt4py.storage.storage import Storage
except ImportError:

    class Storage:
        pass


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
    if isinstance(array, Storage):
        # gt4py storages use numpy arrays for .data attribute instead of memoryvie
        return array.data.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]
    else:
        return array.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]


def is_c_contiguous(array):
    if isinstance(array, Storage):
        # gt4py storages use numpy arrays for .data attribute instead of memoryview
        return array.data.flags["C_CONTIGUOUS"]
    else:
        return array.flags["C_CONTIGUOUS"]


def ensure_contiguous(maybe_array):
    if isinstance(maybe_array, np.ndarray) and not is_contiguous(maybe_array):
        raise ValueError("ndarray is not contiguous")
