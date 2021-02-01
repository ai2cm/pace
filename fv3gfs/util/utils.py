from typing import Union, Sequence, TypeVar, Tuple
from . import constants
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

try:
    from gt4py.storage.storage import Storage
except ImportError:

    class Storage:  # type: ignore[no-redef]
        pass


T = TypeVar("T")


def list_by_dims(
    dims: Sequence[str], horizontal_list: Sequence[T], non_horizontal_value: T
) -> Tuple[T, ...]:
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


def is_contiguous(array: Union[np.ndarray, Storage]) -> bool:
    if isinstance(array, Storage):
        # gt4py storages use numpy arrays for .data attribute instead of memoryvie
        return array.data.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]
    else:
        return array.flags["C_CONTIGUOUS"] or array.flags["F_CONTIGUOUS"]


def is_c_contiguous(array: Union[np.ndarray, Storage]) -> bool:
    if isinstance(array, Storage):
        # gt4py storages use numpy arrays for .data attribute instead of memoryview
        return array.data.flags["C_CONTIGUOUS"]
    else:
        return array.flags["C_CONTIGUOUS"]


def ensure_contiguous(maybe_array: Union[np.ndarray, Storage]) -> None:
    if isinstance(maybe_array, np.ndarray) and not is_contiguous(maybe_array):
        raise ValueError("ndarray is not contiguous")


def assign_array(
    to_array: Union[np.ndarray, Storage], from_array: Union[np.ndarray, Storage],
):
    # The cp.asarray call is required to explicitly copy the data
    # in the case of numpy arrays or to prevent memory ownership
    # errors on the cupy arrays from gt4py storages.
    # This should be fixed in a later version of gt4py
    if cp and isinstance(to_array, cp.ndarray):
        to_array[:] = cp.asarray(from_array)
    elif cp and isinstance(from_array, cp.ndarray):
        to_array[:] = cp.asnumpy(from_array)
    else:
        to_array[:] = from_array
