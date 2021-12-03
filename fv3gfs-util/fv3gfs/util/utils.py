from typing import Iterable, Sequence, Tuple, TypeVar, Union

import numpy as np

from . import constants
from .types import Allocator


try:
    import cupy as cp
except ImportError:
    cp = None

# Run a deviceSynchronize() to check
# that the GPU is present and ready to run
if cp:
    try:
        cp.cuda.runtime.deviceSynchronize()
        GPU_AVAILABLE = True
    except cp.cuda.runtime.CUDARuntimeError:
        GPU_AVAILABLE = False
else:
    GPU_AVAILABLE = False

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


def safe_assign_array(
    to_array: Union[np.ndarray, Storage], from_array: Union[np.ndarray, Storage]
):
    """Failproof assignment for array on different devices.

    The memory will be downloaded/uploaded from GPU if need be.

    Args:
        to_array: destination ndarray
        from_array: source ndarray
    """
    try:
        to_array[:] = from_array
    except (ValueError, TypeError):
        if cp and isinstance(to_array, cp.ndarray):
            to_array[:] = cp.asarray(from_array)
        elif cp and isinstance(from_array, cp.ndarray):
            to_array[:] = cp.asnumpy(from_array)
        else:
            raise


def device_synchronize():
    """Synchronize all memory communication"""
    if GPU_AVAILABLE:
        cp.cuda.runtime.deviceSynchronize()


def safe_mpi_allocate(
    allocator: Allocator, shape: Iterable[int], dtype: type
) -> np.ndarray:
    """Make sure the allocation use an allocator that works with MPI

    For G2G transfer, MPICH requires the allocation to not be done
    with managedmemory. Since we can't know what state `cupy` is in
    with switch for the default pooled allocator.

    If allocator comes from cupy, it must be cupy.empty or cupy.zeros.
    We raise a RuntimeError if a cupy array is allocated outside of
    the safe code path.

    Though the allocation _might_ be safe, the MPI crash that result
    from a managed memory allocation is non trivial and should be
    tightly controlled.
    """
    if cp and (allocator is cp.empty or allocator is cp.zeros):
        original_allocator = cp.cuda.get_allocator()
        cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)
        array = allocator(shape, dtype=dtype)  # type: np.ndarray
        cp.cuda.set_allocator(original_allocator)
    else:
        array = allocator(shape, dtype=dtype)
        if __debug__ and cp and isinstance(array, cp.ndarray):
            raise RuntimeError("cupy allocation might not be MPI-safe")
    return array
