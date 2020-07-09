from typing import Callable, Iterable
from numpy import ndarray
import contextlib
from .utils import is_c_contiguous

BUFFER_CACHE = {}


@contextlib.contextmanager
def array_buffer(allocator: Callable, shape: Iterable[int], dtype: type):
    """
    A context manager providing a contiguous array, which may be re-used between calls.

    Args:
        allocator: a function with the same signature as numpy.zeros which returns
            an ndarray
        shape: the shape of the desired array
        dtype: the dtype of the desired array

    Yields:
        buffer_array: an ndarray created according to the specification in the args.
            May be retained and re-used in subsequent calls.
    """
    key = (allocator, shape, dtype)
    if key in BUFFER_CACHE and len(BUFFER_CACHE[key]) > 0:
        array = BUFFER_CACHE[key].pop()
        yield array
    else:
        if key not in BUFFER_CACHE:
            BUFFER_CACHE[key] = []
        array = allocator(shape, dtype=dtype)
        yield array
    BUFFER_CACHE[key].append(array)


@contextlib.contextmanager
def send_buffer(allocator: Callable, array: ndarray):
    """A context manager ensuring that `array` is contiguous in a context where it is
    being sent as data, copying into a recycled buffer array if necessary.

    Args:
        allocator: a function behaving like numpy.empty
        array: a possibly non-contiguous array for which to provide a buffer

    Yields:
        buffer_array: if array is non-contiguous, a contiguous buffer array containing
            the data from array. Otherwise, yields array.
    """
    if array is None or is_c_contiguous(array):
        yield array
    else:
        with array_buffer(allocator, array.shape, array.dtype) as sendbuf:
            sendbuf[:] = array
            yield sendbuf


@contextlib.contextmanager
def recv_buffer(allocator: Callable, array: ndarray):
    """A context manager ensuring that array is contiguous in a context where it is
    being used to receive data, using a recycled buffer array and then copying the
    result into array if necessary.

    Args:
        allocator: a function behaving like numpy.empty
        array: a possibly non-contiguous array for which to provide a buffer

    Yields:
        buffer_array: if array is non-contiguous, a contiguous buffer array which is
            copied into array when the context is exited. Otherwise, yields array.
    """
    if array is None or is_c_contiguous(array):
        yield array
    else:
        with array_buffer(allocator, array.shape, array.dtype) as recvbuf:
            yield recvbuf
            array[:] = recvbuf
