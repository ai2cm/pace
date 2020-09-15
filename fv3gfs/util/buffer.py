from typing import Callable, Iterable, Optional
from ._timing import Timer, NullTimer
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
def send_buffer(allocator: Callable, array: ndarray, timer: Optional[Timer] = None):
    """A context manager ensuring that `array` is contiguous in a context where it is
    being sent as data, copying into a recycled buffer array if necessary.

    Args:
        allocator: a function behaving like numpy.empty
        array: a possibly non-contiguous array for which to provide a buffer
        timer: object to accumulate timings for "pack"

    Yields:
        buffer_array: if array is non-contiguous, a contiguous buffer array containing
            the data from array. Otherwise, yields array.
    """
    if timer is None:
        timer = NullTimer()
    if array is None or is_c_contiguous(array):
        yield array
    else:
        timer.start("pack")
        with array_buffer(allocator, array.shape, array.dtype) as sendbuf:
            sendbuf[:] = array
            # this is a little dangerous, because if there is an exception in the two
            # lines above the timer may be started but never stopped. However, it
            # cannot be avoided because we cannot put those two lines in a with or
            # try block without also including the yield line.
            timer.stop("pack")
            yield sendbuf


@contextlib.contextmanager
def recv_buffer(allocator: Callable, array: ndarray, timer: Optional[Timer] = None):
    """A context manager ensuring that array is contiguous in a context where it is
    being used to receive data, using a recycled buffer array and then copying the
    result into array if necessary.

    Args:
        allocator: a function behaving like numpy.empty
        array: a possibly non-contiguous array for which to provide a buffer
        timer: object to accumulate timings for "unpack"

    Yields:
        buffer_array: if array is non-contiguous, a contiguous buffer array which is
            copied into array when the context is exited. Otherwise, yields array.
    """
    if timer is None:
        timer = NullTimer()
    if array is None or is_c_contiguous(array):
        yield array
    else:
        timer.start("unpack")
        with array_buffer(allocator, array.shape, array.dtype) as recvbuf:
            timer.stop("unpack")
            yield recvbuf
            with timer.clock("unpack"):
                array[:] = recvbuf
