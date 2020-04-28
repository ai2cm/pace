import contextlib
from .utils import is_contiguous

BUFFER_CACHE = {}


@contextlib.contextmanager
def array_buffer(allocator, shape, dtype):
    """
    A context manager providing a contiguous array, which may be re-used between calls.
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
def send_buffer(numpy, array):
    """A context manager ensuring that `array` is contiguous in a context where it is
    being sent as data, copying into a recycled buffer array if necessary.
    """
    if array is None or is_contiguous(array):
        yield array
    else:
        with array_buffer(numpy.empty, array.shape, array.dtype) as sendbuf:
            sendbuf[:] = array
            yield sendbuf


@contextlib.contextmanager
def recv_buffer(numpy, array):
    """A context manager ensuring that array is contiguous in a context where it is
    being used to receive data, using a recycled buffer array and then copying the
    result into array if necessary.
    """
    if array is None or is_contiguous(array):
        yield array
    else:
        with array_buffer(numpy.empty, array.shape, array.dtype) as recvbuf:
            yield recvbuf
            array[:] = recvbuf
