import pytest

from pace.util.buffer import BUFFER_CACHE, Buffer, recv_buffer, send_buffer
from pace.util.utils import is_c_contiguous, is_contiguous


@pytest.fixture
def contiguous_array(numpy, backend):
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    array = numpy.empty([3, 4, 5])
    array[:] = numpy.random.randn(3, 4, 5)
    return array


@pytest.fixture
def non_contiguous_array(contiguous_array):
    return contiguous_array.transpose(2, 0, 1)


@pytest.fixture(params=["empty", "zeros"])
def allocator(request, numpy):
    return getattr(numpy, request.param)


def test_is_contiguous(contiguous_array):
    assert is_contiguous(contiguous_array)


def test_is_c_contiguous(contiguous_array):
    assert is_c_contiguous(contiguous_array)


def test_not_is_contiguous(non_contiguous_array):
    assert not is_contiguous(non_contiguous_array)


def test_not_is_c_contiguous(non_contiguous_array):
    assert not is_c_contiguous(non_contiguous_array)


def test_sendbuf_uses_buffer(numpy, backend, allocator, non_contiguous_array):
    with send_buffer(allocator, non_contiguous_array) as sendbuf:
        assert sendbuf is not non_contiguous_array
        assert sendbuf.data is not non_contiguous_array.data
        numpy.testing.assert_array_equal(sendbuf, non_contiguous_array)


def test_recvbuf_uses_buffer(numpy, allocator, non_contiguous_array):
    with recv_buffer(allocator, non_contiguous_array) as recvbuf:
        assert recvbuf is not non_contiguous_array
        assert recvbuf.data is not non_contiguous_array.data
        recvbuf[:] = 0.0
        assert not numpy.all(non_contiguous_array == 0.0)
    assert numpy.all(non_contiguous_array == 0.0)


def test_sendbuf_no_buffer(allocator, contiguous_array):
    with send_buffer(allocator, contiguous_array) as sendbuf:
        assert sendbuf is contiguous_array


def test_recvbuf_no_buffer(allocator, contiguous_array):
    with recv_buffer(allocator, contiguous_array) as recvbuf:
        assert recvbuf is contiguous_array


def test_buffer_cache_appends(allocator, backend):
    """
    Test buffer with the same key are appended while not in use for potential reuse
    """
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    BUFFER_CACHE.clear()
    # Cache is cleared - no cache line
    assert len(BUFFER_CACHE) == 0
    shape = (10, 10, 10)
    # Pop two buffers with the same key - this creates a cache line for the key
    first_buffer = Buffer.pop_from_cache(allocator, shape, float)
    first_buffer.array.fill(42)
    assert len(BUFFER_CACHE) == 1
    second_buffer = Buffer.pop_from_cache(allocator, shape, float)
    second_buffer.array.fill(23)
    assert first_buffer._key == second_buffer._key
    assert (first_buffer.array != second_buffer.array).all()
    assert len(BUFFER_CACHE) == 1
    assert len(BUFFER_CACHE[first_buffer._key]) == 0
    # Pushing back the buffers, the cache line should have two items
    Buffer.push_to_cache(first_buffer)
    Buffer.push_to_cache(second_buffer)
    assert len(BUFFER_CACHE[first_buffer._key]) == 2


def test_buffer_reuse(allocator, backend):
    """Test we reuse the buffer when available instead of reallocating one"""
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    BUFFER_CACHE.clear()
    # Cache is cleared - no cache line
    assert len(BUFFER_CACHE) == 0
    shape = (10, 10, 10)
    # We popped a buffer from the cache. This created a cache line for key
    # first_buffer._key.
    # That cache line is an empty array for now (the element was popped)
    first_buffer = Buffer.pop_from_cache(allocator, shape, float)
    fill_scalar = 42
    first_buffer.array.fill(fill_scalar)
    assert len(BUFFER_CACHE) == 1
    assert len(BUFFER_CACHE[first_buffer._key]) == 0
    # Pushing back - the cache line array as the first_buffer in it, if we
    # re-pop it we should have the fill value (same buffer, no re-alloc)
    Buffer.push_to_cache(first_buffer)
    assert len(BUFFER_CACHE) == 1
    assert len(BUFFER_CACHE[first_buffer._key]) == 1
    repop_buffer = Buffer.pop_from_cache(allocator, shape, float)
    assert len(BUFFER_CACHE[first_buffer._key]) == 0
    assert (repop_buffer.array == fill_scalar).all()
    # Clean up
    Buffer.push_to_cache(repop_buffer)


def test_cacheline_differentiation(allocator, backend):
    """Test allocation with different keys creates different cache lines"""
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    BUFFER_CACHE.clear()
    # Cache is cleared - no cache line
    assert len(BUFFER_CACHE) == 0
    shape = (10, 10, 10)
    # Pop a float buffer - create a cache line for the triplet of parameters
    first_buffer = Buffer.pop_from_cache(allocator, shape, float)
    first_fill_scalar = 42
    first_buffer.array.fill(first_fill_scalar)
    assert len(BUFFER_CACHE) == 1
    assert len(BUFFER_CACHE[first_buffer._key]) == 0
    # Pop an int buffer - create a second cache line for the triplet of parameters
    second_buffer = Buffer.pop_from_cache(allocator, shape, int)
    second_fill_scalar = 44
    second_buffer.array.fill(second_fill_scalar)
    assert len(BUFFER_CACHE) == 2
    assert len(BUFFER_CACHE[second_buffer._key]) == 0
    # Check buffer are different
    assert first_buffer._key != second_buffer._key
    assert (first_buffer.array != second_buffer.array).all()
    # Pushing back - the cache line get their buffer back
    Buffer.push_to_cache(first_buffer)
    Buffer.push_to_cache(second_buffer)
    assert len(BUFFER_CACHE) == 2
    assert len(BUFFER_CACHE[first_buffer._key]) == 1
    assert len(BUFFER_CACHE[second_buffer._key]) == 1
    # We pop back the buffer and expect to get the previously fill'ed buffers
    repop_first_buffer = Buffer.pop_from_cache(allocator, shape, float)
    assert len(BUFFER_CACHE[repop_first_buffer._key]) == 0
    assert len(BUFFER_CACHE[second_buffer._key]) == 1
    repop_second_buffer = Buffer.pop_from_cache(allocator, shape, int)
    assert len(BUFFER_CACHE[repop_first_buffer._key]) == 0
    assert len(BUFFER_CACHE[repop_second_buffer._key]) == 0
    assert (repop_first_buffer.array == first_fill_scalar).all()
    assert (repop_second_buffer.array == second_fill_scalar).all()
    # Clean up
    Buffer.push_to_cache(repop_first_buffer)
    Buffer.push_to_cache(repop_second_buffer)


@pytest.mark.parametrize(
    "first_args, second_args",
    [
        pytest.param(((10, 10, 10), float), ((10, 10, 10), int), id="different_dtype"),
        pytest.param(((10, 10, 10), float), ((10, 10, 5), float), id="different_shape"),
    ],
)
def test_new_args_gives_different_buffer(allocator, backend, first_args, second_args):
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    BUFFER_CACHE.clear()
    first_buffer = Buffer.pop_from_cache(allocator, *first_args)
    Buffer.push_to_cache(first_buffer)
    second_buffer = Buffer.pop_from_cache(allocator, *second_args)
    assert not (first_buffer is second_buffer)
    assert first_buffer._key != second_buffer._key
    first_buffer.array[:] = 10.0
    second_buffer.array[:] = 1.0
    assert (first_buffer.array == 10.0).all()
    assert (second_buffer.array == 1.0).all()


@pytest.mark.parametrize("allocator, backend", [["ones", "cupy"]], indirect=True)
def test_mpi_unsafe_allocator_exception(backend, allocator):
    BUFFER_CACHE.clear()
    print(allocator)
    with pytest.raises(RuntimeError):
        Buffer.pop_from_cache(allocator, shape=(10, 10, 10), dtype=float)
