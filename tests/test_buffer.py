import pytest
from fv3util.buffer import send_buffer, recv_buffer
import numpy as np


@pytest.fixture
def contiguous_array(numpy):
    array = numpy.empty([3, 4, 5])
    array[:] = np.random.randn(3, 4, 5)
    return array


@pytest.fixture
def non_contiguous_array(contiguous_array):
    return contiguous_array.transpose(2, 0, 1)


def test_sendbuf_uses_buffer(numpy, non_contiguous_array):
    with send_buffer(numpy, non_contiguous_array) as sendbuf:
        assert sendbuf is not non_contiguous_array
        assert sendbuf.data is not non_contiguous_array.data
        numpy.testing.assert_array_equal(sendbuf, non_contiguous_array)


def test_recvbuf_uses_buffer(numpy, non_contiguous_array):
    with recv_buffer(numpy, non_contiguous_array) as recvbuf:
        assert recvbuf is not non_contiguous_array
        assert recvbuf.data is not non_contiguous_array.data
        recvbuf[:] = 0.0
        assert not numpy.all(non_contiguous_array == 0.0)
    assert numpy.all(non_contiguous_array == 0.0)


def test_sendbuf_no_buffer(numpy, contiguous_array):
    with send_buffer(numpy, contiguous_array) as sendbuf:
        assert sendbuf is contiguous_array


def test_recvbuf_no_buffer(numpy, contiguous_array):
    with recv_buffer(numpy, contiguous_array) as recvbuf:
        assert recvbuf is contiguous_array
