import os
import fv3util
import pytest
import numpy as numpy_cpu


SKIP_TESTS = True
SCRIPT_FILENAME = os.path.basename(__file__)


@pytest.fixture(params=['empty', 'five'])
def extent_1d(request):
    if request.param == 'empty':
        return 0
    elif request.param == 'one':
        return 1
    elif request.param == 'five':
        return 5


@pytest.fixture(params=[0, 3])
def n_halo(request):
    return request.param


@pytest.fixture(params=[1, 2])
def n_dims(request):
    return request.param


@pytest.fixture
def extent(extent_1d, n_dims):
    return (extent_1d,) * n_dims


@pytest.fixture
def numpy():
    return numpy_cpu


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture
def units():
    return 'm'


@pytest.fixture
def dims(n_dims):
    return tuple(f'dimension_{dim}' for dim in range(n_dims))


@pytest.fixture
def origin(n_halo, n_dims):
    return (n_halo,) * n_dims


@pytest.fixture
def data(n_halo, extent_1d, n_dims, numpy, dtype):
    shape = (n_halo * 2 + extent_1d,) * n_dims
    return numpy.ones(shape, dtype=dtype)


@pytest.fixture
def quantity(data, origin, extent, dims, units):
    return fv3util.Quantity(data, origin=origin, extent=extent, dims=dims, units=units)


@pytest.fixture(params=[-1.0, 0.0, 1.0, -1, 0, 1, -5.5, 7.6])
def scalar(request):
    return request.param


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_add_scalar(quantity, scalar, numpy):
    result = quantity + scalar
    numpy.testing.assert_array_equal(result.view[:], quantity.view[:] + scalar)
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_left_add_scalar(quantity, scalar, numpy):
    result = scalar + quantity
    numpy.testing.assert_array_equal(result.view[:], scalar + quantity.view[:])
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_subtract_scalar(quantity, scalar, numpy):
    result = quantity - scalar
    numpy.testing.assert_array_equal(result.view[:], quantity.view[:] - scalar)
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_left_subtract_scalar(quantity, scalar, numpy):
    result = scalar - quantity
    numpy.testing.assert_array_equal(result.view[:], scalar - quantity.view[:])
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_multiply_scalar(quantity, scalar, numpy):
    result = quantity * scalar
    numpy.testing.assert_array_equal(result.view[:], quantity.view[:] * scalar)
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_left_multiply_scalar(quantity, scalar, numpy):
    result = scalar * quantity
    numpy.testing.assert_array_equal(result.view[:], scalar * quantity.view[:])
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_divide_scalar(quantity, scalar, numpy):
    if scalar == 0:
        pytest.skip()
    result = quantity / scalar
    numpy.testing.assert_array_equal(result.view[:], quantity.view[:] / scalar)
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_left_divide_scalar(quantity, scalar, numpy):
    result = scalar / quantity
    numpy.testing.assert_array_equal(result.view[:], scalar / quantity.view[:])
    assert result.dims == quantity.dims
    assert result.units == '1 / ' + quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_add_self(quantity, numpy):
    result = quantity + quantity
    numpy.testing.assert_array_equal(result.view[:], 2 * quantity.view[:])
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_subtract_self(quantity, numpy):
    result = quantity - quantity
    numpy.testing.assert_array_equal(result.view[:], 0.)
    assert result.dims == quantity.dims
    assert result.units == quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_multiply_self(quantity, numpy):
    result = quantity * quantity
    numpy.testing.assert_array_equal(result.view[:], quantity.view[:] * quantity.view[:])
    assert result.dims == quantity.dims
    assert result.units == quantity.units + ' * ' + quantity.units


@pytest.mark.skipif(SKIP_TESTS, reason=f'{SCRIPT_FILENAME} tests disabled by global variable')
def test_divide_self(quantity, numpy):
    result = quantity / quantity
    numpy.testing.assert_array_equal(result.view[:], quantity.view[:] / quantity.view)
    assert result.dims == quantity.dims
    assert result.units == ''
