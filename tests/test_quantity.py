import pytest
import numpy as np
import fv3util


@pytest.fixture(params=['empty', 'one', 'five'])
def extent_1d(request):
    if request.param == 'empty':
        return 0
    elif request.param == 'one':
        return 1
    elif request.param == 'five':
        return 5


@pytest.fixture(params=[0, 1, 3])
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
    return np


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
    return np.empty(shape, dtype=dtype)


@pytest.fixture
def quantity(data, origin, extent, dims, units):
    return fv3util.Quantity(data, origin=origin, extent=extent, dims=dims, units=units)


def test_data_change_affects_quantity(data, quantity, numpy):
    data[:] = 5.0
    numpy.testing.assert_array_equal(quantity.data, 5.0)


def test_quantity_units(quantity, units):
    assert quantity.units == units
    assert quantity.attrs['units'] == units


def test_quantity_dims(quantity, dims):
    assert quantity.dims == dims


def test_quantity_origin(quantity, origin):
    assert quantity.origin == origin


def test_quantity_extent(quantity, extent):
    assert quantity.extent == extent

