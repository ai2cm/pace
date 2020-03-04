import pytest
import numpy as np
import fv3util
import fv3util.quantity


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
    return numpy.empty(shape, dtype=dtype)


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


def test_compute_view_get_value(quantity, n_halo, n_dims, extent_1d):
    if n_dims != 1 or extent_1d < 1:
        pytest.skip()
    quantity.data[:] = 0.
    value = quantity.view[0]
    assert value.shape == ()


def test_compute_view_edit_start_halo(quantity, n_halo):
    if n_dims != 1 or n_halo < 1:
        pytest.skip()
    quantity.data[:] = 0.
    quantity.view[-1] = 1
    assert quantity.data[n_halo - 1] == 1
    assert quantity.np.sum(quantity.data) == 1.0


def test_compute_view_edit_end_halo(quantity, extent_1d, n_halo, n_dims):
    if n_dims != 1 or n_halo < 1:
        pytest.skip()
    quantity.data[:] = 0.
    quantity.view[extent_1d] = 1
    print(quantity.data, extent_1d, n_halo, n_dims)
    assert quantity.data[n_halo + extent_1d] == 1
    assert quantity.np.sum(quantity.data) == 1.0


def test_compute_view_edit_domain(quantity, n_halo, n_dims, extent_1d):
    if n_dims != 1 or extent_1d < 1 or n_halo < 1:
        pytest.skip()
    quantity.data[:] = 0.
    quantity.view[0] = 1
    assert quantity.data[n_halo] == 1
    assert quantity.np.sum(quantity.data) == 1.0


def test_compute_view_edit_all_domain(quantity, n_halo, n_dims, extent_1d):
    if n_dims != 1 or extent_1d < 1 or n_halo < 1:
        pytest.skip()
    quantity.data[:] = 0.
    quantity.view[:] = 1
    assert quantity.np.sum(quantity.data) == extent_1d ** n_dims
    if n_dims > 1:
        quantity.np.testing.assert_array_equal(quantity.data[:n_halo, :], 0.)
        quantity.np.testing.assert_array_equal(quantity.data[n_halo + extent_1d:, :], 0.)
    else:
        quantity.np.testing.assert_array_equal(quantity.data[:n_halo], 0.)
        quantity.np.testing.assert_array_equal(quantity.data[n_halo + extent_1d:], 0.)


@pytest.mark.parametrize(
    'slice_in, shift, slice_out',
    [
        pytest.param(
            slice(0, 1), 0, slice(0, 1),
            id='zero_shift'
        ),
        pytest.param(
            slice(None, None), 1, slice(None, None),
            id='shift_none_slice'
        ),
        pytest.param(
            slice(None, 5), -1, slice(None, 4),
            id='shift_none_start',
        ),
    ]
)
def test_shift_slice(slice_in, shift, slice_out):
    result = fv3util.quantity.shift_slice(slice_in, shift)
    assert result == slice_out
