import numpy as np
import pytest

import pace.util
import pace.util.quantity


try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None

requires_xarray = pytest.mark.skipif(xr is None, reason="xarray is not installed")


@pytest.fixture(params=["empty", "one", "five"])
def extent_1d(request, backend, n_halo):
    if request.param == "empty":
        if "gt4py" in backend and n_halo == 0:
            pytest.skip("gt4py does not support length-zero dimensions")
        else:
            return 0
    elif request.param == "one":
        return 1
    elif request.param == "five":
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
def dtype(numpy):
    return numpy.float64


@pytest.fixture
def units():
    return "m"


@pytest.fixture
def dims(n_dims):
    return tuple(f"dimension_{dim}" for dim in range(n_dims))


@pytest.fixture
def origin(n_halo, n_dims):
    return (n_halo,) * n_dims


@pytest.fixture
def data(n_halo, extent_1d, n_dims, numpy, dtype):
    shape = (n_halo * 2 + extent_1d,) * n_dims
    return numpy.empty(shape, dtype=dtype)


@pytest.fixture
def quantity(data, origin, extent, dims, units):
    return pace.util.Quantity(
        data, origin=origin, extent=extent, dims=dims, units=units
    )


def test_smaller_data_raises(data, origin, extent, dims, units):
    if len(data.shape) > 1:
        try:
            small_data = data[0]
        except IndexError:
            pass
        else:
            with pytest.raises(ValueError):
                pace.util.Quantity(
                    small_data, origin=origin, extent=extent, dims=dims, units=units
                )


def test_smaller_dims_raises(data, origin, extent, dims, units):
    with pytest.raises(ValueError):
        pace.util.Quantity(
            data, origin=origin, extent=extent, dims=dims[:-1], units=units
        )


def test_smaller_origin_raises(data, origin, extent, dims, units):
    with pytest.raises(ValueError):
        pace.util.Quantity(
            data, origin=origin[:-1], extent=extent, dims=dims, units=units
        )


def test_smaller_extent_raises(data, origin, extent, dims, units):
    with pytest.raises(ValueError):
        pace.util.Quantity(
            data, origin=origin, extent=extent[:-1], dims=dims, units=units
        )


def test_data_change_affects_quantity(data, quantity, numpy):
    data[:] = 5.0
    numpy.testing.assert_array_equal(quantity.data, 5.0)


def test_quantity_units(quantity, units):
    assert quantity.units == units
    assert quantity.attrs["units"] == units


def test_quantity_dims(quantity, dims):
    assert quantity.dims == dims


def test_quantity_origin(quantity, origin):
    assert quantity.origin == origin


def test_quantity_extent(quantity, extent):
    assert quantity.extent == extent


def test_compute_view_get_value(quantity, extent_1d, n_halo, n_dims):
    quantity.data[:] = 0.0
    if extent_1d == 0 and n_halo == 0:
        with pytest.raises(IndexError):
            quantity.view[[0] * n_dims]
    else:
        value = quantity.view[[0] * n_dims]
        assert value.shape == ()


def test_compute_view_edit_start_halo(quantity, extent_1d, n_halo, n_dims):
    quantity.data[:] = 0.0
    if extent_1d == 0 and n_halo == 0:
        with pytest.raises(IndexError):
            quantity.view[[-1] * n_dims] = 1
    else:
        quantity.view[[-1] * n_dims] = 1
        assert quantity.np.sum(quantity.data) == 1.0
        assert quantity.data[(n_halo - 1,) * n_dims] == 1


def test_compute_view_edit_end_halo(quantity, extent_1d, n_halo, n_dims):
    quantity.data[:] = 0.0
    if n_halo == 0:
        with pytest.raises(IndexError):
            quantity.view[[extent_1d] * n_dims] = 1
    else:
        quantity.view[(extent_1d,) * n_dims] = 1
        assert quantity.np.sum(quantity.data) == 1.0
        assert quantity.data[(n_halo + extent_1d,) * n_dims] == 1


def test_compute_view_edit_start_of_domain(quantity, extent_1d, n_halo, n_dims):
    if extent_1d == 0:
        pytest.skip("cannot edit an empty domain")
    quantity.data[:] = 0.0
    quantity.view[(0,) * n_dims] = 1
    assert quantity.data[(n_halo,) * n_dims] == 1
    assert quantity.np.sum(quantity.data) == 1.0


def test_compute_view_edit_all_domain(quantity, n_halo, n_dims, extent_1d):
    if extent_1d == 0:
        pytest.skip("cannot edit an empty domain")
    quantity.data[:] = 0.0
    quantity.view[:] = 1
    assert quantity.np.sum(quantity.data) == extent_1d ** n_dims
    if n_dims > 1:
        quantity.np.testing.assert_array_equal(quantity.data[:n_halo, :], 0.0)
        quantity.np.testing.assert_array_equal(
            quantity.data[n_halo + extent_1d :, :], 0.0
        )
    else:
        quantity.np.testing.assert_array_equal(quantity.data[:n_halo], 0.0)
        quantity.np.testing.assert_array_equal(quantity.data[n_halo + extent_1d :], 0.0)


@pytest.mark.parametrize(
    "slice_in, shift, extent, slice_out",
    [
        pytest.param(slice(0, 1), 0, 1, slice(0, 1), id="zero_shift"),
        pytest.param(slice(None, None), 1, 1, slice(None, None), id="shift_none_slice"),
        pytest.param(
            slice(None, 5),
            -1,
            5,
            slice(None, 4),
            id="shift_none_start",
        ),
        pytest.param(
            slice(-3, None),
            0,
            5,
            slice(2, None),
            id="negative_start",
        ),
        pytest.param(
            slice(-3, None),
            1,
            5,
            slice(3, None),
            id="shift_negative_start",
        ),
        pytest.param(
            slice(None, -1),
            0,
            5,
            slice(None, 4),
            id="negative_end",
        ),
        pytest.param(
            slice(0, -1),
            0,
            5,
            slice(0, 4),
            id="negative_end_with_none",
        ),
        pytest.param(
            slice(2, -2),
            1,
            5,
            slice(3, 4),
            id="shift_negative_end",
        ),
    ],
)
def test_shift_slice(slice_in, shift, extent, slice_out):
    result = pace.util.quantity.shift_slice(slice_in, shift, extent)
    assert result == slice_out


@pytest.mark.parametrize(
    "quantity",
    [
        pace.util.Quantity(
            np.array(5),
            dims=[],
            units="",
        ),
        pace.util.Quantity(
            np.array([1, 2, 3]),
            dims=["dimension"],
            units="degK",
        ),
        pace.util.Quantity(
            np.random.randn(3, 2, 4),
            dims=["dim1", "dim_2", "dimension_3"],
            units="m",
        ),
        pace.util.Quantity(
            np.random.randn(8, 6, 6),
            dims=["dim1", "dim_2", "dimension_3"],
            units="km",
            origin=(2, 2, 2),
            extent=(4, 2, 2),
        ),
    ],
)
@requires_xarray
def test_to_data_array(quantity):
    assert quantity.data_array.attrs == quantity.attrs
    assert quantity.data_array.dims == quantity.dims
    assert quantity.data_array.shape == quantity.extent
    np.testing.assert_array_equal(quantity.data_array.values, quantity.view[:])
    if quantity.extent == quantity.data.shape:
        assert (
            quantity.data_array.data.ctypes.data == quantity.data.ctypes.data
        ), "data memory address is not equal"
