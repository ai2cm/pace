import numpy as np
import pytest

import pace.util


try:
    import gt4py
except ImportError:
    gt4py = None
try:
    import cupy
except ImportError:
    cupy = None


@pytest.fixture
def extent_1d():
    return 5


@pytest.fixture(params=[0, 3])
def n_halo(request):
    return request.param


@pytest.fixture(params=[3])
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
    return numpy.zeros(shape, dtype=dtype)


@pytest.fixture
def quantity(data, origin, extent, dims, units):
    return pace.util.Quantity(
        data, origin=origin, extent=extent, dims=dims, units=units
    )


def test_numpy(quantity, backend):
    if "cupy" in backend:
        assert quantity.np is cupy
    else:
        assert quantity.np is np


@pytest.mark.skipif(gt4py is None, reason="requires gt4py")
def test_modifying_numpy_storage_modifies_view():
    shape = (6, 6)
    data = np.zeros(shape, dtype=float)
    quantity = pace.util.Quantity(
        data,
        origin=(0, 0),
        extent=shape,
        dims=["dim1", "dim2"],
        units="units",
        gt4py_backend="gtc:numpy",
    )
    assert np.all(quantity.data == 0)
    quantity.storage[0, 0] = 1
    quantity.data[2, 2] = 5
    quantity.storage[4, 4] = 3
    assert quantity.view[0, 0] == 1
    assert quantity.view[2, 2] == 5
    assert quantity.view[4, 4] == 3
    assert quantity.data[0, 0] == 1
    assert quantity.storage[2, 2] == 5
    assert quantity.data[4, 4] == 3


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
def test_storage_exists(quantity, backend):
    if "numpy" in backend:
        assert isinstance(quantity.storage, gt4py.storage.storage.CPUStorage)
    else:
        assert isinstance(quantity.storage, gt4py.storage.storage.GPUStorage)


@pytest.mark.parametrize("backend", ["numpy", "cupy"], indirect=True)
def test_storage_does_not_exist(quantity, backend):
    with pytest.raises(TypeError):
        quantity.storage


def test_data_is_not_storage(quantity, backend):
    if gt4py is not None:
        assert not isinstance(quantity.data, gt4py.storage.storage.Storage)


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
def test_backend_is_accurate(quantity):
    assert quantity.gt4py_backend == quantity.storage.backend


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
def test_modifying_data_modifies_storage(quantity):
    quantity.storage[:] = 5
    assert quantity.np.all(quantity.data[:] == 5)


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
def test_modifying_storage_modifies_data(quantity):
    storage = quantity.storage
    quantity.data[:] = 5
    assert quantity.np.all(quantity.np.asarray(storage) == 5)
    assert quantity.data.data == quantity.storage.data.data


@pytest.mark.parametrize("backend", ["gt4py_numpy"], indirect=True)
def test_modifying_storage_modifies_data_when_initialized_from_storage(quantity):
    storage = quantity.storage
    quantity = pace.util.Quantity(
        storage,
        dims=quantity.dims,
        units=quantity.units,
        origin=quantity.origin,
        extent=quantity.extent,
    )
    quantity.data[:] = 5
    assert quantity.np.all(quantity.np.asarray(storage) == 5)
    assert quantity.data.data == quantity.storage.data.data


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
def test_modifying_storage_modifies_data_after_transpose(quantity):
    quantity = quantity.transpose(quantity.dims[::-1])
    storage = quantity.storage
    quantity.data[:] = 5
    assert quantity.np.all(quantity.np.asarray(storage) == 5)


@pytest.mark.parametrize("backend", ["numpy", "cupy"], indirect=True)
def test_accessing_storage_does_not_break_view(
    data, origin, extent, dims, units, gt4py_backend
):
    quantity = pace.util.Quantity(
        data,
        origin=origin,
        extent=extent,
        dims=dims,
        units=units,
        gt4py_backend=gt4py_backend,
    )
    quantity.storage[origin] = -1.0
    assert quantity.data[origin] == quantity.view[tuple(0 for _ in origin)]


# run using cupy backend even though unused, to mark this as a "gpu" test
@pytest.mark.parametrize("backend", ["cupy"], indirect=True)
def test_numpy_data_becomes_cupy_with_gpu_backend(
    data, origin, extent, dims, units, gt4py_backend
):
    cpu_data = np.zeros(data.shape)
    quantity = pace.util.Quantity(
        cpu_data,
        origin=origin,
        extent=extent,
        dims=dims,
        units=units,
        gt4py_backend=gt4py_backend,
    )
    assert isinstance(quantity.data, cupy.ndarray)
    assert isinstance(quantity.storage, gt4py.storage.storage.GPUStorage)


@pytest.mark.parametrize("backend", ["gt4py_numpy"], indirect=True)
def test_cannot_use_cpu_storage_with_gpu_backend(
    data, origin, extent, dims, units, gt4py_backend
):
    assert isinstance(data, gt4py.storage.storage.CPUStorage)
    with pytest.raises(TypeError):
        pace.util.Quantity(
            data,
            origin=origin,
            extent=extent,
            dims=dims,
            units=units,
            gt4py_backend=gt4py_backend,
        )


@pytest.mark.parametrize("backend", ["gt4py_cupy"], indirect=True)
def test_cannot_use_gpu_storage_with_cpu_backend(
    data, origin, extent, dims, units, gt4py_backend
):
    assert isinstance(data, gt4py.storage.storage.GPUStorage)
    with pytest.raises(TypeError):
        pace.util.Quantity(
            data,
            origin=origin,
            extent=extent,
            dims=dims,
            units=units,
            gt4py_backend=gt4py_backend,
        )
