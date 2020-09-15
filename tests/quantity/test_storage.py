import pytest
import fv3gfs.util
import numpy as np

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
    return fv3gfs.util.Quantity(
        data, origin=origin, extent=extent, dims=dims, units=units
    )


def test_numpy(quantity, backend):
    if "cupy" in backend:
        assert quantity.np is cupy
    else:
        assert quantity.np is np


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
def test_storage_exists(quantity, backend):
    if "numpy" in backend:
        assert isinstance(quantity.storage, gt4py.storage.storage.CPUStorage)
    else:
        assert isinstance(quantity.storage, gt4py.storage.storage.GPUStorage)


@pytest.mark.parametrize("backend", ["numpy", "cupy"], indirect=True)
def test_storage_does_not_exist(quantity, backend):
    if gt4py is None:
        with pytest.raises(ImportError):
            quantity.storage
    else:
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
    quantity.data[:] = 5
    assert quantity.np.all(quantity.np.asarray(quantity.storage[:]) == 5)
