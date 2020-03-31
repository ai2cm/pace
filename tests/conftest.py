import pytest
import fv3util
import numpy as np

try:
    import gt4py
except ImportError:
    gt4py = None
try:
    import cupy
except ImportError:
    cupy = None


@pytest.fixture(params=["numpy", "cupy", "gt4py_numpy", "gt4py_cupy"])
def backend(request):
    if cupy is None and request.param.endswith("cupy"):
        pytest.skip("cupy is not available for GPU backend")
    elif gt4py is None and request.param.startswith("gt4py"):
        pytest.skip("gt4py backend is not available")
    else:
        return request.param


@pytest.fixture
def numpy(backend):
    if backend == "numpy":
        return np
    elif backend == "cupy":
        return cupy
    elif backend.startswith("gt4py"):
        if backend.endswith("numpy"):
            return fv3util.testing.gt4py_numpy
        elif backend.endswith("cupy"):
            return fv3util.testing.gt4py_cupy
    else:
        raise NotImplementedError()
