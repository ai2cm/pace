import pytest

from pace.dsl.gt4py_utils import make_storage_from_shape


@pytest.fixture
def shape():
    return (3, 4, 5)


@pytest.fixture
def origin():
    return (0, 0, 0)


@pytest.fixture(params=("zeros", "empty"))
def init(request):
    if request.param == "zeros":
        return True
    elif request.param == "empty":
        return False
    else:
        raise NotImplementedError(request.param)


def test_uncached_storage_is_not_cached(backend, shape, origin, init):
    outputs = []
    for _ in range(2):
        outputs.append(
            make_storage_from_shape(shape, origin=origin, init=init, backend=backend)
        )
    assert not (outputs[0] is outputs[1])
