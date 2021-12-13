import pytest

from pace.dsl.gt4py_utils import (
    make_storage_from_shape,
    make_storage_from_shape_uncached,
)


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


# having backend as an arg for tests means the test runs on each backend


def test_storage_is_cached(backend, shape, origin, init):
    outputs = []
    for _ in range(2):
        outputs.append(
            make_storage_from_shape(
                shape,
                origin=origin,
                init=init,
                cache_key="test-cached",
                backend=backend,
            )
        )
    assert outputs[0] is outputs[1]


def test_uncached_storage_is_not_cached(backend, shape, origin, init):
    outputs = []
    for _ in range(2):
        outputs.append(
            make_storage_from_shape_uncached(
                shape, origin=origin, init=init, backend=backend
            )
        )
    assert not (outputs[0] is outputs[1])


def test_cached_storage_on_different_lines_arent_same(backend, shape, origin, init):
    out1 = make_storage_from_shape(shape, origin=origin, init=init, backend=backend)
    out2 = make_storage_from_shape(shape, origin=origin, init=init, backend=backend)
    assert not (out1 is out2)
