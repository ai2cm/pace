import pytest

import fv3core


@pytest.fixture()
def backend(pytestconfig):
    backend = pytestconfig.getoption("backend")
    fv3core.set_backend(backend)
    return backend


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default="numpy")
