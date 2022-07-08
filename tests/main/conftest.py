import pytest


@pytest.fixture()
def backend(pytestconfig):
    backend = pytestconfig.getoption("backend")
    return backend


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default="numpy")
