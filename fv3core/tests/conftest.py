import pytest


@pytest.fixture()
def backend(pytestconfig):
    backend = pytestconfig.getoption("backend")
    return backend


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default="numpy")
    parser.addoption("--which_modules", action="store")
    parser.addoption("--which_rank", action="store")
    parser.addoption("--skip_modules", action="store")
    parser.addoption("--print_failures", action="store_true")
    parser.addoption("--failure_stride", action="store", default=1)
    parser.addoption("--data_path", action="store", default="./")
    parser.addoption("--threshold_overrides_file", action="store", default=None)
    parser.addoption("--compute_grid", action="store_true")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "sequential(name): mark test as running sequentially on ranks"
    )
    config.addinivalue_line(
        "markers", "parallel(name): mark test as running in parallel across ranks"
    )
    config.addinivalue_line(
        "markers",
        "mock_parallel(name): mark test as running in mock parallel across ranks",
    )
