import os

import pytest


DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture()
def backend(pytestconfig):
    backend = pytestconfig.getoption("backend")
    return backend


@pytest.fixture()
def data_path(pytestconfig):
    data_path = pytestconfig.getoption("data_path")
    return data_path


@pytest.fixture()
def threshold_path(pytestconfig):
    threshold_path = pytestconfig.getoption("threshold_path")
    if threshold_path is None:
        threshold_path = os.path.join(DIR, "thresholds")
    return threshold_path


@pytest.fixture()
def calibrate_thresholds(pytestconfig):
    calibrate_thresholds = pytestconfig.getoption("calibrate_thresholds")
    return calibrate_thresholds


def pytest_addoption(parser):
    parser.addoption(
        "--backend", action="store", default="numpy", help="gt4py backend name"
    )
    parser.addoption(
        "--data_path", action="store", default="./", help="location of reference data"
    )
    parser.addoption(
        "--threshold_path",
        action="store",
        default=None,
        help="directory containing comparison thresholds for tests",
    )
    parser.addoption(
        "--calibrate_thresholds",
        action="store_true",
        default=False,
        help="re-calibrate error thresholds for comparison to reference",
    )
