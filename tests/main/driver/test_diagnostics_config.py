import unittest.mock

import pytest

import pace.driver
import pace.driver.diagnostics


def test_returns_null_diagnostics_if_no_path_given():
    config = pace.driver.DiagnosticsConfig(path=None, names=[])
    assert isinstance(
        config.diagnostics_factory(unittest.mock.MagicMock()),
        pace.driver.diagnostics.NullDiagnostics,
    )


def test_returns_monitor_diagnostics_if_path_given(tmpdir):
    config = pace.driver.DiagnosticsConfig(
        path=tmpdir, names=["foo"], derived_names=["bar"]
    )
    result = config.diagnostics_factory(unittest.mock.MagicMock())
    assert isinstance(result, pace.driver.diagnostics.MonitorDiagnostics)


def test_raises_if_names_given_but_no_path():
    with pytest.raises(ValueError):
        pace.driver.DiagnosticsConfig(path=None, names=["foo"])

    with pytest.raises(ValueError):
        pace.driver.DiagnosticsConfig(path=None, derived_names=["foo"])
