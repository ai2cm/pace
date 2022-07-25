import unittest.mock

import pytest

import pace.driver
from pace.driver.diagnostics import NullDiagnostics


def test_returns_null_diagnostics_if_no_path_given():
    config = pace.driver.DiagnosticsConfig(path=None, names=[])
    assert isinstance(
        config.diagnostics_factory(
            unittest.mock.MagicMock(), unittest.mock.MagicMock()
        ),
        NullDiagnostics,
    )


def test_returns_zarr_diagnostics_if_path_given(tmpdir):
    config = pace.driver.DiagnosticsConfig(
        path=tmpdir, names=["foo"], derived_names=["bar"]
    )
    with unittest.mock.patch(target="pace.driver.diagnostics.ZarrDiagnostics") as mock:
        config.diagnostics_factory(unittest.mock.MagicMock(), unittest.mock.MagicMock())
        mock.assert_called_once_with(
            path=tmpdir,
            names=["foo"],
            derived_names=["bar"],
            partitioner=unittest.mock.ANY,
            comm=unittest.mock.ANY,
        )


def test_raises_if_names_given_but_no_path():
    with pytest.raises(ValueError):
        pace.driver.DiagnosticsConfig(path=None, names=["foo"])

    with pytest.raises(ValueError):
        pace.driver.DiagnosticsConfig(path=None, derived_names=["foo"])
