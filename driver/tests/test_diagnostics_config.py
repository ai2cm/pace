import unittest.mock

import pace.driver
from pace.driver.diagnostics import NullDiagnostics


def test_returns_null_diagnostics_if_no_path_given():
    config = pace.driver.DiagnosticsConfig(path=None, names=["foo"])
    assert isinstance(
        config.diagnostics_factory(
            unittest.mock.MagicMock(), unittest.mock.MagicMock()
        ),
        NullDiagnostics,
    )


def test_returns_zarr_diagnostics_if_path_given(tmpdir):
    config = pace.driver.DiagnosticsConfig(path=tmpdir, names=["foo"])
    with unittest.mock.patch(target="pace.driver.diagnostics.ZarrDiagnostics") as mock:
        config.diagnostics_factory(unittest.mock.MagicMock(), unittest.mock.MagicMock())
        mock.assert_called_once_with(
            path=tmpdir,
            names=["foo"],
            partitioner=unittest.mock.ANY,
            comm=unittest.mock.ANY,
        )
