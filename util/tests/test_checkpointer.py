import numpy as np
import pytest

import pace.util
from pace.util._optional_imports import xarray as xr


requires_xarray = pytest.mark.skipif(xr is None, reason="xarray is not installed")


@requires_xarray
def test_snapshot_checkpointer_no_data():
    checkpointer = pace.util.SnapshotCheckpointer(rank=0)
    xr.testing.assert_identical(checkpointer.dataset, xr.Dataset())


@requires_xarray
def test_snapshot_checkpointer_one_snapshot():
    checkpointer = pace.util.SnapshotCheckpointer(rank=0)
    val1 = np.random.randn(2, 3, 4)
    checkpointer("savepoint_name", val1=val1)
    xr.testing.assert_identical(
        checkpointer.dataset,
        xr.Dataset(
            data_vars={
                "val1": xr.DataArray(
                    val1[None, :],
                    dims=["sp_val1", "val1_dim0", "val1_dim1", "val1_dim2"],
                ),
                "val1_savepoints": xr.DataArray(["savepoint_name"], dims=["sp_val1"]),
            }
        ),
    )


@requires_xarray
def test_snapshot_checkpointer_multiple_snapshots():
    checkpointer = pace.util.SnapshotCheckpointer(rank=0)
    val1 = np.random.randn(2, 2, 3, 4)
    val2 = np.random.randn(1, 3, 2, 4)
    checkpointer("savepoint_name_1", val1=val1[0, :])
    checkpointer("savepoint_name_2", val1=val1[1, :], val2=val2[0, :])
    xr.testing.assert_identical(
        checkpointer.dataset,
        xr.Dataset(
            data_vars={
                "val1": xr.DataArray(
                    val1, dims=["sp_val1", "val1_dim0", "val1_dim1", "val1_dim2"]
                ),
                "val2": xr.DataArray(
                    val2, dims=["sp_val2", "val2_dim0", "val2_dim1", "val2_dim2"]
                ),
                "val1_savepoints": xr.DataArray(
                    ["savepoint_name_1", "savepoint_name_2"], dims=["sp_val1"]
                ),
                "val2_savepoints": xr.DataArray(["savepoint_name_2"], dims=["sp_val2"]),
            }
        ),
    )
