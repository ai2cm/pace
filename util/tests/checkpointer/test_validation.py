import os
import tempfile

import numpy as np
import pytest

from pace.util._optional_imports import xarray as xr
from pace.util.checkpointer import (
    SavepointThresholds,
    Threshold,
    ValidationCheckpointer,
)
from pace.util.checkpointer.validation import _clip_pace_array_to_target


requires_xarray = pytest.mark.skipif(xr is None, reason="xarray is not installed")


def get_dataset(
    n_savepoints: int, n_vars: int, n_ranks: int, nx: int, ny: int, nz: int
):
    data_vars = {}
    for i in range(n_vars):
        data_vars["data{}".format(i)] = xr.DataArray(
            np.zeros((n_savepoints, n_ranks, nx, ny, nz)),
            dims=[
                "savepoint",
                "rank",
                "dim_data{}_0".format(i),
                "dim_data{}_1".format(i),
                "dim_data{}_2".format(i),
            ],
        )
    return xr.Dataset(data_vars=data_vars)


@requires_xarray
def test_validation_validates_onevar_onecall():
    temp_dir = tempfile.TemporaryDirectory()
    nx_compute = 12
    nz = 20
    n_halo = 3
    savepoint_name = "savepoint_name"
    ds = get_dataset(
        n_savepoints=1, n_vars=1, n_ranks=1, nx=nx_compute, ny=nx_compute, nz=nz
    )
    ds["data0"].values[:] = 1.0
    ds.to_netcdf(os.path.join(temp_dir.name, savepoint_name + ".nc"))

    data = np.full(
        (nx_compute + 2 * n_halo + 1, nx_compute + 2 * n_halo + 1, nz), fill_value=2.0
    )

    checkpointer = ValidationCheckpointer(
        temp_dir.name,
        SavepointThresholds(
            {savepoint_name: [{"data0": Threshold(relative=1.0, absolute=1.0)}]}
        ),
        rank=0,
    )
    with checkpointer.trial():
        checkpointer(savepoint_name, data0=data)


@pytest.mark.parametrize(
    "relative_threshold, absolute_threshold",
    [
        pytest.param(0.99, 1.0, id="relative_failure"),
        pytest.param(1.0, 0.99, id="absolute_failure"),
    ],
)
@requires_xarray
def test_validation_asserts_onevar_onecall(relative_threshold, absolute_threshold):
    temp_dir = tempfile.TemporaryDirectory()
    nx_compute = 12
    nz = 20
    n_halo = 3
    savepoint_name = "savepoint_name"
    ds = get_dataset(
        n_savepoints=1, n_vars=1, n_ranks=1, nx=nx_compute, ny=nx_compute, nz=nz
    )
    ds["data0"].values[:] = 1.0
    ds.to_netcdf(os.path.join(temp_dir.name, savepoint_name + ".nc"))

    data = np.full(
        (nx_compute + 2 * n_halo + 1, nx_compute + 2 * n_halo + 1, nz), fill_value=2.0
    )

    checkpointer = ValidationCheckpointer(
        temp_dir.name,
        SavepointThresholds(
            {
                savepoint_name: [
                    {
                        "data0": Threshold(
                            relative=relative_threshold,
                            absolute=absolute_threshold,
                        )
                    }
                ]
            }
        ),
        rank=0,
    )
    with checkpointer.trial():
        with pytest.raises(AssertionError):
            checkpointer(savepoint_name, data0=data)


@pytest.mark.parametrize(
    "relative_threshold, absolute_threshold",
    [
        pytest.param(0.99, 1.0, id="relative_threshold"),
        pytest.param(1.0, 0.99, id="absolute_threshold"),
    ],
)
@requires_xarray
def test_validation_passes_onevar_two_calls(relative_threshold, absolute_threshold):
    temp_dir = tempfile.TemporaryDirectory()
    nx_compute = 12
    nz = 20
    n_halo = 3
    savepoint_name = "savepoint_name"
    ds = get_dataset(
        n_savepoints=2, n_vars=1, n_ranks=1, nx=nx_compute, ny=nx_compute, nz=nz
    )
    ds["data0"].values[:] = 2.0
    ds.to_netcdf(os.path.join(temp_dir.name, savepoint_name + ".nc"))

    data = np.full(
        (nx_compute + 2 * n_halo + 1, nx_compute + 2 * n_halo + 1, nz), fill_value=2.0
    )

    checkpointer = ValidationCheckpointer(
        temp_dir.name,
        SavepointThresholds(
            {
                savepoint_name: [
                    {
                        "data0": Threshold(
                            relative=relative_threshold,
                            absolute=absolute_threshold,
                        )
                    },
                    {
                        "data0": Threshold(
                            relative=relative_threshold,
                            absolute=absolute_threshold,
                        )
                    },
                ]
            }
        ),
        rank=0,
    )
    with checkpointer.trial():
        checkpointer(savepoint_name, data0=data)
        checkpointer(savepoint_name, data0=data)


@pytest.mark.parametrize(
    "relative_threshold, absolute_threshold",
    [
        pytest.param(0.99, 1.0, id="relative_failure"),
        pytest.param(1.0, 0.99, id="absolute_failure"),
    ],
)
@requires_xarray
def test_validation_asserts_onevar_two_calls(relative_threshold, absolute_threshold):
    temp_dir = tempfile.TemporaryDirectory()
    nx_compute = 12
    nz = 20
    n_halo = 3
    savepoint_name = "savepoint_name"
    ds = get_dataset(
        n_savepoints=2, n_vars=1, n_ranks=1, nx=nx_compute, ny=nx_compute, nz=nz
    )
    ds["data0"].values[0, :] = 2.0
    ds["data0"].values[1, :] = 1.0
    ds.to_netcdf(os.path.join(temp_dir.name, savepoint_name + ".nc"))

    data = np.full(
        (nx_compute + 2 * n_halo + 1, nx_compute + 2 * n_halo + 1, nz), fill_value=2.0
    )

    checkpointer = ValidationCheckpointer(
        temp_dir.name,
        SavepointThresholds(
            {
                savepoint_name: [
                    {
                        "data0": Threshold(
                            relative=relative_threshold,
                            absolute=absolute_threshold,
                        )
                    },
                    {
                        "data0": Threshold(
                            relative=relative_threshold,
                            absolute=absolute_threshold,
                        )
                    },
                ]
            }
        ),
        rank=0,
    )
    with checkpointer.trial():
        checkpointer(savepoint_name, data0=data)
        with pytest.raises(AssertionError):
            checkpointer(savepoint_name, data0=data)


@pytest.mark.parametrize(
    "relative_threshold, absolute_threshold",
    [
        pytest.param(0.99, 1.0, id="relative_failure"),
        pytest.param(1.0, 0.99, id="absolute_failure"),
    ],
)
@requires_xarray
def test_validation_asserts_twovar_onecall(relative_threshold, absolute_threshold):
    temp_dir = tempfile.TemporaryDirectory()
    nx_compute = 12
    nz = 20
    n_halo = 3
    savepoint_name = "savepoint_name"
    ds = get_dataset(
        n_savepoints=1, n_vars=2, n_ranks=1, nx=nx_compute, ny=nx_compute, nz=nz
    )
    ds["data0"].values[:] = 1.0
    ds["data1"].values[:] = 1.0
    ds.to_netcdf(os.path.join(temp_dir.name, savepoint_name + ".nc"))

    data = np.full(
        (nx_compute + 2 * n_halo + 1, nx_compute + 2 * n_halo + 1, nz), fill_value=2.0
    )

    checkpointer = ValidationCheckpointer(
        temp_dir.name,
        SavepointThresholds(
            {
                savepoint_name: [
                    {
                        "data0": Threshold(
                            relative=1.0,
                            absolute=1.0,
                        ),
                        "data1": Threshold(
                            relative=relative_threshold,
                            absolute=absolute_threshold,
                        ),
                    }
                ]
            }
        ),
        rank=0,
    )
    with checkpointer.trial():
        with pytest.raises(AssertionError):
            checkpointer(savepoint_name, data0=data, data1=data)


@pytest.mark.parametrize(
    "array, target_shape, target_array",
    [
        pytest.param(
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
            (3,),
            np.asarray([2.0, 3.0, 4.0]),
            id="interface_dim",
        ),
        pytest.param(
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
            (2,),
            np.asarray([2.0, 3.0]),
            id="centered_dim",
        ),
        pytest.param(
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
            (5,),
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
            id="all_points",
        ),
    ],
)
def test_clip_pace_array_to_target(array, target_shape, target_array):
    clipped = _clip_pace_array_to_target(array, target_shape=target_shape)
    np.testing.assert_array_equal(clipped, target_array)
