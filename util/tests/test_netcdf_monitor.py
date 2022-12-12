import logging
from datetime import timedelta
from typing import List

import cftime
import numpy as np
import pytest

import pace.util
from pace.util._optional_imports import xarray as xr
from pace.util.testing import DummyComm


requires_xarray = pytest.mark.skipif(xr is None, reason="xarray is not installed")

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (4, 4)])
@pytest.mark.parametrize(
    "nt, time_chunk_size",
    [pytest.param(1, 1, id="single_time"), pytest.param(5, 2, id="chunked_time")],
)
@pytest.mark.parametrize(
    "shape, ny_rank_add, nx_rank_add, dims",
    [
        pytest.param((5, 4, 4), 0, 0, ("z", "y", "x"), id="cell_center"),
        pytest.param(
            (5, 4, 4), 1, 1, ("z", "y_interface", "x_interface"), id="cell_corner"
        ),
        pytest.param((5, 4, 4), 0, 1, ("z", "y", "x_interface"), id="cell_edge"),
    ],
)
@requires_xarray
def test_monitor_store_multi_rank_state(
    layout, nt, time_chunk_size, tmpdir, shape, ny_rank_add, nx_rank_add, dims, numpy
):
    units = "m"
    nz, ny, nx = shape
    ny_rank = int(ny / layout[0] + ny_rank_add)
    nx_rank = int(nx / layout[1] + nx_rank_add)
    tile = pace.util.TilePartitioner(layout)
    time = cftime.DatetimeJulian(2010, 6, 20, 6, 0, 0)
    timestep = timedelta(hours=1)
    total_ranks = 6 * layout[0] * layout[1]
    partitioner = pace.util.CubedSpherePartitioner(tile)
    shared_buffer = {}
    monitor_list: List[pace.util.NetCDFMonitor] = []

    for rank in range(total_ranks):
        communicator = pace.util.CubedSphereCommunicator(
            partitioner=partitioner,
            comm=DummyComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
            ),
        )
        # must eagerly initialize the tile object so that their ranks are
        # created in ascending order
        communicator.tile
        monitor_list.append(
            pace.util.NetCDFMonitor(
                path=tmpdir,
                communicator=communicator,
                time_chunk_size=time_chunk_size,
            )
        )

    for rank in range(total_ranks - 1, -1, -1):
        state = {
            "var_const1": pace.util.Quantity(
                numpy.ones([nz, ny_rank, nx_rank]),
                dims=dims,
                units=units,
            ),
        }
        monitor_list[rank].store_constant(state)

    tile_gathered = []
    for i_t in range(nt):
        for rank in range(total_ranks - 1, -1, -1):
            state = {
                "time": time + i_t * timestep,
                "var1": pace.util.Quantity(
                    numpy.ones([nz, ny_rank, nx_rank]),
                    dims=dims,
                    units=units,
                ),
            }
            monitor_list[rank].store(state)
            tile_gathered.append(
                monitor_list[rank]._communicator.tile.gather_state(state)
            )

    for rank in range(total_ranks - 1, -1, -1):
        state = {
            "var_const2": pace.util.Quantity(
                numpy.ones([nz, ny_rank, nx_rank]),
                dims=dims,
                units=units,
            ),
        }
        monitor_list[rank].store_constant(state)

    for monitor in monitor_list:
        monitor.cleanup()

    ds = xr.open_mfdataset(str(tmpdir / "state_*_tile*.nc"), decode_times=True)
    assert "var1" in ds
    np.testing.assert_array_equal(
        ds["var1"].shape, (nt, 6, nz, ny + ny_rank_add, nx + nx_rank_add)
    )
    assert ds["var1"].dims == ("time", "tile") + dims
    assert ds["var1"].attrs["units"] == units
    assert ds["time"].shape == (nt,)
    assert ds["time"].dims == ("time",)
    assert ds["time"].values[0] == time
    np.testing.assert_array_equal(ds["var1"].values, 1.0)

    ds_const = xr.open_dataset(str(tmpdir / "constants_var_const1.nc"))
    assert "var_const1" in ds_const
    np.testing.assert_array_equal(
        ds_const["var_const1"].shape, (6, nz, ny + ny_rank_add, nx + nx_rank_add)
    )
    assert ds_const["var_const1"].dims == ("tile",) + dims
    assert ds_const["var_const1"].attrs["units"] == units
    np.testing.assert_array_equal(ds_const["var_const1"].values, 1.0)
    ds_const2 = xr.open_dataset(str(tmpdir / "constants_var_const2.nc"))
    assert "var_const2" in ds_const2
    np.testing.assert_array_equal(
        ds_const2["var_const2"].shape, (6, nz, ny + ny_rank_add, nx + nx_rank_add)
    )
    assert ds_const2["var_const2"].dims == ("tile",) + dims
    assert ds_const2["var_const2"].attrs["units"] == units
    np.testing.assert_array_equal(ds_const2["var_const2"].values, 1.0)
