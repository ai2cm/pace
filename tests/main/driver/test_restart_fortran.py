import os

import numpy as np
import xarray as xr

import pace.driver
import pace.util
from pace.driver.initialization import FortranRestartInit
from pace.util import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)


DIR = os.path.dirname(os.path.abspath(__file__))
PACE_DIR = os.path.join(DIR, "../../../")


def test_state_from_fortran_restart():
    layout = (1, 1)
    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    # need a local communicator to mock "scatter" for the restart data,
    # but need null communicator to handle grid initialization
    local_comm = pace.util.LocalComm(rank=0, total_ranks=6, buffer_dict={})
    null_comm = pace.util.NullComm(rank=0, total_ranks=6)
    local_communicator = CubedSphereCommunicator(local_comm, partitioner)
    null_communicator = CubedSphereCommunicator(null_comm, partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=12,
        ny_tile=12,
        nz=63,
        n_halo=3,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=0,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend="numpy")
    restart_dir = os.path.join(PACE_DIR, "util/tests/data/c12_restart")

    (
        damping_coefficients,
        driver_grid_data,
        grid_data,
    ) = pace.driver.GeneratedGridConfig(restart_path=restart_dir).get_grid(
        quantity_factory, null_communicator
    )

    restart_config = FortranRestartInit(path=restart_dir)
    driver_state = restart_config.get_driver_state(
        quantity_factory,
        local_communicator,
        damping_coefficients=damping_coefficients,
        driver_grid_data=driver_grid_data,
        grid_data=grid_data,
    )
    ds = xr.open_dataset(os.path.join(restart_dir, "fv_core.res.tile1.nc"))
    np.testing.assert_array_equal(
        ds["u"].values[0, :].transpose(2, 1, 0), driver_state.dycore_state.u.view[:]
    )
