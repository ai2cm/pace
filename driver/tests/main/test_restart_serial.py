import shutil
import unittest.mock

import gt4py
import numpy as np
import xarray as xr

import pace.dsl
from pace.driver import CreatesComm, DriverConfig
from pace.driver.initialization import BaroclinicConfig
from pace.util.null_comm import NullComm


class NullCommConfig(CreatesComm):
    def __init__(self, layout):
        self.layout = layout

    def get_comm(self):
        return NullComm(
            rank=0,
            total_ranks=6 * self.layout[0] * self.layout[1],
            fill_value=0.0,
        )

    def cleanup(self, comm):
        pass


def test_default_save_restart():
    driver_config = DriverConfig(
        stencil_config=pace.dsl.StencilConfig(),
        nx_tile=12,
        nz=79,
        dt_atmos=225,
        days=0,
        hours=0,
        minutes=0,
        seconds=225,
        layout=(1, 1),
        initialization=unittest.mock.MagicMock(),
        performance_config=unittest.mock.MagicMock(),
        comm_config=NullCommConfig((1, 1)),
        diagnostics_config=unittest.mock.MagicMock(),
        dycore_config=unittest.mock.MagicMock(nwat=6),
        physics_config=unittest.mock.MagicMock(),
    )
    assert driver_config.save_restart is False


def test_restart_save_to_disk():
    try:
        backend = "gtc:numpy"
        mpi_comm = NullComm(rank=0, total_ranks=6, fill_value=0.0)
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner((1, 1))
        )
        communicator = pace.util.CubedSphereCommunicator(mpi_comm, partitioner)
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=12,
            ny_tile=12,
            nz=79,
            n_halo=3,
            extra_dim_lengths={},
            layout=(1, 1),
            tile_partitioner=partitioner.tile,
            tile_rank=communicator.tile.rank,
        )
        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer=sizer, backend=backend
        )

        init = BaroclinicConfig()
        driver_state = init.get_driver_state(
            quantity_factory=quantity_factory, communicator=communicator
        )
        driver_state.save_state(mpi_comm)

        restart_dycore = xr.open_dataset(
            f"RESTART/restart_dycore_state_{mpi_comm.rank}.nc"
        )
        for var in driver_state.dycore_state.__dict__.keys():
            if isinstance(driver_state.dycore_state.__dict__[var], pace.util.Quantity):
                np.testing.assert_allclose(
                    driver_state.dycore_state.__dict__[var].data,
                    restart_dycore[var].values,
                )
            else:
                if var in restart_dycore.keys():
                    raise KeyError(
                        f"{var} is not a quantity and \
                        should not be in dycore restart file"
                    )

        restart_physics = xr.open_dataset(
            f"RESTART/restart_physics_state_{mpi_comm.rank}.nc"
        )
        for var in driver_state.physics_state.__dict__.keys():
            if isinstance(
                driver_state.physics_state.__dict__[var],
                gt4py.storage.storage.CPUStorage,
            ):
                np.testing.assert_allclose(
                    driver_state.physics_state.__dict__[var].data,
                    restart_physics[var].values,
                )
            else:
                if var in restart_physics.keys():
                    raise KeyError(
                        f"{var} is not a storage and \
                            should not be in physics restart file"
                    )
    finally:
        shutil.rmtree("RESTART")
