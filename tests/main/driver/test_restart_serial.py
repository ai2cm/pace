import os
import shutil
from datetime import datetime

import gt4py
import numpy as np
import xarray as xr
import yaml

import pace.dsl
from pace.driver import CreatesComm, DriverConfig
from pace.driver.driver import RestartConfig
from pace.driver.initialization import BaroclinicInit
from pace.util.null_comm import NullComm


DIR = os.path.dirname(os.path.abspath(__file__))


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
    restart_config = RestartConfig()
    assert restart_config.save_restart is False


def test_restart_save_to_disk():
    try:
        with open(
            os.path.join(
                DIR,
                "../../../driver/examples/configs/baroclinic_c12_write_restart.yaml",
            ),
            "r",
        ) as f:
            driver_config = DriverConfig.from_dict(yaml.safe_load(f))
        backend = "numpy"
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

        (
            damping_coefficients,
            driver_grid_data,
            grid_data,
        ) = pace.driver.GeneratedGridConfig().get_grid(quantity_factory, communicator)
        init = BaroclinicInit()
        driver_state = init.get_driver_state(
            quantity_factory=quantity_factory,
            communicator=communicator,
            damping_coefficients=damping_coefficients,
            driver_grid_data=driver_grid_data,
            grid_data=grid_data,
        )
        time = datetime(2016, 1, 1, 0, 0, 0)

        driver_config.restart_config.write_final_if_enabled(
            state=driver_state,
            comm=mpi_comm,
            time=time,
            driver_config=driver_config,
            restart_path="RESTART",
        )

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

        # TODO: the physics state isn't actually needed in the restart folders as
        # all prognostic state is in dycore state, we could refactor it out
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
        # test we can use the saved driver config in the restart to load it
        with open("RESTART/restart.yaml", "r") as f:
            restart_config = DriverConfig.from_dict(yaml.safe_load(f))

            (
                damping_coefficients,
                driver_grid_data,
                grid_data,
            ) = restart_config.get_grid(
                communicator=communicator,
            )

            restart_state = restart_config.get_driver_state(
                communicator=communicator,
                damping_coefficients=damping_coefficients,
                driver_grid_data=driver_grid_data,
                grid_data=grid_data,
            )
            for var in driver_state.dycore_state.__dict__.keys():
                before_restart = driver_state.dycore_state.__dict__[var]
                after_restart = restart_state.dycore_state.__dict__[var]
                if isinstance(before_restart, pace.util.Quantity):
                    np.testing.assert_allclose(
                        before_restart.view[:],
                        after_restart.view[:],
                    )
                else:
                    np.testing.assert_allclose(
                        before_restart,
                        after_restart,
                    )

    finally:
        shutil.rmtree("RESTART")
