import subprocess
import unittest.mock

import gt4py
import numpy as np
import xarray as xr
import yaml
import zarr
from mpi4py import MPI

import pace.dsl
from pace.driver import CreatesComm, DriverConfig
from pace.driver.initialization import BaroclinicConfig
from pace.driver.state import DriverState
from pace.util.null_comm import NullComm


# The packages we import will import MPI, causing an MPI init, but we don't actually
# want to use MPI under this script. We have to finalize so mpirun will work on
# the test scripts we call that *do* need MPI.
MPI.Finalize()


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


def test_restart():
    try:
        subprocess.check_output("tests/mpi/run_save_and_load_restart.sh")
        restart = xr.open_zarr(
            store=zarr.DirectoryStore(path="output.zarr"), consolidated=False
        )
        regular = xr.open_zarr(
            store=zarr.DirectoryStore(path="run_two_steps_output.zarr"),
            consolidated=False,
        )
        assert restart["time"][0] == regular["time"][-1]
        for var in [
            "u",
            "v",
            "ua",
            "va",
            "pt",
            "delp",
            "qvapor",
            "qliquid",
            "qice",
            "qrain",
            "qsnow",
            "qgraupel",
        ]:
            np.testing.assert_allclose(
                restart[var].isel(time=0).values, regular[var].isel(time=-1).values
            )

        # now use the same restart to test DriverState load function
        with open("RESTART/restart.yaml", "r") as f:
            config = yaml.safe_load(f)
            config["comm_config"]["type"] = "null_comm"
            config["comm_config"]["config"]["rank"] = 0
            config["comm_config"]["config"]["total_ranks"] = 6
            driver_config = DriverConfig.from_dict(config)
        driver_state = DriverState.load_state_from_restart("RESTART", driver_config)
        assert isinstance(driver_state, DriverState)

        restart_dycore = xr.open_dataset(
            f"RESTART/restart_dycore_state_{driver_config.comm_config.config.rank}.nc"
        )
        for var in driver_state.dycore_state.__dict__.keys():
            if isinstance(driver_state.dycore_state.__dict__[var], pace.util.Quantity):
                np.testing.assert_allclose(
                    driver_state.dycore_state.__dict__[var].data,
                    restart_dycore[var].values,
                )

        restart_physics = xr.open_dataset(
            f"RESTART/restart_physics_state_{driver_config.comm_config.config.rank}.nc"
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
    finally:
        subprocess.Popen(["make", "clean"])


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
        subprocess.Popen(["make", "clean"])
