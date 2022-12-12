import shutil
import subprocess

import numpy as np
import xarray as xr
import yaml
import zarr
from mpi4py import MPI

import pace.dsl
import pace.util
from pace.driver import DriverConfig
from pace.driver.state import DriverState


# The packages we import will import MPI, causing an MPI init, but we don't actually
# want to use MPI under this script. We have to finalize so mpirun will work on
# the test scripts we call that *do* need MPI.
MPI.Finalize()


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

        with open("RESTART/restart.yaml", "r") as f:
            restart_config = DriverConfig.from_dict(yaml.safe_load(f))

        mpi_comm = pace.util.NullComm(rank=0, total_ranks=6, fill_value=0.0)
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner((1, 1))
        )
        communicator = pace.util.CubedSphereCommunicator(mpi_comm, partitioner)
        (damping_coefficients, driver_grid_data, grid_data,) = restart_config.get_grid(
            communicator=communicator,
        )

        driver_state = restart_config.get_driver_state(
            communicator=communicator,
            damping_coefficients=damping_coefficients,
            driver_grid_data=driver_grid_data,
            grid_data=grid_data,
        )

        assert isinstance(driver_state, DriverState)

        restart_dycore = xr.open_dataset(
            f"RESTART/restart_dycore_state_{communicator.rank}.nc"
        )
        for var in driver_state.dycore_state.__dict__.keys():
            if isinstance(driver_state.dycore_state.__dict__[var], pace.util.Quantity):
                np.testing.assert_allclose(
                    driver_state.dycore_state.__dict__[var].data,
                    restart_dycore[var].values,
                )

    finally:
        shutil.rmtree("RESTART")
