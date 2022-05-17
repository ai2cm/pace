import shutil
import subprocess

import gt4py
import numpy as np
import xarray as xr
import yaml
import zarr
from mpi4py import MPI

import pace.dsl
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
        shutil.rmtree("RESTART")
