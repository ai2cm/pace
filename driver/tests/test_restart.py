import unittest.mock

import numpy as np
import xarray as xr
import yaml
import zarr

import pace.dsl
from pace.driver import DriverConfig
from pace.driver.state import DriverState

from .test_driver import NullCommConfig


def test_save_restart():
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
        save_restart=True,
    )
    assert driver_config.save_restart is True


def test_restart_results():
    restart = xr.open_zarr(
        store=zarr.DirectoryStore(path="output.zarr"), consolidated=False
    )
    regular = xr.open_zarr(
        store=zarr.DirectoryStore(path="run_two_steps_output.zarr"), consolidated=False
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


def test_driver_state_load_restart():
    with open("RESTART/restart.yaml", "r") as f:
        config = yaml.safe_load(f)
        config["comm_config"]["type"] = "null_comm"
        config["comm_config"]["config"]["rank"] = 0
        config["comm_config"]["config"]["total_ranks"] = 6
        driver_config = DriverConfig.from_dict(config)
    driver_state = DriverState.load_state_from_restart("RESTART", driver_config)
    assert isinstance(driver_state, DriverState)
