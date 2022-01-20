import contextlib
import dataclasses
import unittest.mock
from datetime import datetime, timedelta
from typing import Literal, Tuple

import pytest

import pace.dsl
from fv3core.utils.null_comm import NullComm
from pace.driver.run import Driver, DriverConfig


def get_driver_config(
    nx_tile: int = 12,
    nz: int = 79,
    dt_atmos: int = 450,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    layout: Tuple[int, int] = (1, 1),
    start_time_type: Literal["timedelta", "datetime"] = "timedelta",
) -> DriverConfig:
    initialization_config = unittest.mock.MagicMock()
    if start_time_type == "timedelta":
        initialization_config.start_time = timedelta(0)
    else:
        initialization_config.start_time = datetime(2000, 1, 1)
    return DriverConfig(
        stencil_config=pace.dsl.StencilConfig(),
        nx_tile=nx_tile,
        nz=nz,
        dt_atmos=dt_atmos,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        layout=layout,
        initialization_type="baroclinic",
        initialization_config=initialization_config,
        diagnostics_config=unittest.mock.MagicMock(),
        dycore_config=unittest.mock.MagicMock(),
        physics_config=unittest.mock.MagicMock(),
    )


@pytest.mark.parametrize(
    "days, hours, minutes, seconds, expected",
    [
        pytest.param(1, 0, 0, 0, timedelta(days=1), id="day"),
        pytest.param(0, 1, 0, 0, timedelta(hours=1), id="hour"),
        pytest.param(0, 0, 1, 0, timedelta(minutes=1), id="minute"),
        pytest.param(0, 0, 0, 1, timedelta(seconds=1), id="second"),
        pytest.param(
            1, 2, 3, 4, timedelta(days=1, hours=2, minutes=3, seconds=4), id="all"
        ),
    ],
)
def test_total_time(days, hours, minutes, seconds, expected):
    config = get_driver_config(days=days, hours=hours, minutes=minutes, seconds=seconds)
    assert config.total_time == expected


@pytest.mark.parametrize(
    "timestep, minutes",
    [
        pytest.param(timedelta(minutes=5), 5, id="one_step"),
        pytest.param(timedelta(minutes=5), 10, id="two_steps"),
        pytest.param(timedelta(minutes=10), 50, id="many_longer_steps"),
        pytest.param(timedelta(minutes=5), 1, id="no_step"),
    ],
)
def test_driver(timestep: timedelta, minutes: int):
    config = get_driver_config(
        dt_atmos=int(timestep.total_seconds()),
        minutes=minutes,
    )
    n_timesteps = int((minutes * 60) / timestep.total_seconds())
    comm = NullComm(
        rank=0,
        total_ranks=6 * config.layout[0] * config.layout[1],
        fill_value=0.0,
    )
    with mocked_components() as mock:
        driver = Driver(
            config=config,
            comm=comm,
        )
        driver.step_all()
    assert driver.dycore.step_dynamics.call_count == n_timesteps
    assert driver.physics.call_count == n_timesteps
    # we store an extra step at the start of the run
    assert driver.diagnostics.store.call_count == n_timesteps + 1
    assert driver.dycore_to_physics.call_count == n_timesteps
    assert driver.physics_to_dycore.call_count == n_timesteps


@dataclasses.dataclass
class MockedComponents:
    dycore: unittest.mock.MagicMock
    step_dynamics: unittest.mock.MagicMock
    physics: unittest.mock.MagicMock
    diagnostics: unittest.mock.MagicMock
    dycore_to_physics: unittest.mock.MagicMock
    physics_to_dycore: unittest.mock.MagicMock


@contextlib.contextmanager
def mocked_components():
    with unittest.mock.patch("fv3core.DynamicalCore", spec=True) as dycore_mock:
        with unittest.mock.patch("fv3gfs.physics.Physics") as physics_mock:
            with unittest.mock.patch(
                "pace.stencils.UpdateAtmosphereState"
            ) as physics_to_dycore_mock:
                with unittest.mock.patch(
                    "pace.stencils.DycoreToPhysics"
                ) as dycore_to_physics_mock:
                    with unittest.mock.patch(
                        "pace.driver.run.Diagnostics"
                    ) as diagnostics_mock:
                        with unittest.mock.patch(
                            "fv3core.DynamicalCore.step_dynamics"
                        ) as step_dynamics_mock:
                            yield MockedComponents(
                                dycore=dycore_mock,
                                step_dynamics=step_dynamics_mock,
                                physics=physics_mock,
                                diagnostics=diagnostics_mock,
                                dycore_to_physics=dycore_to_physics_mock,
                                physics_to_dycore=physics_to_dycore_mock,
                            )
