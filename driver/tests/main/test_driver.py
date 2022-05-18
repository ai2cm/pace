import contextlib
import dataclasses
import math
import unittest.mock
from datetime import datetime, timedelta
from typing import Literal, Tuple

import pytest

import pace.dsl
from pace.driver import CreatesComm, Driver, DriverConfig
from pace.driver.report import (
    TimeReport,
    gather_hit_counts,
    gather_timing_data,
    get_sypd,
)
from pace.util.null_comm import NullComm


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
        initialization=initialization_config,
        performance_config=unittest.mock.MagicMock(),
        comm_config=NullCommConfig(layout),
        diagnostics_config=unittest.mock.MagicMock(),
        dycore_config=unittest.mock.MagicMock(fv_sg_adj=1),
        physics_config=unittest.mock.MagicMock(),
    )


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
        pytest.param(timedelta(minutes=5), 11, id="three_steps_past_duration"),
    ],
)
def test_driver(timestep: timedelta, minutes: int):
    config = get_driver_config(
        dt_atmos=int(timestep.total_seconds()),
        minutes=minutes,
    )
    n_timesteps = math.ceil((minutes * 60) / timestep.total_seconds())
    with mocked_components() as mock:
        driver = Driver(
            config=config,
        )
        driver.step_all()
    assert driver.dycore.step_dynamics.call_count == n_timesteps
    assert driver.physics.call_count == n_timesteps
    # we store an extra step at the start of the run
    assert driver.diagnostics.store.call_count == n_timesteps
    assert driver.dycore_to_physics.call_count == n_timesteps
    assert driver.end_of_step_update.call_count == n_timesteps


test_data = [
    list(
        [
            {
                "DynCore": 0.8,
                "TracerAdvection": 0.15,
                "Remapping": 0.05,
                "mainloop": 1.0,
            }
        ]
    ),
    list(
        [
            {"DynCore": 1, "TracerAdvection": 1, "Remapping": 1, "mainloop": 1},
            {"DynCore": 1, "TracerAdvection": 1, "Remapping": 1, "mainloop": 1},
            {"DynCore": 1, "TracerAdvection": 1, "Remapping": 1, "mainloop": 1},
        ]
    ),
    365.0,
    3,
    1.0,
]


@pytest.mark.parametrize(
    "times_per_step, hits_per_step, dt_atmos, expected_hits, expected_SYPD",
    [test_data],
)
def test_timing_info(
    times_per_step, hits_per_step, dt_atmos, expected_hits, expected_SYPD
):
    comm = NullComm(
        rank=0,
        total_ranks=6,
        fill_value=0.0,
    )
    timing_info = gather_timing_data(times_per_step, comm)
    timing_info = gather_hit_counts(hits_per_step, timing_info)
    nrank = len(timing_info["mainloop"].times)
    timing_info["mainloop"].times = [
        [times_per_step[0]["mainloop"]] for i in range(nrank)
    ]
    sypd = get_sypd(timing_info, dt_atmos)
    assert timing_info["mainloop"].hits == expected_hits
    assert sypd == expected_SYPD


timing_info = {
    "mainloop": TimeReport(
        hits=3,
        times=[
            [1.0, 0.8, 1.2],
            [1.2, 0.9, 0.9],
            [1.0, 1.0, 1.0],
        ],
    )
}

test_data = [timing_info, 365.0, 1.0]


@pytest.mark.parametrize(
    "timing_info, dt_atmos, expected_SYPD",
    [test_data],
)
def test_sypd(timing_info, dt_atmos, expected_SYPD):
    sypd = get_sypd(timing_info, dt_atmos)
    assert sypd == expected_SYPD


@dataclasses.dataclass
class MockedComponents:
    dycore: unittest.mock.MagicMock
    step_dynamics: unittest.mock.MagicMock
    physics: unittest.mock.MagicMock
    diagnostics: unittest.mock.MagicMock
    dycore_to_physics: unittest.mock.MagicMock
    end_of_step_update: unittest.mock.MagicMock


@contextlib.contextmanager
def mocked_components():
    with unittest.mock.patch("fv3core.DynamicalCore", spec=True) as dycore_mock:
        with unittest.mock.patch("fv3gfs.physics.Physics") as physics_mock:
            with unittest.mock.patch(
                "pace.stencils.update_atmos_state.UpdateAtmosphereState"
            ) as end_of_step_update_mock:
                with unittest.mock.patch(
                    "pace.stencils.update_atmos_state.DycoreToPhysics"
                ) as dycore_to_physics_mock:
                    with unittest.mock.patch(
                        "pace.driver.diagnostics.Diagnostics"
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
                                end_of_step_update=end_of_step_update_mock,
                            )
