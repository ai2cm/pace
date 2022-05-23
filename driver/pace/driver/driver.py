import dataclasses
import functools
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple, Union

import dacite
import yaml

import fv3core
import fv3gfs.physics
import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid

# TODO: move update_atmos_state into pace.driver
from pace.stencils import update_atmos_state

from . import diagnostics
from .comm import CreatesCommSelector
from .initialization import InitializerSelector
from .performance import PerformanceConfig


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    """
    Configuration for a run of the Pace model.

    Attributes:
        stencil_config: configuration for stencil compilation
        initialization_type: must be
             "baroclinic", "restart", or "predefined"
        initialization_config: configuration for the chosen initialization
            type, see documentation for its corresponding configuration
            dataclass
        nx_tile: number of gridpoints along the horizontal dimension of a cube
            tile face, same value used for both horizontal dimensions
        nz: number of gridpoints in the vertical dimension
        layout: number of ranks along the x and y dimensions
        dt_atmos: atmospheric timestep in seconds
        diagnostics_config: configuration for output diagnostics
        dycore_config: configuration for dynamical core
        physics_config: configuration for physics
        days: days to add to total simulation time
        hours: hours to add to total simulation time
        minutes: minutes to add to total simulation time
        seconds: seconds to add to total simulation time
        dycore_only: whether to run just the dycore, or physics too
        disable_step_physics: whether to completely disable the step_physics call,
            including coupling code between the dycore and physics, as well as
            dry static adjustment. This is a development flag and will be removed
            in a later commit.
        save_restart: whether to save the state output as restart files at
            cleanup time
    """

    stencil_config: pace.dsl.StencilConfig
    initialization: InitializerSelector
    nx_tile: int
    nz: int
    layout: Tuple[int, int]
    dt_atmos: float
    diagnostics_config: diagnostics.DiagnosticsConfig = dataclasses.field(
        default_factory=diagnostics.DiagnosticsConfig
    )
    performance_config: PerformanceConfig = dataclasses.field(
        default_factory=PerformanceConfig
    )
    comm_config: CreatesCommSelector = dataclasses.field(
        default_factory=CreatesCommSelector
    )
    dycore_config: fv3core.DynamicalCoreConfig = dataclasses.field(
        default_factory=fv3core.DynamicalCoreConfig
    )
    physics_config: fv3gfs.physics.PhysicsConfig = dataclasses.field(
        default_factory=fv3gfs.physics.PhysicsConfig
    )
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    dycore_only: bool = False
    disable_step_physics: bool = False
    save_restart: bool = False

    @functools.cached_property
    def timestep(self) -> timedelta:
        return timedelta(seconds=self.dt_atmos)

    @property
    def start_time(self) -> Union[datetime, timedelta]:
        return self.initialization.start_time

    @functools.cached_property
    def total_time(self) -> timedelta:
        return timedelta(
            days=self.days, hours=self.hours, minutes=self.minutes, seconds=self.seconds
        )

    @functools.cached_property
    def do_dry_convective_adjustment(self) -> bool:
        return self.dycore_config.do_dry_convective_adjustment

    @functools.cached_property
    def apply_tendencies(self) -> bool:
        return self.do_dry_convective_adjustment or not self.dycore_only

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "DriverConfig":
        if isinstance(kwargs["dycore_config"], dict):
            for derived_name in ("dt_atmos", "layout", "npx", "npy", "npz", "ntiles"):
                if derived_name in kwargs["dycore_config"]:
                    raise ValueError(
                        f"you cannot set {derived_name} directly in dycore_config, "
                        "as it is determined based on top-level configuration"
                    )

            kwargs["dycore_config"] = dacite.from_dict(
                data_class=fv3core.DynamicalCoreConfig,
                data=kwargs.get("dycore_config", {}),
                config=dacite.Config(strict=True),
            )

        if isinstance(kwargs["physics_config"], dict):
            kwargs["physics_config"] = dacite.from_dict(
                data_class=fv3gfs.physics.PhysicsConfig,
                data=kwargs.get("physics_config", {}),
                config=dacite.Config(strict=True),
            )

        kwargs["layout"] = tuple(kwargs["layout"])
        kwargs["dycore_config"].layout = kwargs["layout"]
        kwargs["dycore_config"].dt_atmos = kwargs["dt_atmos"]
        kwargs["dycore_config"].npx = kwargs["nx_tile"] + 1
        kwargs["dycore_config"].npy = kwargs["nx_tile"] + 1
        kwargs["dycore_config"].npz = kwargs["nz"]
        kwargs["dycore_config"].ntiles = 6
        kwargs["physics_config"].layout = kwargs["layout"]
        kwargs["physics_config"].dt_atmos = kwargs["dt_atmos"]
        kwargs["physics_config"].npx = kwargs["nx_tile"] + 1
        kwargs["physics_config"].npy = kwargs["nx_tile"] + 1
        kwargs["physics_config"].npz = kwargs["nz"]
        kwargs["comm_config"] = CreatesCommSelector.from_dict(
            kwargs.get("comm_config", {})
        )
        kwargs["initialization"] = InitializerSelector.from_dict(
            kwargs["initialization"]
        )

        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )


class Driver:
    def __init__(
        self,
        config: DriverConfig,
    ):
        """
        Initializes a pace Driver.

        Args:
            config: driver configuration
            comm: communication object behaving like mpi4py.Comm
        """
        logger.info("initializing driver")
        self.config = config
        self.time = self.config.start_time
        self.comm_config = config.comm_config
        self.comm = config.comm_config.get_comm()
        self.performance_config = self.config.performance_config
        with self.performance_config.total_timer.clock("initialization"):
            communicator = pace.util.CubedSphereCommunicator.from_layout(
                comm=self.comm, layout=self.config.layout
            )
            self.quantity_factory, self.stencil_factory = _setup_factories(
                config=config, communicator=communicator
            )

            self.state = self.config.initialization.get_driver_state(
                quantity_factory=self.quantity_factory, communicator=communicator
            )
            self._start_time = self.config.initialization.start_time
            self.dycore = fv3core.DynamicalCore(
                comm=communicator,
                grid_data=self.state.grid_data,
                stencil_factory=self.stencil_factory,
                damping_coefficients=self.state.damping_coefficients,
                config=self.config.dycore_config,
                phis=self.state.dycore_state.phis,
            )

            self.physics = fv3gfs.physics.Physics(
                stencil_factory=self.stencil_factory,
                grid_data=self.state.grid_data,
                namelist=self.config.physics_config,
                active_packages=["microphysics"],
            )
            self.dycore_to_physics = update_atmos_state.DycoreToPhysics(
                stencil_factory=self.stencil_factory,
                dycore_config=self.config.dycore_config,
                do_dry_convective_adjustment=self.config.do_dry_convective_adjustment,
                dycore_only=self.config.dycore_only,
            )
            self.end_of_step_update = update_atmos_state.UpdateAtmosphereState(
                stencil_factory=self.stencil_factory,
                grid_data=self.state.grid_data,
                namelist=self.config.physics_config,
                comm=communicator,
                grid_info=self.state.driver_grid_data,
                quantity_factory=self.quantity_factory,
                dycore_only=self.config.dycore_only,
                apply_tendencies=self.config.apply_tendencies,
            )
            self.diagnostics = config.diagnostics_config.diagnostics_factory(
                partitioner=communicator.partitioner,
                comm=self.comm,
            )
        log_subtile_location(
            partitioner=communicator.partitioner.tile, rank=communicator.rank
        )
        self.diagnostics.store_grid(
            grid_data=self.state.grid_data,
            metadata=self.state.dycore_state.ps.metadata,
        )

    def step_all(self):
        logger.info("integrating driver forward in time")
        with self.performance_config.total_timer.clock("total"):
            end_time = self.config.start_time + self.config.total_time
            while self.time < end_time:
                self.step(timestep=self.config.timestep)

    def step(self, timestep: timedelta):
        with self.performance_config.timestep_timer.clock("mainloop"):
            self._step_dynamics(timestep=timestep.total_seconds())
            if not self.config.disable_step_physics:
                self._step_physics(timestep=timestep.total_seconds())
        self.time += timestep
        self.diagnostics.store(time=self.time, state=self.state)
        self.performance_config.collect_performance()

    def _step_dynamics(self, timestep: float):
        self.dycore.step_dynamics(
            state=self.state.dycore_state,
            conserve_total_energy=self.config.dycore_config.consv_te,
            n_split=self.config.dycore_config.n_split,
            do_adiabatic_init=False,
            timestep=float(timestep),
            timer=self.performance_config.timestep_timer,
        )

    def _step_physics(self, timestep: float):
        self.dycore_to_physics(
            dycore_state=self.state.dycore_state,
            physics_state=self.state.physics_state,
            tendency_state=self.state.tendency_state,
            timestep=float(timestep),
        )
        if not self.config.dycore_only:
            self.physics(self.state.physics_state, timestep=float(timestep))
        self.end_of_step_update(
            dycore_state=self.state.dycore_state,
            phy_state=self.state.physics_state,
            tendency_state=self.state.tendency_state,
            dt=float(timestep),
        )

    def _write_performance_json_output(self):
        self.performance_config.write_out_performance(
            self.comm,
            self.config.stencil_config.backend,
            self.config.dt_atmos,
        )

    def _write_restart_files(self):
        self.state.save_state(comm=self.comm)
        if self.comm.Get_rank() == 0:
            config_dict = dataclasses.asdict(self.config)
            config_dict["performance_config"].pop("times_per_step", None)
            config_dict["performance_config"].pop("hits_per_step", None)
            config_dict["performance_config"].pop("timestep_timer", None)
            config_dict["performance_config"].pop("total_timer", None)
            for field in ["dt_atmos", "layout", "npx", "npy", "npz", "ntiles"]:
                config_dict["dycore_config"].pop(field, None)
                config_dict["physics_config"].pop(field, None)
            config_dict["initialization"]["type"] = "restart"
            config_dict["initialization"]["config"]["start_time"] = self.time
            config_dict["initialization"]["config"]["path"] = f"{os.getcwd()}/RESTART"
            with open("RESTART/restart.yaml", "w") as file:
                yaml.safe_dump(config_dict, file)

    def cleanup(self):
        logger.info("cleaning up driver")
        if self.config.save_restart:
            self._write_restart_files()
        self._write_performance_json_output()
        self.comm_config.cleanup(self.comm)


def log_subtile_location(partitioner: pace.util.TilePartitioner, rank: int):
    location_info = {
        "north": partitioner.on_tile_top(rank),
        "south": partitioner.on_tile_bottom(rank),
        "east": partitioner.on_tile_right(rank),
        "west": partitioner.on_tile_left(rank),
    }
    logger.info(f"running on rank {rank} with subtile location {location_info}")


def _setup_factories(
    config: DriverConfig, communicator: pace.util.CubedSphereCommunicator
) -> Tuple["pace.util.QuantityFactory", "pace.dsl.StencilFactory"]:
    sizer = pace.util.SubtileGridSizer.from_tile_params(
        nx_tile=config.nx_tile,
        ny_tile=config.nx_tile,
        nz=config.nz,
        n_halo=pace.util.N_HALO_DEFAULT,
        extra_dim_lengths={},
        layout=config.layout,
        tile_partitioner=communicator.partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
        sizer=sizer, cube=communicator
    )
    quantity_factory = pace.util.QuantityFactory.from_backend(
        sizer, backend=config.stencil_config.backend
    )
    stencil_factory = pace.dsl.StencilFactory(
        config=config.stencil_config,
        grid_indexing=grid_indexing,
    )
    return quantity_factory, stencil_factory
