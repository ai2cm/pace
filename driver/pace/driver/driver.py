import dataclasses
import functools
import logging
import warnings
from datetime import datetime, timedelta
from math import floor
from typing import Any, Dict, List, Tuple, Union

import dace
import dacite

import pace.driver
import pace.dsl
import pace.physics
import pace.stencils
import pace.util
import pace.util.grid
from pace import fv3core
from pace.dsl.dace.dace_config import DaceConfig
from pace.dsl.dace.orchestration import dace_inhibitor, orchestrate
from pace.dsl.stencil_config import CompilationConfig, RunMode
from pace.fv3core.initialization.dycore_state import DycoreState

# TODO: move update_atmos_state into pace.driver
from pace.stencils import update_atmos_state
from pace.util.communicator import CubedSphereCommunicator

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
        pair_debug: if True, runs two copies of the model each using
            half of the available MPI ranks, and compares results across these two
            copies during model execution, raising an exception if results ever
            differ. This is considerably more expensive since all model data must
            be communicated at the start and end of every stencil call.
        intermediate_restart: list of time steps to save intermediate restart files
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
    physics_config: pace.physics.PhysicsConfig = dataclasses.field(
        default_factory=pace.physics.PhysicsConfig
    )
    days: int = 0
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    dycore_only: bool = False
    disable_step_physics: bool = False
    save_restart: bool = False
    pair_debug: bool = False
    intermediate_restart: List[int] = dataclasses.field(default_factory=list)

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

    def n_timesteps(self) -> int:
        """Computing how many timestep required to carry the simulation."""
        if self.total_time < self.timestep:
            warnings.warn(
                f"No simulation possible: you asked for {self.total_time} "
                f"simulation time but the timestep is {self.timestep}"
            )
        return floor(self.total_time.total_seconds() / self.timestep.total_seconds())

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
                data_class=pace.physics.PhysicsConfig,
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

        if (
            isinstance(kwargs["stencil_config"], dict)
            and "dace_config" in kwargs["stencil_config"].keys()
        ):
            kwargs["stencil_config"]["dace_config"] = DaceConfig.from_dict(
                data=kwargs["stencil_config"]["dace_config"]
            )
        if (
            isinstance(kwargs["stencil_config"], dict)
            and "compilation_config" in kwargs["stencil_config"].keys()
        ):
            kwargs["stencil_config"][
                "compilation_config"
            ] = CompilationConfig.from_dict(
                data=kwargs["stencil_config"]["compilation_config"]
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
        self.config: DriverConfig = config
        self.time = self.config.start_time
        self.comm_config = config.comm_config
        global_comm = config.comm_config.get_comm()
        if config.pair_debug:
            world = int(global_comm.Get_rank() >= (global_comm.Get_size() // 2))
            self.comm = global_comm.Split(color=world, key=global_comm.Get_rank())
            stencil_compare_comm = global_comm
        else:
            self.comm = global_comm
            stencil_compare_comm = None
        self.performance_config = self.config.performance_config
        with self.performance_config.total_timer.clock("initialization"):
            communicator = CubedSphereCommunicator.from_layout(
                comm=self.comm, layout=self.config.layout
            )
            self.update_driver_config_with_communicator(communicator)

            if self.config.stencil_config.compilation_config.run_mode == RunMode.Build:

                def exit_function(*args, **kwargs):
                    print(
                        "Running in build-only mode and compilation finished, exiting"
                    )
                    exit(0)

                setattr(self, "step_all", exit_function)
            elif self.config.stencil_config.compilation_config.run_mode == RunMode.Run:

                def exit_instead_of_build(self):
                    stencil_class = (
                        None if self.options.rebuild else self.backend.load()
                    )
                    if stencil_class is None:
                        raise RuntimeError(
                            "Stencil needs to be compiled first in run mode, exiting"
                        )
                    return stencil_class

                from gt4py.stencil_builder import StencilBuilder

                setattr(StencilBuilder, "build", exit_instead_of_build)

            self.config.stencil_config.dace_config = DaceConfig(
                communicator=communicator,
                backend=self.config.stencil_config.backend,
                tile_nx=self.config.nx_tile,
                tile_nz=self.config.nz,
            )
            orchestrate(
                obj=self,
                config=self.config.stencil_config.dace_config,
                method_to_orchestrate="_critical_path_step_all",
                dace_constant_args=["timer"],
            )
            orchestrate(
                obj=self,
                config=self.config.stencil_config.dace_config,
                method_to_orchestrate="_step_dynamics",
                dace_constant_args=["state", "timer"],
            )
            orchestrate(
                obj=self,
                config=self.config.stencil_config.dace_config,
                method_to_orchestrate="_step_physics",
            )

            self.quantity_factory, self.stencil_factory = _setup_factories(
                config=config,
                communicator=communicator,
                stencil_compare_comm=stencil_compare_comm,
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
                state=self.state.dycore_state,
            )

            self.dycore.update_state(
                conserve_total_energy=self.config.dycore_config.consv_te,
                do_adiabatic_init=False,
                timestep=self.config.timestep.total_seconds(),
                n_split=self.config.dycore_config.n_split,
                state=self.state.dycore_state,
            )
            if not config.dycore_only and not config.disable_step_physics:
                self.physics = pace.physics.Physics(
                    stencil_factory=self.stencil_factory,
                    grid_data=self.state.grid_data,
                    namelist=self.config.physics_config,
                    active_packages=["microphysics"],
                )
            else:
                # Make sure those are set to None to raise any issues
                self.physics = None
            if not config.disable_step_physics:
                self.dycore_to_physics = update_atmos_state.DycoreToPhysics(
                    stencil_factory=self.stencil_factory,
                    dycore_config=self.config.dycore_config,
                    do_dry_convective_adjust=config.do_dry_convective_adjustment,
                    dycore_only=self.config.dycore_only,
                )
                self.end_of_step_update = update_atmos_state.UpdateAtmosphereState(
                    stencil_factory=self.stencil_factory,
                    grid_data=self.state.grid_data,
                    namelist=self.config.physics_config,
                    comm=communicator,
                    grid_info=self.state.driver_grid_data,
                    state=self.state.dycore_state,
                    quantity_factory=self.quantity_factory,
                    dycore_only=self.config.dycore_only,
                    apply_tendencies=self.config.apply_tendencies,
                    tendency_state=self.state.tendency_state,
                )
            else:
                # Make sure those are set to None to raise any issues
                self.dycore_to_physics = None
                self.end_of_step_update = None
            self.diagnostics = config.diagnostics_config.diagnostics_factory(
                partitioner=communicator.partitioner,
                comm=self.comm,
            )
            self.restart = pace.driver.Restart(
                save_restart=self.config.save_restart,
                intermediate_restart=self.config.intermediate_restart,
            )
        log_subtile_location(
            partitioner=communicator.partitioner.tile, rank=communicator.rank
        )
        self.diagnostics.store_grid(
            grid_data=self.state.grid_data,
            metadata=self.state.dycore_state.ps.metadata,
        )
        if config.diagnostics_config.output_initial_state:
            self.diagnostics.store(time=self.time, state=self.state)

        self._time_run = self.config.start_time

    def update_driver_config_with_communicator(
        self, communicator: CubedSphereCommunicator
    ) -> None:
        dace_config = DaceConfig(
            communicator=communicator,
            backend=self.config.stencil_config.compilation_config.backend,
            tile_nx=self.config.nx_tile,
            tile_nz=self.config.nz,
        )
        self.config.stencil_config.dace_config = dace_config
        original_config = self.config.stencil_config.compilation_config
        compilation_config = CompilationConfig(
            backend=original_config.backend,
            rebuild=original_config.rebuild,
            validate_args=original_config.validate_args,
            format_source=original_config.format_source,
            device_sync=original_config.device_sync,
            run_mode=original_config.run_mode,
            use_minimal_caching=original_config.use_minimal_caching,
            communicator=communicator,
        )
        self.config.stencil_config.compilation_config = compilation_config

    @dace_inhibitor
    def _callback_diagnostics(self):
        self._time_run += self.config.timestep
        self.diagnostics.store(time=self._time_run, state=self.state)

    @dace_inhibitor
    def _callback_restart(self, restart_path: str):
        self.restart.save_state_as_restart(
            state=self.state,
            comm=self.comm,
            restart_path=restart_path,
        )
        self.restart.write_restart_config(
            comm=self.comm,
            time=self.time,
            driver_config=self.config,
            restart_path=restart_path,
        )

    @dace_inhibitor
    def end_of_step_actions(self, step: int):
        """Gather operations unrelated to computation.
        Using a function allows those actions to be removed from the orchestration path.
        """
        if __debug__:
            logger.info(f"Finished stepping {step}")
        self.time += self.config.timestep
        if not ((step + 1) % self.config.diagnostics_config.output_frequency):
            self.diagnostics.store(time=self.time, state=self.state)
        if (
            self.restart.save_intermediate_restart
            and step in self.config.intermediate_restart
        ):
            self._write_restart_files(restart_path=f"RESTART_{step}")
        self.performance_config.collect_performance()

    def _critical_path_step_all(
        self,
        steps_count: int,
        timer: pace.util.Timer,
        dt: float,
    ):
        """Start of code path where performance is critical.

        This function must remain orchestrateable by DaCe (e.g.
        all code not parsable due to python complexity needs to be moved
        to a callback, like end_of_step_actions)."""
        for step in dace.nounroll(range(steps_count)):
            with timer.clock("mainloop"):
                self._step_dynamics(
                    self.state.dycore_state,
                    self.performance_config.timestep_timer,
                )
                if not self.config.disable_step_physics:
                    self._step_physics(timestep=dt)
            self.end_of_step_actions(step)

    def step_all(self):
        logger.info("integrating driver forward in time")
        with self.performance_config.total_timer.clock("total"):
            self._critical_path_step_all(
                steps_count=self.config.n_timesteps(),
                timer=self.performance_config.timestep_timer,
                dt=self.config.timestep.total_seconds(),
            )

    def _step_dynamics(
        self,
        state: DycoreState,
        timer: pace.util.Timer,
    ):
        self.dycore.step_dynamics(
            state=state,
            timer=timer,
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
            u_dt=self.state.tendency_state.u_dt.storage,
            v_dt=self.state.tendency_state.v_dt.storage,
            pt_dt=self.state.tendency_state.pt_dt.storage,
            dt=float(timestep),
        )

    def _write_performance_json_output(self):
        self.performance_config.write_out_performance(
            self.comm,
            self.config.stencil_config.compilation_config.backend,
            self.config.stencil_config.dace_config.is_dace_orchestrated(),
            self.config.dt_atmos,
        )

    @dace_inhibitor
    def _write_restart_files(self, restart_path="RESTART"):
        self.restart.save_state_as_restart(
            state=self.state,
            comm=self.comm,
            restart_path=restart_path,
        )
        self.restart.write_restart_config(
            comm=self.comm,
            time=self.time,
            driver_config=self.config,
            restart_path=restart_path,
        )

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
    config: DriverConfig,
    communicator: pace.util.CubedSphereCommunicator,
    stencil_compare_comm,
) -> Tuple["pace.util.QuantityFactory", "pace.dsl.StencilFactory"]:
    """
    Args:
        config: configuration of driver
        communicator: performs communication needed for grid initialization
        stencil_compare_comm: if given, all stencils created by the returned factory
            will compare data to its "pair" rank before and after every stencil call,
            used for debugging in "pair_debug" mode.

    Returns:
        quantity_factory: creates Quantities
        stencil_factory: creates Stencils
    """

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
        sizer, backend=config.stencil_config.compilation_config.backend
    )
    stencil_factory = pace.dsl.StencilFactory(
        config=config.stencil_config,
        grid_indexing=grid_indexing,
        comm=stencil_compare_comm,
    )
    return quantity_factory, stencil_factory
