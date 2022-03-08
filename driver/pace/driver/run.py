import abc
import dataclasses
import functools
import os
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Type, Union

import click
import dacite
import f90nml
import yaml
import zarr.storage
from mpi4py import MPI

import fv3core
import fv3core.initialization.baroclinic as baroclinic_init
import fv3gfs.physics
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from fv3core.testing import TranslateFVDynamics
from pace.dsl.stencil import StencilFactory
from fv3core.stencils.fv_subgridz import DryConvectiveAdjustment
# TODO: move update_atmos_state into pace.driver
from pace.stencils import update_atmos_state
from pace.stencils.testing import TranslateGrid, TranslateUpdateDWindsPhys
from pace.util.grid import DampingCoefficients
from pace.util.namelist import Namelist
from pace.stencils.testing.grid import Grid
import pace.driver
from pace.util.quantity import QuantityMetadata

from .report import collect_data_and_write_to_file


@dataclasses.dataclass
class DriverState:
    dycore_state: fv3core.DycoreState
    physics_state: fv3gfs.physics.PhysicsState
    tendency_state: pace.driver.TendencyState
    grid_data: pace.util.grid.GridData
    damping_coefficients: pace.util.grid.DampingCoefficients
    driver_grid_data: pace.util.grid.DriverGridData


class InitializationConfig(abc.ABC):
    @property
    @abc.abstractmethod
    def start_time(self) -> datetime:
        ...

    @abc.abstractmethod
    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        ...


@dataclasses.dataclass
class BaroclinicConfig(InitializationConfig):
    """
    Configuration for baroclinic initialization.
    """

    @property
    def start_time(self) -> datetime:
        # TODO: instead of arbitrary start time, enable use of timedeltas
        return datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=communicator
        )
        grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
        damping_coeffient = DampingCoefficients.new_from_metric_terms(metric_terms)
        driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(
            metric_terms
        )
        dycore_state = baroclinic_init.init_baroclinic_state(
            metric_terms,
            adiabatic=False,
            hydrostatic=False,
            moist_phys=True,
            comm=communicator,
        )
        physics_state = fv3gfs.physics.PhysicsState.init_zeros(
            quantity_factory=quantity_factory, active_packages=["microphysics"]
        )
        tendency_state = pace.driver.TendencyState.init_zeros(
            quantity_factory=quantity_factory,
        )
        return DriverState(
            dycore_state=dycore_state,
            physics_state=physics_state,
            tendency_state=tendency_state,
            grid_data=grid_data,
            damping_coefficients=damping_coeffient,
            driver_grid_data=driver_grid_data,
        )


@dataclasses.dataclass
class RestartConfig(InitializationConfig):
    """
    Configuration for restart initialization.
    """

    path: str

    @property
    def start_time(self) -> datetime:
        return datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=communicator
        )
        state = pace.util.open_restart(
            dirname=self.path,
            communicator=communicator,
            quantity_factory=quantity_factory,
        )
        raise NotImplementedError()


@dataclasses.dataclass
class SerialboxConfig(InitializationConfig):
    """
    Configuration for Serialbox initialization.
    """

    path: str
    serialized_grid: bool

    @property
    def start_time(self) -> datetime:
        return datetime(2000, 1, 1)

    @property
    def _f90_namelist(self) -> f90nml.Namelist:
        return f90nml.read(self.path + "/input.nml")

    @property
    def _namelist(self) -> Namelist:
        return Namelist.from_f90nml(self._f90_namelist)

    def _get_serialized_grid(
        self,
        communicator: pace.util.CubedSphereCommunicator,
        backend: str,
    ) -> pace.stencils.testing.grid.Grid:
        ser = self._serializer(communicator)
        grid = TranslateGrid(
            grid_data, communicator.rank, self._namelist.layout, backend=backend
        ).python_grid()
        grid = TranslateGrid.new_from_serialized_data(serializer, communicator.rank, self._namelist.layout, backend).python_grid()
        return grid

    def _serializer(self, communicator: pace.util.CubedSphereCommunicator):
        import serialbox

        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read,
            self.path,
            "Generator_rank" + str(communicator.rank),
        )
        return serializer

    def _get_grid_data_damping_coeff_and_driver_grid(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        backend: str,
    ):
        if self.serialized_grid:
            (
                grid,
            ) = self._get_serialized_grid(
                communicator, backend
            )
            grid_data = grid.grid_data
            driver_grid_data = grid.driver_grid_data
            damping_coeff = grid.damping_coefficients
        else:
            grid = Grid.with_data_from_namelist(self._namelist, communicator, backend)
            metric_terms = pace.util.grid.MetricTerms(
                quantity_factory=quantity_factory, communicator=communicator
            )
            grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
            damping_coeff = DampingCoefficients.new_from_metric_terms(metric_terms)
            driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(
                metric_terms
            )
        return grid, grid_data, damping_coeff, driver_grid_data

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        backend = quantity_factory.empty(
            dims=[pace.util.X_DIM, pace.util.Y_DIM], units="unknown"
        ).gt4py_backend
        (
            grid,
            grid_data,
            damping_coeff,
            driver_grid_data,
        ) = self._get_grid_data_damping_coeff_and_driver_grid(
            quantity_factory, communicator, backend
        )
        dycore_state = self._initialize_dycore_state(
            quantity_factory, communicator, backend
        )
        physics_state = fv3gfs.physics.PhysicsState.init_zeros(
            quantity_factory=quantity_factory,
            active_packages=["microphysics"],
        )
        tendency_state = pace.driver.TendencyState.init_zeros(
            quantity_factory=quantity_factory,
        )
        return DriverState(
            dycore_state=dycore_state,
            physics_state=physics_state,
            tendency_state=tendency_state,
            grid_data=grid_data,
            damping_coefficients=damping_coeff,
            driver_grid_data=driver_grid_data,
        )

    def _initialize_dycore_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        backend: str,
    ) -> fv3core.DycoreState:
        (
            grid,
            grid_data,
            damping_coeff,
            driver_grid_data,
        ) = self._get_grid_data_damping_coeff_and_driver_grid(
            quantity_factory, communicator, backend
        )
        ser = self._serializer(communicator)
        savepoint_in = ser.get_savepoint("Driver-In")[0]
        stencil_config = pace.dsl.stencil.StencilConfig(
            backend=backend,
        )
        stencil_factory = StencilFactory(
            config=stencil_config, grid_indexing=grid.grid_indexing
        )
        translate_object = TranslateFVDynamics([grid], self._namelist, stencil_factory)
        input_data = translate_object.collect_input_data(ser, savepoint_in)
        dycore_state = translate_object.state_from_inputs(input_data)
        return dycore_state

@dataclasses.dataclass    
class TranslateConfig(InitializationConfig):
    """
    Configuration for pre-existing dycore state
    """
    dycore_state: Any
    grid: Grid
    
    @property
    def start_time(self) -> datetime:
        # TODO: instead of arbitrary start time, enable use of timedeltas
        return datetime(2016, 8, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:       
        physics_state = fv3gfs.physics.PhysicsState.init_zeros(
            quantity_factory=quantity_factory, active_packages=["microphysics"]
        )
        tendency_state = pace.driver.TendencyState.init_zeros(
            quantity_factory=quantity_factory,
        )
        return DriverState(
            dycore_state=self.dycore_state,
            physics_state=physics_state,
            tendency_state=tendency_state,
            grid_data=self.grid.grid_data,
            damping_coefficients=self.grid.damping_coefficients,
            driver_grid_data=self.grid.driver_grid_data,
        )


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    path: str
    names: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PerformanceConfig:
    performance_mode: bool = False
    experiment_name: str = "test"
    timestep_timer: pace.util.Timer = pace.util.NullTimer()
    total_timer: pace.util.Timer = pace.util.NullTimer()
    times_per_step: List = dataclasses.field(default_factory=list)
    hits_per_step: List = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.performance_mode:
            self.timestep_timer = pace.util.Timer()
            self.total_timer = pace.util.Timer()

    def collect_performance(self):
        """
        Take the accumulated timings and flush them into a new entry
        in times_per_step and hits_per_step.
        """
        if self.performance_mode:
            self.times_per_step.append(self.timestep_timer.times)
            self.hits_per_step.append(self.timestep_timer.hits)
            self.timestep_timer.reset()

    def write_out_performance(
        self,
        comm,
        backend: str,
        dt_atmos: float,
    ):
        if self.performance_mode:
            try:
                driver_path = os.path.dirname(__file__)
                git_hash = (
                    subprocess.check_output(
                        ["git", "-C", driver_path, "rev-parse", "HEAD"]
                    )
                    .decode()
                    .rstrip()
                )
            except subprocess.CalledProcessError:
                git_hash = "notarepo"

            self.times_per_step.append(self.total_timer.times)
            self.hits_per_step.append(self.total_timer.hits)
            comm.Barrier()
            while {} in self.hits_per_step:
                self.hits_per_step.remove({})
            collect_data_and_write_to_file(
                len(self.hits_per_step) - 1,
                backend,
                git_hash,
                comm,
                self.hits_per_step,
                self.times_per_step,
                self.experiment_name,
                dt_atmos,
            )


class Diagnostics:
    def __init__(
        self,
        config: DiagnosticsConfig,
        partitioner: pace.util.CubedSpherePartitioner,
        comm,
    ):
        self.config = config
        store = zarr.storage.DirectoryStore(path=self.config.path)
        self.monitor = pace.util.ZarrMonitor(
            store=store, partitioner=partitioner, mpi_comm=comm
        )

    def store(self, time: datetime, state: DriverState):
        if len(self.config.names) == 0:
            return
        zarr_state = {"time": time}
        for name in self.config.names:
            try:
                quantity = getattr(state.dycore_state, name)
            except AttributeError:
                quantity = getattr(state.physics_state, name)
            zarr_state[name] = quantity
        assert time is not None
        self.monitor.store(zarr_state)

    def store_grid(
        self, grid_data: pace.util.grid.GridData, metadata: QuantityMetadata
    ):
        zarr_grid = {}
        for name in ["lat", "lon"]:
            grid_quantity = pace.util.Quantity(
                getattr(grid_data, name),
                dims=("x_interface", "y_interface"),
                origin=metadata.origin,
                extent=(metadata.extent[0] + 1, metadata.extent[1] + 1),
                units="rad",
            )
            zarr_grid[name] = grid_quantity
        self.monitor.store_constant(zarr_grid)
        

@dataclasses.dataclass(frozen=True)
class DriverConfig:
    """
    Configuration for a run of the Pace model.

    Attributes:
        stencil_config: configuration for stencil compilation
        initialization_type: must be "baroclinic", "restart", "serialbox", or "regression"
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
    """

    stencil_config: pace.dsl.StencilConfig
    initialization_type: str
    initialization_config: InitializationConfig
    nx_tile: int
    nz: int
    layout: Tuple[int, int]
    dt_atmos: float
    diagnostics_config: DiagnosticsConfig
    performance_config: PerformanceConfig
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
    
    @functools.cached_property
    def timestep(self) -> timedelta:
        return timedelta(seconds=self.dt_atmos)

    @property
    def start_time(self) -> Union[datetime, timedelta]:
        return self.initialization_config.start_time

    @functools.cached_property
    def total_time(self) -> timedelta:
        return timedelta(
            days=self.days, hours=self.hours, minutes=self.minutes, seconds=self.seconds
        )

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "DriverConfig":
        initialization_type = kwargs["initialization_type"]
        if initialization_type == "serialbox":
            initialization_class = SerialboxConfig
        if initialization_type == "regression":
            initialization_class = TranslateConfig
        elif initialization_type == "baroclinic":
            initialization_class = BaroclinicConfig
        elif initialization_type == "restart":
            initialization_class = RestartConfig
        else:
            raise ValueError(
                "initialization_type must be one of 'baroclinic' or 'restart', "
                f"got {initialization_type}"
            )

        kwargs["initialization_config"] = dacite.from_dict(
            data_class=initialization_class,
            data=kwargs.get("initialization_config", {}),
            config=dacite.Config(strict=True),
        )
        if not isinstance(kwargs["dycore_config"], fv3core.DynamicalCoreConfig):
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
                                 
        if not isinstance(kwargs["physics_config"], fv3gfs.physics.PhysicsConfig):
            kwargs["physics_config"] = dacite.from_dict(
                data_class=fv3gfs.physics.PhysicsConfig,
                data=kwargs.get("physics_config", {}),
                config=dacite.Config(strict=True),
            )
                                 
        if initialization_type not in ["serialbox", "regression"]:
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

        elif initialization_type in ["serialbox", "regression"]:
            kwargs["nx_tile"] = kwargs["dycore_config"].npx - 1
            kwargs["nz"] = kwargs["dycore_config"].npz
            kwargs["layout"] = tuple(kwargs["dycore_config"].layout)

        return dacite.from_dict(
            data_class=cls, data=kwargs, config=dacite.Config(strict=True)
        )
    
                                 
class Driver:
    def __init__(
        self,
        config: DriverConfig,
        comm,
    ):
        """
        Initializes a pace Driver.

        Args:
            config: driver configuration
            comm: communication object behaving like mpi4py.Comm
        """
       
        self.config = config
        self.comm = comm
        self.performance_config = self.config.performance_config
        with self.performance_config.total_timer.clock("initialization"):                         
            communicator = pace.util.CubedSphereCommunicator.from_layout(
                comm=self.comm, layout=self.config.layout
            )
            quantity_factory, stencil_factory = _setup_factories(
                config=config, communicator=communicator
            )
       
            self.state = self.config.initialization_config.get_driver_state(
                quantity_factory=quantity_factory, communicator=communicator
            )
            self._start_time = self.config.initialization_config.start_time
            self.dycore = fv3core.stencils.fv_dynamics.DynamicalCore(
                comm=communicator,
                grid_data=self.state.grid_data,
                stencil_factory=stencil_factory,
                damping_coefficients=self.state.damping_coefficients,
                config=self.config.dycore_config,
                phis=self.state.dycore_state.phis,
            )
            if self.config.dycore_config.fv_sg_adj > 0:
                self.fv_subgridz = DryConvectiveAdjustment(
                    stencil_factory=stencil_factory,
                    nwat=self.config.dycore_config.nwat,
                    fv_sg_adj=self.config.dycore_config.fv_sg_adj,
                    n_sponge=self.config.dycore_config.n_sponge,
                    hydrostatic=self.config.dycore_config.hydrostatic,
                )
            self.physics = fv3gfs.physics.Physics(
                stencil_factory=stencil_factory,
                grid_data=self.state.grid_data,
                namelist=self.config.physics_config,
                active_packages=["microphysics"],
            )
            self.dycore_to_physics = update_atmos_state.DycoreToPhysics(
                stencil_factory=stencil_factory
            )
            self.physics_to_dycore = update_atmos_state.UpdateAtmosphereState(
                stencil_factory=stencil_factory,
                grid_data=self.state.grid_data,
                namelist=self.config.physics_config,
                comm=communicator,
                grid_info=self.state.driver_grid_data,
                quantity_factory=quantity_factory,
            )
            self.diagnostics = Diagnostics(
                config=config.diagnostics_config,
                partitioner=communicator.partitioner,
                comm=self.comm,
            )

   


    def step_all(self):
        with self.performance_config.total_timer.clock("total"):
            time = self.config.start_time
            end_time = self.config.start_time + self.config.total_time
            self.diagnostics.store_grid(
                grid_data=self.state.grid_data,
                metadata=self.state.dycore_state.ps.metadata,
            )
            while time < end_time:
                self._step(timestep=self.config.timestep.total_seconds())
                time += self.config.timestep
                self.diagnostics.store(time=time, state=self.state)
            self.performance_config.collect_performance()

    def _step(self, timestep: float):
        with self.performance_config.timestep_timer.clock("mainloop"):
            self._step_dynamics(timestep=timestep)
            if not self.config.dycore_only:
                self._step_physics(timestep=timestep)
            self.physics_to_dycore(
                dycore_state=self.state.dycore_state,
                phy_state=self.state.physics_state,
                tendency_state=self.state.tendency_state,
                dt=float(timestep),
                dycore_only=self.config.dycore_only
            )
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
        self.state.tendency_state.u_dt.data[:] = 0.0
        self.state.tendency_state.v_dt.data[:] = 0.0
        self.state.tendency_state.pt_dt.data[:] = 0.0
        if self.config.dycore_config.fv_sg_adj > 0:
            self.fv_subgridz(state=self.state.dycore_state, tendency_state=self.state.tendency_state, timestep=float(timestep))
    
    def _step_physics(self, timestep: float):
        self.dycore_to_physics(
            dycore_state=self.state.dycore_state, physics_state=self.state.physics_state
        )
        self.physics(self.state.physics_state, timestep=float(timestep))
      

    def write_performance_json_output(self):
        self.performance_config.write_out_performance(
            self.comm,
            self.config.stencil_config.backend,
            self.config.dt_atmos,
        )


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


@click.command()
@click.argument(
    "config_path",
    required=True,
)
def command_line(config_path: str):
    with open(config_path, "r") as f:
        driver_config = DriverConfig.from_dict(yaml.safe_load(f))
    main(driver_config=driver_config, comm=MPI.COMM_WORLD)


def main(driver_config: DriverConfig, comm):
    driver = Driver(
        config=driver_config,
        comm=comm,
    )
    driver.step_all()
    driver.write_performance_json_output()


if __name__ == "__main__":
    command_line()
