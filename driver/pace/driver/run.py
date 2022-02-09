import abc
import dataclasses
import functools
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

# TODO: move update_atmos_state into pace.driver
from pace.stencils import update_atmos_state
from pace.stencils.testing import TranslateGrid, TranslateUpdateDWindsPhys
from pace.util.grid import DampingCoefficients
from pace.util.namelist import Namelist


@dataclasses.dataclass
class DriverState:
    dycore_state: fv3core.DycoreState
    physics_state: fv3gfs.physics.PhysicsState
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
        return DriverState(
            dycore_state=dycore_state,
            physics_state=physics_state,
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

    def _get_serialized_grid_damping_coeff_and_driver_grid(
        self,
        communicator: pace.util.CubedSphereCommunicator,
        backend: str,
    ) -> pace.stencils.testing.grid.Grid:
        ser = self._serializer(communicator)
        grid_savepoint = ser.get_savepoint("Grid-Info")[0]
        grid_data = {}
        grid_fields = ser.fields_at_savepoint(grid_savepoint)
        for field in grid_fields:
            grid_data[field] = ser.read(field, grid_savepoint)
            if len(grid_data[field].flatten()) == 1:
                grid_data[field] = grid_data[field][0]
        savepoint_in = ser.get_savepoint("FVDynamics-In")[0]
        for field in ["ak", "bk", "ptop"]:
            grid_data[field] = ser.read(field, savepoint_in)
            if len(grid_data[field].flatten()) == 1:
                grid_data[field] = grid_data[field][0]
        grid = TranslateGrid(
            grid_data, communicator.rank, self._namelist.layout, backend=backend
        ).python_grid()
        grid.grid_data.ak = grid_data["ak"]
        grid.grid_data.bk = grid_data["bk"]
        grid.grid_data.ptop = grid_data["ptop"]
        damping_coefficients = DampingCoefficients(
            divg_u=grid_data["divg_u"],
            divg_v=grid_data["divg_v"],
            del6_u=grid_data["del6_u"],
            del6_v=grid_data["del6_v"],
            da_min=grid_data["da_min"],
            da_min_c=grid_data["da_min_c"],
        )
        stencil_config = pace.dsl.stencil.StencilConfig(
            backend=backend,
        )
        stencil_factory = StencilFactory(
            config=stencil_config, grid_indexing=grid.grid_indexing
        )
        driver_grid_info_object = TranslateUpdateDWindsPhys(
            grid, self._namelist, stencil_factory
        )
        extra_grid_data = driver_grid_info_object.collect_input_data(
            ser, ser.get_savepoint("FVUpdatePhys-In")[0]
        )
        driver_grid_data = pace.util.grid.DriverGridData(
            vlon1=extra_grid_data["vlon1"],
            vlon2=extra_grid_data["vlon2"],
            vlon3=extra_grid_data["vlon3"],
            vlat1=extra_grid_data["vlat1"],
            vlat2=extra_grid_data["vlat2"],
            vlat3=extra_grid_data["vlat3"],
            edge_vect_w=extra_grid_data["edge_vect_w"],
            edge_vect_e=extra_grid_data["edge_vect_e"],
            edge_vect_s=extra_grid_data["edge_vect_s"],
            edge_vect_n=extra_grid_data["edge_vect_n"],
            es1_1=extra_grid_data["es1_1"],
            es1_2=extra_grid_data["es2_1"],
            es1_3=extra_grid_data["es3_1"],
            ew2_1=extra_grid_data["ew1_2"],
            ew2_2=extra_grid_data["ew2_2"],
            ew2_3=extra_grid_data["ew3_2"],
        )
        return grid, damping_coefficients, driver_grid_data

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
                damping_coeff,
                driver_grid_data,
            ) = self._get_serialized_grid_damping_coeff_and_driver_grid(
                communicator, backend
            )
            grid_data = grid.grid_data
        else:
            grid = fv3core._config.make_grid_with_data_from_namelist(
                self._namelist, communicator, backend
            )
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
        return DriverState(
            dycore_state=dycore_state,
            physics_state=physics_state,
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
        savepoint_in = ser.get_savepoint("FVDynamics-In")[0]
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


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    path: str
    names: List[str] = dataclasses.field(default_factory=list)


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
        zarr_state = {"time": time}
        for name in self.config.names:
            try:
                quantity = getattr(state.dycore_state, name)
            except AttributeError:
                quantity = getattr(state.physics_state, name)
            zarr_state[name] = quantity
        assert time is not None
        self.monitor.store(zarr_state)


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    """
    Configuration for a run of the Pace model.

    Attributes:
        stencil_config: configuration for stencil compilation
        initialization_type: must be "baroclinic" or "restart"
        initialization_config: configuration for the chosen initialization
            type, see documentation for its corresponding configuration
            dataclass
        nx_tile: number of gridpoints along the horizontal dimension of a cube
            tile face, same value used for both horizontal dimensions
        nz: number of gridpoints in the vertical dimension
        layout: number of ranks along the x and y dimensions
        dt_atmos: atmospheric timestep in seconds
    """

    stencil_config: pace.dsl.StencilConfig
    initialization_type: str
    initialization_config: InitializationConfig
    nx_tile: int
    nz: int
    layout: Tuple[int, int]
    dt_atmos: float
    diagnostics_config: DiagnosticsConfig
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
            initialization_class: Type[InitializationConfig] = SerialboxConfig
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
        communicator = pace.util.CubedSphereCommunicator.from_layout(
            comm=comm, layout=self.config.layout
        )
        quantity_factory, stencil_factory = _setup_factories(
            config=config, communicator=communicator
        )

        self.state = self.config.initialization_config.get_driver_state(
            quantity_factory=quantity_factory, communicator=communicator
        )
        self._start_time = self.config.initialization_config.start_time
        self.dycore = fv3core.DynamicalCore(
            comm=communicator,
            grid_data=self.state.grid_data,
            stencil_factory=stencil_factory,
            damping_coefficients=self.state.damping_coefficients,
            config=self.config.dycore_config,
            phis=self.state.dycore_state.phis,
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
            comm=comm,
        )

    def step_all(self):
        time = self.config.start_time
        end_time = self.config.start_time + self.config.total_time
        self.diagnostics.store(time=time, state=self.state)
        while time < end_time:
            self._step(timestep=self.config.timestep.total_seconds())
            time += self.config.timestep
            self.diagnostics.store(time=time, state=self.state)

    def _step(self, timestep: float):
        self._step_dynamics(timestep=timestep)
        self._step_physics(timestep=timestep)

    def _step_dynamics(self, timestep: float):
        self.dycore.step_dynamics(
            state=self.state.dycore_state,
            conserve_total_energy=self.config.dycore_config.consv_te,
            n_split=self.config.dycore_config.n_split,
            do_adiabatic_init=False,
            timestep=float(timestep),
        )

    def _step_physics(self, timestep: float):
        self.dycore_to_physics(
            dycore_state=self.state.dycore_state, physics_state=self.state.physics_state
        )
        self.physics(self.state.physics_state, timestep=float(timestep))
        self.physics_to_dycore(
            dycore_state=self.state.dycore_state,
            phy_state=self.state.physics_state,
            dt=float(timestep),
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


if __name__ == "__main__":
    command_line()
