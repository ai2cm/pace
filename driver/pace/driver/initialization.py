import abc
import dataclasses
from datetime import datetime

import f90nml

import fv3core
import fv3core.initialization.baroclinic as baroclinic_init
import fv3gfs.physics
import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from fv3core.testing import TranslateFVDynamics
from pace.dsl.stencil import StencilFactory

# TODO: move update_atmos_state into pace.driver
from pace.stencils.testing import TranslateGrid
from pace.util.grid import DampingCoefficients
from pace.util.namelist import Namelist

from .state import DriverState, TendencyState


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
        tendency_state = TendencyState.init_zeros(
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
        grid = TranslateGrid.new_from_serialized_data(
            ser, communicator.rank, self._namelist.layout, backend
        ).python_grid()
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
            grid = self._get_serialized_grid(communicator, backend)
            grid_data = grid.grid_data
            driver_grid_data = grid.driver_grid_data
            damping_coeff = grid.damping_coefficients
        else:
            grid = pace.stencils.testing.grid.Grid.with_data_from_namelist(
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
        tendency_state = TendencyState.init_zeros(
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
class PredefinedStateConfig(InitializationConfig):
    """
    Configuration if the states are already defined
    """

    dycore_state: fv3core.DycoreState
    physics_state: fv3gfs.physics.PhysicsState
    tendency_state: TendencyState
    grid_data: pace.util.grid.GridData
    damping_coefficients: pace.util.grid.DampingCoefficients
    driver_grid_data: pace.util.grid.DriverGridData

    @property
    def start_time(self) -> datetime:
        # TODO: instead of arbitrary start time, enable use of timedeltas
        return datetime(2016, 8, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:

        return DriverState(
            dycore_state=self.dycore_state,
            physics_state=self.physics_state,
            tendency_state=self.tendency_state,
            grid_data=self.grid_data,
            damping_coefficients=self.damping_coefficients,
            driver_grid_data=self.driver_grid_data,
        )
