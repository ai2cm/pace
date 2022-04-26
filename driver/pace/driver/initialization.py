import abc
import dataclasses
from datetime import datetime

import fv3core
import fv3core.initialization.baroclinic as baroclinic_init
import fv3gfs.physics
import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from pace.util.grid import DampingCoefficients

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
