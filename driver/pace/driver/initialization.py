import abc
import dataclasses
from datetime import datetime
from typing import ClassVar

import fv3core
import fv3core.initialization.baroclinic as baroclinic_init
import fv3gfs.physics
import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from pace.util.grid import DampingCoefficients

from .registry import Registry
from .state import DriverState, TendencyState, _restart_driver_state


class Initializer(abc.ABC):
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
class InitializerSelector(Initializer):
    """
    Dataclass for selecting the implementation of Initializer to use.

    Used to circumvent the issue that dacite expects static class definitions,
    but we would like to dynamically define which Initializer to use. Does this
    by representing the part of the yaml specification that asks which initializer
    to use, but deferring to the implementation in that initializer when called.
    """

    type: str
    config: Initializer
    registry: ClassVar[Registry] = Registry()

    @classmethod
    def register(cls, type_name):
        return cls.registry.register(type_name)

    @property
    def start_time(self) -> datetime:
        return self.config.start_time

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        return self.config.get_driver_state(
            quantity_factory=quantity_factory, communicator=communicator
        )

    @classmethod
    def from_dict(cls, config: dict):
        instance = cls.registry.from_dict(config)
        return cls(config=instance, type=config["type"])


@InitializerSelector.register("baroclinic")
@dataclasses.dataclass
class BaroclinicConfig(Initializer):
    """
    Configuration for baroclinic initialization.
    """

    start_time: datetime = datetime(2000, 1, 1)

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


@InitializerSelector.register("restart")
@dataclasses.dataclass
class RestartConfig(Initializer):
    """
    Configuration for restart initialization.
    """

    path: str = "."
    start_time: datetime = datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        state = _restart_driver_state(
            self.path, communicator.rank, quantity_factory, communicator
        )
        return state


@InitializerSelector.register("predefined")
@dataclasses.dataclass
class PredefinedStateConfig(Initializer):
    """
    Configuration if the states are already defined.

    Generally you will not want to use this class when initializing from yaml,
    as it requires numpy array data to be part of the configuration dictionary
    used to construct the class.
    """

    dycore_state: fv3core.DycoreState
    physics_state: fv3gfs.physics.PhysicsState
    tendency_state: TendencyState
    grid_data: pace.util.grid.GridData
    damping_coefficients: pace.util.grid.DampingCoefficients
    driver_grid_data: pace.util.grid.DriverGridData
    start_time: datetime = datetime(2016, 8, 1)

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
