import abc
import dataclasses
import logging
import os
import pathlib
from datetime import datetime
from typing import Callable, ClassVar, Type, TypeVar

import f90nml

import pace.driver
import pace.dsl
import pace.fv3core.initialization.baroclinic as baroclinic_init
import pace.physics
import pace.stencils
import pace.util
import pace.util.grid
from pace import fv3core
from pace.dsl.dace.orchestration import DaceConfig
from pace.dsl.stencil import StencilFactory
from pace.dsl.stencil_config import CompilationConfig
from pace.fv3core.testing import TranslateFVDynamics
from pace.stencils.testing import TranslateGrid
from pace.util.namelist import Namelist

from .registry import Registry
from .state import DriverState, TendencyState, _restart_driver_state


logger = logging.getLogger(__name__)


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
        damping_coefficients: pace.util.grid.DampingCoefficients,
        driver_grid_data: pace.util.grid.DriverGridData,
        grid_data: pace.util.grid.GridData,
    ) -> DriverState:
        ...


IT = TypeVar("IT", bound=Type[Initializer])


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
    def register(cls, type_name) -> Callable[[IT], IT]:
        return cls.registry.register(type_name)

    @property
    def start_time(self) -> datetime:
        return self.config.start_time

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        damping_coefficients: pace.util.grid.DampingCoefficients,
        driver_grid_data: pace.util.grid.DriverGridData,
        grid_data: pace.util.grid.GridData,
    ) -> DriverState:
        return self.config.get_driver_state(
            quantity_factory=quantity_factory,
            communicator=communicator,
            damping_coefficients=damping_coefficients,
            driver_grid_data=driver_grid_data,
            grid_data=grid_data,
        )

    @classmethod
    def from_dict(cls, config: dict):
        instance = cls.registry.from_dict(config)
        return cls(config=instance, type=config["type"])


@InitializerSelector.register("baroclinic")
@dataclasses.dataclass
class BaroclinicInit(Initializer):
    """
    Configuration for baroclinic initialization.
    """

    start_time: datetime = datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        damping_coefficients: pace.util.grid.DampingCoefficients,
        driver_grid_data: pace.util.grid.DriverGridData,
        grid_data: pace.util.grid.GridData,
    ) -> DriverState:
        dycore_state = baroclinic_init.init_baroclinic_state(
            grid_data=grid_data,
            quantity_factory=quantity_factory,
            adiabatic=False,
            hydrostatic=False,
            moist_phys=True,
            comm=communicator,
        )
        physics_state = pace.physics.PhysicsState.init_zeros(
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
            damping_coefficients=damping_coefficients,
            driver_grid_data=driver_grid_data,
        )


@InitializerSelector.register("restart")
@dataclasses.dataclass
class RestartInit(Initializer):
    """
    Configuration for pace restart initialization.
    """

    path: str = "."
    start_time: datetime = datetime(2000, 1, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        damping_coefficients: pace.util.grid.DampingCoefficients,
        driver_grid_data: pace.util.grid.DriverGridData,
        grid_data: pace.util.grid.GridData,
    ) -> DriverState:
        state = _restart_driver_state(
            self.path,
            communicator.rank,
            quantity_factory,
            communicator,
            damping_coefficients,
            driver_grid_data,
            grid_data,
        )

        return state


@InitializerSelector.register("fortran_restart")
@dataclasses.dataclass
class FortranRestartInit(Initializer):
    """
    Configuration for fortran restart initialization.
    """

    path: str = "."

    @property
    def start_time(self) -> datetime:
        """Reads the last line in coupler.res to find the restart time"""
        restart_files = os.listdir(self.path)

        coupler_file = restart_files[
            [fname.endswith("coupler.res") for fname in restart_files].index(True)
        ]
        restart_doc = pathlib.Path(self.path) / coupler_file
        fl = open(restart_doc, "r")
        contents = fl.readlines()
        fl.close()
        last_line = contents.pop(-1)
        date = [
            dt if len(dt) == 4 else "%02d" % int(dt) for dt in last_line.split()[:6]
        ]
        date_dt = datetime.strptime("".join(date), "%Y%m%d%H%M%S")
        return date_dt

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        damping_coefficients: pace.util.grid.DampingCoefficients,
        driver_grid_data: pace.util.grid.DriverGridData,
        grid_data: pace.util.grid.GridData,
    ) -> DriverState:
        state = _restart_driver_state(
            self.path,
            communicator.rank,
            quantity_factory,
            communicator,
            damping_coefficients,
            driver_grid_data,
            grid_data,
        )

        _update_fortran_restart_pe_peln(state)

        # TODO
        # follow what fortran does with restart data after reading it
        # should eliminate small differences between restart input and
        # serialized test data

        return state


@InitializerSelector.register("serialbox")
@dataclasses.dataclass
class SerialboxInit(Initializer):
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
    ) -> pace.stencils.testing.grid.Grid:  # type: ignore
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

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        damping_coefficients: pace.util.grid.DampingCoefficients,
        driver_grid_data: pace.util.grid.DriverGridData,
        grid_data: pace.util.grid.GridData,
    ) -> DriverState:
        backend = quantity_factory.empty(
            dims=[pace.util.X_DIM, pace.util.Y_DIM], units="unknown"
        ).gt4py_backend

        dycore_state = self._initialize_dycore_state(communicator, backend)
        physics_state = pace.physics.PhysicsState.init_zeros(
            quantity_factory=quantity_factory,
            active_packages=["microphysics"],
        )
        tendency_state = TendencyState.init_zeros(quantity_factory=quantity_factory)

        return DriverState(
            dycore_state=dycore_state,
            physics_state=physics_state,
            tendency_state=tendency_state,
            grid_data=grid_data,
            damping_coefficients=damping_coefficients,
            driver_grid_data=driver_grid_data,
        )

    def _initialize_dycore_state(
        self,
        communicator: pace.util.CubedSphereCommunicator,
        backend: str,
    ) -> fv3core.DycoreState:

        grid = self._get_serialized_grid(communicator=communicator, backend=backend)

        ser = self._serializer(communicator)
        savepoint_in = ser.get_savepoint("Driver-In")[0]
        dace_config = DaceConfig(
            communicator,
            backend,
            tile_nx=self._namelist.npx,
            tile_nz=self._namelist.npz,
        )
        stencil_config = pace.dsl.stencil.StencilConfig(
            compilation_config=CompilationConfig(
                backend=backend, communicator=communicator
            ),
            dace_config=dace_config,
        )
        stencil_factory = StencilFactory(
            config=stencil_config, grid_indexing=grid.grid_indexing
        )
        translate_object = TranslateFVDynamics(grid, self._namelist, stencil_factory)
        input_data = translate_object.collect_input_data(ser, savepoint_in)
        dycore_state = translate_object.state_from_inputs(input_data)
        return dycore_state


@InitializerSelector.register("predefined")
@dataclasses.dataclass
class PredefinedStateInit(Initializer):
    """
    Configuration if the states are already defined.

    Generally you will not want to use this class when initializing from yaml,
    as it requires numpy array data to be part of the configuration dictionary
    used to construct the class.
    """

    dycore_state: fv3core.DycoreState
    physics_state: pace.physics.PhysicsState
    tendency_state: TendencyState
    grid_data: pace.util.grid.GridData
    damping_coefficients: pace.util.grid.DampingCoefficients
    driver_grid_data: pace.util.grid.DriverGridData
    start_time: datetime = datetime(2016, 8, 1)

    def get_driver_state(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
        damping_coefficients: pace.util.grid.DampingCoefficients,
        driver_grid_data: pace.util.grid.DriverGridData,
        grid_data: pace.util.grid.GridData,
    ) -> DriverState:

        return DriverState(
            dycore_state=self.dycore_state,
            physics_state=self.physics_state,
            tendency_state=self.tendency_state,
            grid_data=self.grid_data,
            damping_coefficients=self.damping_coefficients,
            driver_grid_data=self.driver_grid_data,
        )


# TODO: refactor fv3core so that pe and peln are internal temporaries
# of the dynamical core, computed automatically, so that this helper
# can be eliminated from initialization
def _update_fortran_restart_pe_peln(state: DriverState) -> None:
    """
    Fortran restart data don't have information on pressure interface values
    and their logs.
    This function takes the delp data (that is present in restart files), and
    top level pressure to calculate pressure at interfaces and their log,
    and updates the driver state with values.
    """

    ptop = state.grid_data.ak.view[0]
    pe = state.dycore_state.pe
    peln = state.dycore_state.peln
    delp = state.dycore_state.delp

    for level in range(pe.data.shape[2]):
        pe.data[:, :, level] = ptop + delp.np.sum(delp.data[:, :, :level], 2)

    peln.data[:] = pe.np.log(pe.data[:])

    state.dycore_state.pe = pe
    state.dycore_state.peln = peln
