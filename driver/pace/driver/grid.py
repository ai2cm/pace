import abc
import dataclasses
import logging
from datetime import datetime
from typing import ClassVar, Optional, Tuple

import f90nml
import os
import xarray as xr

import pace.driver
import pace.dsl
import pace.fv3core.initialization.baroclinic as baroclinic_init
import pace.physics
import pace.stencils
from pace.util import CubedSphereCommunicator, QuantityFactory
import pace.util.grid
from pace import fv3core
from pace.dsl.dace.orchestration import DaceConfig
from pace.dsl.stencil import StencilFactory
from pace.dsl.stencil_config import CompilationConfig
from pace.fv3core.testing import TranslateFVDynamics
from pace.stencils.testing import TranslateGrid
from pace.util.grid import DampingCoefficients, DriverGridData, GridData, MetricTerms, direct_transform
from pace.util.namelist import Namelist

from .registry import Registry
from .state import DriverState, TendencyState, _restart_driver_state


logger = logging.getLogger(__name__)


class GridInitializer(abc.ABC):

    @abc.abstractmethod
    def get_grid(
        self,
        quantity_factory: pace.util.QuantityFactory,
        communicator: pace.util.CubedSphereCommunicator,
    ) -> DriverState:
        ...


@dataclasses.dataclass
class GridInitializerSelector(GridInitializer):
    """
    Dataclass for selecting the implementation of Initializer to use.

    Used to circumvent the issue that dacite expects static class definitions,
    but we would like to dynamically define which Initializer to use. Does this
    by representing the part of the yaml specification that asks which initializer
    to use, but deferring to the implementation in that initializer when called.
    """

    type: str
    config: GridInitializer
    registry: ClassVar[Registry] = Registry()

    @classmethod
    def register(cls, type_name):
        return cls.registry.register(type_name)


    def get_grid(
        self,
        quantity_factory: QuantityFactory,
        communicator: CubedSphereCommunicator,
    ) -> DriverState:
        return self.config.get_grid(
            quantity_factory=quantity_factory, communicator=communicator
        )

    @classmethod
    def from_dict(cls, config: dict):
        instance = cls.registry.from_dict(config)
        return cls(config=instance, type=config["type"])


@GridInitializerSelector.register("generated")
@dataclasses.dataclass
class GeneratedConfig(GridInitializer):
    """
    Configuration for baroclinic initialization.
    
    Attributes:
        1
        2
        3


    """
    stretch_grid: bool = False
    stretch_factor: Optional[float] = None
    lon_target: Optional[float] = None
    lat_target: Optional[float] = None
    tc_ks: int = 0
    restart_path: Optional[str] = None # can this be just path from config? can I see that?
    # if restart_path, then read in vertical grid

    def __post_init__(self):
        if self.stretch_grid:
            if not self.stretch_factor:
                raise ValueError(
                    "Stretch_mode is true, but no stretch_factor is provided."
                )
            if not self.lon_target:
                raise ValueError("Stretch_grid is true, but no lon_target is provided.")
            if not self.lat_target:
                raise ValueError("Stretch_grid is true, but no lat_target is provided.")


    def get_grid(
        self,
        quantity_factory: QuantityFactory,
        communicator: CubedSphereCommunicator,
    ) -> Tuple[DampingCoefficients, DriverGridData, GridData]:

        metric_terms = MetricTerms(
            quantity_factory=quantity_factory, communicator=communicator
        )
        np = np # figure out how to determine this
        if self.stretch_grid: # do horizontal grid transformation
            metric_terms = _transform_horizontal_grid(self, metric_terms, np)
            grid_data = GridData.new_from_metric_terms(metric_terms)

            if self.restart_path: # read in vertical grid
                grid_data = _replace_vertical_grid(self, metric_terms, np)

        else:
            grid_data = GridData.new_from_metric_terms(metric_terms)

        damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
        driver_grid_data = DriverGridData.new_from_metric_terms(metric_terms)

        return damping_coefficients, driver_grid_data, grid_data
    


def _transform_horizontal_grid(self, metric_terms: MetricTerms, np) -> MetricTerms:
    """
    Uses the Schmidt transform to locally refine the horizontal grid.
    """
    # TODO figure out backend
    # if self.stencil_factory.config.is_gpu_backend:
    #     np = cp
    # else:
    # np = np

    grid = metric_terms.grid
    lon_transform, lat_transform = direct_transform(
        lon=grid.data[:, :, 0],
        lat=grid.data[:, :, 1],
        stretch_factor=self.stretch_factor,
        lon_target=self.lon_target,
        lat_target=self.lat_target,
        np=np,
    )
    grid.data[:, :, 0] = lon_transform[:]
    grid.data[:, :, 1] = lat_transform[:]

    metric_terms._grid.data[:] = grid.data[:]
    metric_terms._init_agrid()

    return metric_terms
        
        
def _replace_vertical_grid(self, metric_terms: MetricTerms, np) -> GridData:
    """
    Replaces the vertical grid generators from metric terms (ak and bk) with
    their fortran restart values (in fv_core.res.nc).
    Then re-generates grid data with the new vertical inputs.
    """

    ak_bk_data_file = self.restart_path + "/fv_core.res.nc"
    if not os.path.isfile(ak_bk_data_file):
        raise ValueError(
            """use_tc_vertical_grid is true,
            but no fv_core.res.nc in restart data file."""
        )

    p_ref = 101500 # somehow read from dycore config?


    file = self.path + '/fv_core.res.nc'
    ds = xr.open_dataset(file).isel(Time=0).drop_vars("Time")
    metric_terms._ak = ds["ak"].data
    metric_terms._bk = ds["bk"].data
    ds.close()

    delp = metric_terms.ak.data[1:] - metric_terms.ak.data[:-1] + p_ref * (metric_terms.bk.data[1:] - metric_terms.bk.data[:-1])
    pres = p_ref - np.cumsum(delp)
    ptop = pres[-1]
    vertical_data = pace.util.grid.VerticalGridData(ptop, self.ks, metric_terms.ak.data, metric_terms.bk.data, p_ref=p_ref)
    horizontal_data = pace.util.grid.HorizontalGridData.new_from_metric_terms(metric_terms)
    contravariant_data = pace.util.grid.ContravariantGridData.new_from_metric_terms(metric_terms)
    angle_data = pace.util.grid.AngleGridData.new_from_metric_terms(metric_terms)

    grid_data = pace.util.grid.GridData(
        horizontal_data=horizontal_data,
        vertical_data=vertical_data,
        contravariant_data=contravariant_data,
        angle_data=angle_data,
    )
    return grid_data






# @InitializerSelector.register("baroclinic")
# @dataclasses.dataclass
# class BaroclinicConfig(Initializer):
#     """
#     Configuration for baroclinic initialization.
#     """

#     start_time: datetime = datetime(2000, 1, 1)

#     def get_driver_state(
#         self,
#         quantity_factory: pace.util.QuantityFactory,
#         communicator: pace.util.CubedSphereCommunicator,
#     ) -> DriverState:
#         metric_terms = pace.util.grid.MetricTerms(
#             quantity_factory=quantity_factory, communicator=communicator
#         )
#         grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
#         damping_coeffient = DampingCoefficients.new_from_metric_terms(metric_terms)
#         driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(
#             metric_terms
#         )
#         dycore_state = baroclinic_init.init_baroclinic_state(
#             metric_terms,
#             adiabatic=False,
#             hydrostatic=False,
#             moist_phys=True,
#             comm=communicator,
#         )
#         physics_state = pace.physics.PhysicsState.init_zeros(
#             quantity_factory=quantity_factory, active_packages=["microphysics"]
#         )
#         tendency_state = TendencyState.init_zeros(
#             quantity_factory=quantity_factory,
#         )
#         return DriverState(
#             dycore_state=dycore_state,
#             physics_state=physics_state,
#             tendency_state=tendency_state,
#             grid_data=grid_data,
#             damping_coefficients=damping_coeffient,
#             driver_grid_data=driver_grid_data,
#         )


# @InitializerSelector.register("restart")
# @dataclasses.dataclass
# class RestartConfig(Initializer):
#     """
#     Configuration for restart initialization.
#     """

#     path: str = "."
#     start_time: datetime = datetime(2000, 1, 1)
#     fortran_data: bool = False

#     def get_driver_state(
#         self,
#         quantity_factory: pace.util.QuantityFactory,
#         communicator: pace.util.CubedSphereCommunicator,
#     ) -> DriverState:
#         state = _restart_driver_state(
#             self.path,
#             communicator.rank,
#             quantity_factory,
#             communicator,
#             self.fortran_data,
#         )

#         # TODO
#         # follow what fortran does with restart data after reading it
#         # should eliminate small differences between restart input and
#         # serialized test data
#         return state


# @InitializerSelector.register("fortran_restart")
# @dataclasses.dataclass
# class FortranRestartConfig(Initializer):
#     """
#     Configuration for fortran restart initialization.
#     """

#     path: str = "."
#     fortran_data: bool = False

#     @property
#     def start_time(self) -> datetime:
#         """make it get start time from restart file"""

#         return

#     def get_driver_state(
#         self,
#         quantity_factory: pace.util.QuantityFactory,
#         communicator: pace.util.CubedSphereCommunicator,
#     ) -> DriverState:
#         state = _restart_driver_state(
#             self.path,
#             communicator.rank,
#             quantity_factory,
#             communicator,
#             self.fortran_data,
#         )

#         # TODO
#         # follow what fortran does with restart data after reading it
#         # should eliminate small differences between restart input and
#         # serialized test data
#         return state




# @InitializerSelector.register("serialbox")
# @dataclasses.dataclass
# class SerialboxConfig(Initializer):
#     """
#     Configuration for Serialbox initialization.
#     """

#     path: str
#     serialized_grid: bool

#     @property
#     def start_time(self) -> datetime:
#         return datetime(2000, 1, 1)

#     @property
#     def _f90_namelist(self) -> f90nml.Namelist:
#         return f90nml.read(self.path + "/input.nml")

#     @property
#     def _namelist(self) -> Namelist:
#         return Namelist.from_f90nml(self._f90_namelist)

#     def _get_serialized_grid(
#         self,
#         communicator: pace.util.CubedSphereCommunicator,
#         backend: str,
#     ) -> pace.stencils.testing.grid.Grid:
#         ser = self._serializer(communicator)
#         grid = TranslateGrid.new_from_serialized_data(
#             ser, communicator.rank, self._namelist.layout, backend
#         ).python_grid()
#         return grid

#     def _serializer(self, communicator: pace.util.CubedSphereCommunicator):
#         import serialbox

#         serializer = serialbox.Serializer(
#             serialbox.OpenModeKind.Read,
#             self.path,
#             "Generator_rank" + str(communicator.rank),
#         )
#         return serializer

#     def _get_grid_data_damping_coeff_and_driver_grid(
#         self,
#         quantity_factory: pace.util.QuantityFactory,
#         communicator: pace.util.CubedSphereCommunicator,
#         backend: str,
#     ):
#         if self.serialized_grid:
#             logger.info("Using serialized grid data")
#             grid = self._get_serialized_grid(communicator, backend)
#             grid_data = grid.grid_data
#             driver_grid_data = grid.driver_grid_data
#             damping_coeff = grid.damping_coefficients
#         else:
#             logger.info("Using a grid generated from metric terms")
#             grid = pace.stencils.testing.grid.Grid.with_data_from_namelist(
#                 self._namelist, communicator, backend
#             )
#             metric_terms = pace.util.grid.MetricTerms(
#                 quantity_factory=quantity_factory, communicator=communicator
#             )
#             grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
#             damping_coeff = DampingCoefficients.new_from_metric_terms(metric_terms)
#             driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(
#                 metric_terms
#             )
#         return grid, grid_data, damping_coeff, driver_grid_data

#     def get_driver_state(
#         self,
#         quantity_factory: pace.util.QuantityFactory,
#         communicator: pace.util.CubedSphereCommunicator,
#     ) -> DriverState:
#         backend = quantity_factory.empty(
#             dims=[pace.util.X_DIM, pace.util.Y_DIM], units="unknown"
#         ).gt4py_backend
#         (
#             grid,
#             grid_data,
#             damping_coeff,
#             driver_grid_data,
#         ) = self._get_grid_data_damping_coeff_and_driver_grid(
#             quantity_factory, communicator, backend
#         )
#         dycore_state = self._initialize_dycore_state(
#             quantity_factory, communicator, backend
#         )
#         physics_state = pace.physics.PhysicsState.init_zeros(
#             quantity_factory=quantity_factory,
#             active_packages=["microphysics"],
#         )
#         tendency_state = TendencyState.init_zeros(quantity_factory=quantity_factory)
#         return DriverState(
#             dycore_state=dycore_state,
#             physics_state=physics_state,
#             tendency_state=tendency_state,
#             grid_data=grid_data,
#             damping_coefficients=damping_coeff,
#             driver_grid_data=driver_grid_data,
#         )

#     def _initialize_dycore_state(
#         self,
#         quantity_factory: pace.util.QuantityFactory,
#         communicator: pace.util.CubedSphereCommunicator,
#         backend: str,
#     ) -> fv3core.DycoreState:
#         (
#             grid,
#             grid_data,
#             damping_coeff,
#             driver_grid_data,
#         ) = self._get_grid_data_damping_coeff_and_driver_grid(
#             quantity_factory, communicator, backend
#         )
#         ser = self._serializer(communicator)
#         savepoint_in = ser.get_savepoint("Driver-In")[0]
#         dace_config = DaceConfig(
#             communicator,
#             backend,
#             tile_nx=self._namelist.npx,
#             tile_nz=self._namelist.npz,
#         )
#         stencil_config = pace.dsl.stencil.StencilConfig(
#             compilation_config=CompilationConfig(
#                 backend=backend, communicator=communicator
#             ),
#             dace_config=dace_config,
#         )
#         stencil_factory = StencilFactory(
#             config=stencil_config, grid_indexing=grid.grid_indexing
#         )
#         translate_object = TranslateFVDynamics(grid, self._namelist, stencil_factory)
#         input_data = translate_object.collect_input_data(ser, savepoint_in)
#         dycore_state = translate_object.state_from_inputs(input_data)
#         return dycore_state


# @InitializerSelector.register("predefined")
# @dataclasses.dataclass
# class PredefinedStateConfig(Initializer):
#     """
#     Configuration if the states are already defined.

#     Generally you will not want to use this class when initializing from yaml,
#     as it requires numpy array data to be part of the configuration dictionary
#     used to construct the class.
#     """

#     dycore_state: fv3core.DycoreState
#     physics_state: pace.physics.PhysicsState
#     tendency_state: TendencyState
#     grid_data: pace.util.grid.GridData
#     damping_coefficients: pace.util.grid.DampingCoefficients
#     driver_grid_data: pace.util.grid.DriverGridData
#     start_time: datetime = datetime(2016, 8, 1)

#     def get_driver_state(
#         self,
#         quantity_factory: pace.util.QuantityFactory,
#         communicator: pace.util.CubedSphereCommunicator,
#     ) -> DriverState:

#         return DriverState(
#             dycore_state=self.dycore_state,
#             physics_state=self.physics_state,
#             tendency_state=self.tendency_state,
#             grid_data=self.grid_data,
#             damping_coefficients=self.damping_coefficients,
#             driver_grid_data=self.driver_grid_data,
#         )
