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
    Dataclass for selecting the implementation of GridInitializer to use.

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
        stretch_grid: whether to Schmidt transform the grid
            (local refinement)
        stretch_factor: refinement amount
        lon_target: desired center longitude for refined tile (deg)
        lat_target: desired center latitude for refined tile (deg)
        tc_ks: something to do with friction in the vertical ???Ajda
        restart_path: path to restart data
    """
    stretch_grid: bool = False
    stretch_factor: Optional[float] = None
    lon_target: Optional[float] = None
    lat_target: Optional[float] = None
    ks: int = 0
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
        np = metric_terms.lat.np

        if self.stretch_grid: # do horizontal grid transformation
            metric_terms = _transform_horizontal_grid(self, metric_terms)
        grid_data = GridData.new_from_metric_terms(metric_terms)

        if self.restart_path is not None: # read in vertical grid
            grid_data = _replace_vertical_grid(self, metric_terms)

        damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
        driver_grid_data = DriverGridData.new_from_metric_terms(metric_terms)

        return damping_coefficients, driver_grid_data, grid_data
    

@GridInitializerSelector.register("serialbox")
@dataclasses.dataclass
class SerialboxConfig(GridInitializer):
    """
    Configuration for Serialbox initialization.
    """

    path: str
    serialized_grid: bool

    @property
    def _namelist(self) -> Namelist:
        return Namelist.from_f90nml(self._f90_namelist)

    def _serializer(self, communicator: pace.util.CubedSphereCommunicator):
        import serialbox

        serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read,
            self.path,
            "Generator_rank" + str(communicator.rank),
        )
        return serializer


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


    def get_grid(
        self,
        quantity_factory: QuantityFactory,
        communicator: CubedSphereCommunicator,
        backend: str,
    ) -> Tuple[DampingCoefficients, DriverGridData, GridData]:

        backend = quantity_factory.empty(
            dims=[pace.util.X_DIM, pace.util.Y_DIM], units="unknown"
        ).gt4py_backend
        
        if self.serialized_grid:
            logger.info("Using serialized grid data")
            grid = self._get_serialized_grid(communicator, backend)
            grid_data = grid.grid_data
            driver_grid_data = grid.driver_grid_data
            damping_coefficients = grid.damping_coefficients
        else:
            logger.info("Using a grid generated from metric terms")
            grid = pace.stencils.testing.grid.Grid.with_data_from_namelist(
                self._namelist, communicator, backend
            )
            metric_terms = pace.util.grid.MetricTerms(
                quantity_factory=quantity_factory, communicator=communicator
            )
            grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
            damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
            driver_grid_data = pace.util.grid.DriverGridData.new_from_metric_terms(
                metric_terms
            )
        
        return damping_coefficients, driver_grid_data, grid_data



def _transform_horizontal_grid(self, metric_terms: MetricTerms) -> MetricTerms:
    """
    Uses the Schmidt transform to locally refine the horizontal grid.
    """
    grid = metric_terms.grid
    lon_transform, lat_transform = direct_transform(
        lon=grid.data[:, :, 0],
        lat=grid.data[:, :, 1],
        stretch_factor=self.stretch_factor,
        lon_target=self.lon_target,
        lat_target=self.lat_target,
        np=grid.np,
    )
    grid.data[:, :, 0] = lon_transform[:]
    grid.data[:, :, 1] = lat_transform[:]

    metric_terms._grid.data[:] = grid.data[:]
    metric_terms._init_agrid()

    return metric_terms
        
        
def _replace_vertical_grid(self, metric_terms) -> GridData:
    """
    Replaces the vertical grid generators from metric terms (ak and bk) with
    their fortran restart values (in fv_core.res.nc).
    Then re-generates grid data with the new vertical inputs.
    p_ref(?)
    """
    
    restart_files = os.listdir(self.restart_path)
    data_file = [fl for fl in restart_files if "fv_core.res.nc" in fl][0] 

    ak_bk_data_file = self.restart_path + data_file
    if not os.path.isfile(ak_bk_data_file):
        raise ValueError(
            """use_tc_vertical_grid is true,
            but no fv_core.res.nc in restart data file."""
        )

    ds = xr.open_dataset(ak_bk_data_file).isel(Time=0).drop_vars("Time")
    metric_terms._ak.data[:] = ds["ak"].values
    metric_terms._bk.data[:] = ds["bk"].values
    ds.close()

    vertical_data = pace.util.grid.VerticalGridData(ks=self.ks, ak=metric_terms.ak.data, bk=metric_terms.bk.data)
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