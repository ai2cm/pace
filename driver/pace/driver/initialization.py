import abc
import dataclasses
import logging
import os
from datetime import datetime
from typing import ClassVar, Union
import xarray as xr
from dataclasses import fields

import f90nml
import pace.dsl.gt4py_utils as gt_utils
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
from pace.dsl.gt4py_utils import is_gpu_backend

from .registry import Registry
from .state import DriverState, TendencyState# , _restart_driver_state

try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np


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
class BaroclinicConfig(Initializer):
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
        # Ajda
        # do I need to keep metric terms here?
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=communicator
        )
        dycore_state = baroclinic_init.init_baroclinic_state(
            metric_terms,
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
class FortranRestartConfig(Initializer):
    """
    Configuration for fortran restart initialization.
    """

    path: str = "."

    def __post_init__(self):
        if "gs://" in self.path:
            # this works for the TC case
            # which is the only one that currently lives in the public bucket
            new_path = '/'.join(self.path.split(os.path.sep)[-1:]) # last dirs
            new_path = "restart_tmp"
            if os.path.isdir(new_path):
                fls = os.listdir(new_path)
                if len(fls) == 0:
                    os.system("gsutil cp -r %s/* %s" % (self.path, new_path)) # copy data
            else:
                os.makedirs(new_path, exist_ok=True) # create new dir
                os.system("gsutil cp -r %s/* %s" % (self.path, new_path)) # copy data
            self.path = new_path + os.path.sep # replace path with local path

    @property
    def start_time(self) -> datetime:
        """Reads the last line in coupler.res to find the restart time"""
        restart_files = os.listdir(self.path)
        coupler_file = [fl for fl in restart_files if "coupler.res" in fl][0]
        restart_doc = self.path + coupler_file
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
class SerialboxConfig(Initializer):
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
class PredefinedStateConfig(Initializer):
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
    # Ajda
    # not sure what to do here ... Keep arguments?

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


def _restart_driver_state(
    path: str,
    rank: int,
    quantity_factory: pace.util.QuantityFactory,
    communicator: pace.util.CubedSphereCommunicator,
    damping_coefficients: pace.util.grid.DampingCoefficients,
    driver_grid_data: pace.util.grid.DriverGridData,
    grid_data: pace.util.grid.GridData,
):

    dycore_state = fv3core.DycoreState.init_zeros(quantity_factory=quantity_factory)
    backend_uses_gpu = is_gpu_backend(dycore_state.u.metadata.gt4py_backend)

    is_fortran_restart = False
    restart_files = os.listdir(path)
    is_fortran_restart = any(fname.endswith("fv_core.res.nc") for fname in restart_files)

    if is_fortran_restart:
        _overwrite_state_from_fortran_restart(
            path,
            communicator,
            dycore_state,
            backend_uses_gpu,
        )
    else:
        _overwrite_state_from_restart(
            path,
            rank,
            dycore_state,
            "restart_dycore_state",
            backend_uses_gpu,
        )

    active_packages = ["microphysics"]
    physics_state = pace.physics.PhysicsState.init_zeros(
        quantity_factory=quantity_factory, active_packages=active_packages
    )

    physics_state.__post_init__(quantity_factory, active_packages)
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


def _overwrite_state_from_restart(
    path: str,
    rank: int,
    state: Union[fv3core.DycoreState, pace.physics.PhysicsState, TendencyState],
    restart_file_prefix: str,
    is_gpu_backend: bool,
):
    """
    Args:
        path: path to restart files
        rank: current rank number
        state: an empty state
        restart_file_prefix: file prefix name to read

    """
    df = xr.open_dataset(path + f"/{restart_file_prefix}_{rank}.nc")
    for _field in fields(type(state)):
        if "units" in _field.metadata.keys():
            if is_gpu_backend:
                if "physics" in restart_file_prefix:
                    state.__dict__[_field.name][:] = gt_utils.asarray(
                        df[_field.name].data[:], to_type=cp.ndarray
                    )
                else:
                    state.__dict__[_field.name].data[:] = gt_utils.asarray(
                        df[_field.name].data[:], to_type=cp.ndarray
                    )
            else:
                state.__dict__[_field.name].data[:] = df[_field.name].data[:]


def _overwrite_state_from_fortran_restart(
    path: str,
    communicator: pace.util.CubedSphereCommunicator,
    state: Union[fv3core.DycoreState, pace.physics.PhysicsState, TendencyState],
    is_gpu_backend: bool,
):
    """
    Args:
        path: path to restart files
        communicator:
        state: an empty state
        is_gpu_backend: 

    Returns:
        state: new state filled with restart files
    """

    state_dict = pace.util.open_restart(path, communicator, tracer_properties=extra_restart_properties, fortran_dict=fortran_restart_to_pace_dict)

    _dict_state_to_driver_state(state_dict, state, is_gpu_backend)


def _dict_state_to_driver_state(
    fortran_state: dict,
    driver_state: Union[fv3core.DycoreState, pace.physics.PhysicsState, TendencyState],
    is_gpu_backend: bool,
):
    """
    Takes a dict of state quantities with their Fortran names and a driver state
    and populates the driver state with quantities from the dict.
    """

    for field in fortran_restart_to_pace_dict.keys():
        driver_state.__dict__[field].view[:] = np.transpose(fortran_state[field].data)

        if is_gpu_backend:
            # driver_state.__dict__[field].view[:] = gt_utils.asarray(
            #     np.transpose(fortran_state[field].data), to_type=cp.ndarray,
            # )
            # Ajda
            # not sure if this will work?? Internet told me cupy has transpose
            driver_state.__dict__[field].view[:] = cp.transpose(
                fortran_state[field].data
            )
        else:
            driver_state.__dict__[field].view[:] = np.transpose(
                fortran_state[field].data
            )


def _update_fortran_restart_pe_peln(state: DriverState) -> DriverState:
    """
    Fortran restart data don't have information on pressure interface values
    and their logs. 
    This function takes the delp data (that is present in restart files), and 
    top level pressure to calculate pressure at interfaces and their log, 
    and updates the driver state with values.
    """

    ptop = state.grid_data._vertical_data.ak[0]
    pe = state.dycore_state.pe
    peln = state.dycore_state.peln
    delp = state.dycore_state.delp

    for level in range(pe.data.shape[2]):
        pe.data[:, :, level] = ptop + delp.np.sum(delp.data[:, :, :level], 2)

    peln.data[:] = pe.np.log(pe.data[:])

    state.dycore_state.pe = pe
    state.dycore_state.peln = peln

    #return state













fortran_restart_to_pace_dict = {
    "pt": "T", # air temperature
    "delp": "delp", # pressure thickness of atmospheric layer
    "phis": "phis", # surface geopotential
    "w": "W", # vertical wind
    "u": "u", # x_wind
    "v": "v", # y_wind
    "qvapor": "sphum", # specific humidity
    "qliquid": "liq_wat", # liquid water mixing ratio
    "qice": "ice_wat", # cloud ice mixing ratio
    "qrain": "rainwat", # rain mixing ratio
    "qsnow": "snowwat", # snow mixing ratio
    "qgraupel": "graupel", # graupel mixing ratio
    "qo3mr": "o3mr", # ozone mixing ratio
    #"qsgs_tke": "sgs_tke", # turbulent kinetic energy
    "qcld": "cld_amt", # cloud fraction
    "delz": "DZ", # vertical thickness of atmospheric layer
}
# not sure why qsgs breaks this... maybe it doesn't exist?

from pace.util._properties import RestartProperties
from pace.util.constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
    Z_SOIL_DIM,
)

extra_restart_properties: RestartProperties = {
    "specific humidity": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "sphum",
        "units": "g/kg",
    },
    "liquid water mixing ratio": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "liq_wat",
        "units": "g/kg",
    },
    "cloud ice mixing ratio": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "ice_wat",
        "units": "g/kg",
    },
    "rain mixing ratio": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "rainwat",
        "units": "g/kg",
    },
    "snow mixing ratio": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "snowwat",
        "units": "g/kg",
    },
    "graupel mixing ratio": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "graupel",
        "units": "g/kg",
    },
    "ozone mixing ratio": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "o3mr",
        "units": "g/kg",
    },
    "turublent kinetic energy": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "sgs_tke",
        "units": "g/kg",
    },
    "cloud fraction": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "cld_amt",
        "units": "g/kg",
    },
}