import abc
import dataclasses
import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from pace.dsl.dace.orchestration import dace_inhibitor
from pace.fv3core.initialization.dycore_state import DycoreState
from pace.util.constants import RGRAV

from .state import DriverState


try:
    import zarr.storage as zarr_storage
except ModuleNotFoundError:
    zarr_storage = None


class Diagnostics(abc.ABC):
    @abc.abstractmethod
    def store(self, time: Union[datetime, timedelta], state: DriverState):
        ...

    @abc.abstractmethod
    def store_grid(self, grid_data: pace.util.grid.GridData):
        ...

    @abc.abstractmethod
    def cleanup(self):
        ...


@dataclasses.dataclass
class ZSelect:
    level: int
    names: List[str]

    def select_data(self, state: DycoreState):
        output = {}
        for name in self.names:
            if name not in state.__dict__.keys():
                raise ValueError(f"Invalid state variable {name} for level select")
            assert len(getattr(state, name).dims) > 2
            if getattr(state, name).dims[2] != (
                pace.util.Z_DIM or pace.util.Z_INTERFACE_DIM
            ):
                raise ValueError(
                    f"z_select only works for state variables with dimension (x, y, z). \
                        \n {name} has dimension {getattr(state, name).dims}"
                )
            var_name = f"{name}_z{self.level}"
            output[var_name] = pace.util.Quantity(
                getattr(state, name).data[:, :, self.level],
                dims=getattr(state, name).dims[0:2],
                origin=getattr(state, name).origin[0:2],
                extent=getattr(state, name).extent[0:2],
                units=getattr(state, name).units,
            )
        return output


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """
    Attributes:
        path: directory to save diagnostics if given, otherwise no diagnostics
            will be stored
        output_format: one of "zarr" or "netcdf", be careful when using the "netcdf"
            format as this requires all diagnostics to be stored in memory on the
            root rank before saving, which can cause out-of-memory errors if the
            global data size or number of variables is too large
        time_chunk_size: number of timesteps stored in each netcdf file, only used if
            output_format is "netcdf"
        names: state variables to save as diagnostics
        derived_names: derived diagnostics to save
        z_select: save a veritcal slice of a 3D state
    """

    path: Optional[str] = None
    output_format: str = "zarr"
    time_chunk_size: int = 1
    names: List[str] = dataclasses.field(default_factory=list)
    derived_names: List[str] = dataclasses.field(default_factory=list)
    z_select: List[ZSelect] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if (len(self.names) > 0 or len(self.derived_names) > 0) and self.path is None:
            raise ValueError(
                "DiagnosticsConfig.path must be given to enable diagnostics"
            )
        if self.output_format not in ["zarr", "netcdf"]:
            raise ValueError(
                "output_format must be one of 'zarr' or 'netcdf', "
                f"got {self.output_format}"
            )

    def diagnostics_factory(self, communicator: pace.util.Communicator) -> Diagnostics:
        """
        Create a diagnostics object.

        Args:
            communicator: provides global communication e.g. to gather state
                or to coordinate filesystem access between ranks
        """
        if self.path is None:
            diagnostics: Diagnostics = NullDiagnostics()
        else:
            fs = pace.util.get_fs(self.path)
            if not fs.exists(self.path):
                fs.makedirs(self.path, exist_ok=True)
            if self.output_format == "zarr":
                store = zarr_storage.DirectoryStore(path=self.path)
                monitor: pace.util.Monitor = pace.util.ZarrMonitor(
                    store=store,
                    partitioner=communicator.partitioner,
                    mpi_comm=communicator.comm,
                )
            elif self.output_format == "netcdf":
                monitor = pace.util.NetCDFMonitor(
                    path=self.path,
                    communicator=communicator,
                    time_chunk_size=self.time_chunk_size,
                )
            else:
                raise ValueError(
                    "output_format must be one of 'zarr' or 'netcdf', "
                    f"got {self.output_format}"
                )
            diagnostics = MonitorDiagnostics(
                monitor=monitor,
                names=self.names,
                derived_names=self.derived_names,
                z_select=self.z_select,
            )
        return diagnostics


class MonitorDiagnostics(Diagnostics):
    """Diagnostics that save to a sympl-style Monitor."""

    def __init__(
        self,
        monitor: pace.util.Monitor,
        names: List[str],
        derived_names: List[str],
        z_select: List[ZSelect],
    ):
        """
        Args:
            monitor: a sympl-style Monitor object
            names: list of names of diagnostics to save
            derived_names: list of names of derived diagnostics to save
        """
        self.names = names
        self.derived_names = derived_names
        self.z_select = z_select
        self.monitor = monitor

    @dace_inhibitor
    def store(self, time: Union[datetime, timedelta], state: DriverState):
        monitor_state = {"time": time}
        for name in self.names:
            try:
                quantity = getattr(state.dycore_state, name)
            except AttributeError:
                quantity = getattr(state.physics_state, name)
            monitor_state[name] = quantity
        derived_state = self._get_derived_state(state)
        level_select_state = self._get_z_select_state(state.dycore_state)
        monitor_state.update(derived_state)
        monitor_state.update(level_select_state)
        self.monitor.store(monitor_state)

    def _get_derived_state(self, state: DriverState):
        output = {}
        if len(self.derived_names) > 0:
            for name in self.derived_names:
                if name.startswith("column_integrated_"):
                    tracer = name[len("column_integrated_") :]
                    output[name] = _compute_column_integral(
                        name,
                        getattr(state.dycore_state, tracer),
                        state.dycore_state.delp,
                    )
                else:
                    warnings.warn(f"{name} is not a supported diagnostic variable.")
        return output

    def _get_z_select_state(self, state: DycoreState):
        z_select_state = {}
        for zselect in self.z_select:
            z_select_state.update(zselect.select_data(state))
        return z_select_state

    def store_grid(self, grid_data: pace.util.grid.GridData):
        zarr_grid = {
            "lat": grid_data.lat,
            "lon": grid_data.lon,
        }
        self.monitor.store_constant(zarr_grid)

    def cleanup(self):
        self.monitor.cleanup()


class NullDiagnostics(Diagnostics):
    """Diagnostics that do nothing."""

    def store(self, time: Union[datetime, timedelta], state: DriverState):
        pass

    def store_grid(self, grid_data: pace.util.grid.GridData):
        pass

    def cleanup(self):
        pass


def _compute_column_integral(
    name: str, q_in: pace.util.Quantity, delp: pace.util.Quantity
):
    """
    Compute column integrated mixing ratio (e.g., total liquid water path)

    Args:
        name: name of the tracer
        q_in: tracer mixing ratio
        delp: pressure thickness of atmospheric layer
    """
    assert len(q_in.dims) > 2
    if q_in.dims[2] != pace.util.Z_DIM:
        raise NotImplementedError(
            "this function assumes the z-dimension is the third dimension"
        )
    k_slice = slice(q_in.origin[2], q_in.origin[2] + q_in.extent[2])
    column_integral = pace.util.Quantity(
        RGRAV
        * q_in.np.sum(q_in.data[:, :, k_slice] * delp.data[:, :, k_slice], axis=2),
        dims=tuple(q_in.dims[:2]) + tuple(q_in.dims[3:]),
        origin=q_in.metadata.origin[0:2],
        extent=(q_in.metadata.extent[0], q_in.metadata.extent[1]),
        units="kg/m**2",
    )
    return column_integral
