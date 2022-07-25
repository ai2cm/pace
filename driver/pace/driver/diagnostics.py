import abc
import dataclasses
import warnings
from datetime import datetime, timedelta
from typing import List, Optional, Union

from sympy import ShapeError

import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from pace.dsl.dace.orchestration import dace_inhibitor
from pace.util.quantity import QuantityMetadata

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
    def store_grid(
        self, grid_data: pace.util.grid.GridData, metadata: QuantityMetadata
    ):
        ...


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    """
    Attributes:
        path: location to save diagnostics if given, otherwise no diagnostics
            will be stored
        output_frequency: frequency in which diagnostics writes output, defaults
            to every timestep
        output_initial_state: flag to determine if the first output should be the
            initial state of the model before timestepping
        names: diagnostics to save
        derived_names: derived diagnostics to save
    """

    path: Optional[str] = None
    output_frequency: int = 1
    output_initial_state: bool = False
    names: List[str] = dataclasses.field(default_factory=list)
    derived_names: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if (len(self.names) > 0 or len(self.derived_names) > 0) and self.path is None:
            raise ValueError(
                "DiagnosticsConfig.path must be given to enable diagnostics"
            )

    def diagnostics_factory(
        self, partitioner: pace.util.CubedSpherePartitioner, comm
    ) -> Diagnostics:
        """
        Create a diagnostics object.

        Args:
            partitioner: defines the grid on which diagnostics are stored
            comm: an mpi4py-style communicator required to coordinate the
                storage of the diagnostics
        """
        if self.path is None:
            return NullDiagnostics()
        else:
            return ZarrDiagnostics(
                path=self.path,
                names=self.names,
                derived_names=self.derived_names,
                partitioner=partitioner,
                comm=comm,
            )


class ZarrDiagnostics(Diagnostics):
    """Diagnostics that saves to a zarr store."""

    def __init__(
        self,
        path: str,
        names: List[str],
        derived_names: List[str],
        partitioner: pace.util.CubedSpherePartitioner,
        comm,
    ):
        if zarr_storage is None:
            raise ModuleNotFoundError("zarr must be installed to use this class")
        else:
            self.names = names
            self.derived_names = derived_names
            store = zarr_storage.DirectoryStore(path=path)
            self.monitor = pace.util.ZarrMonitor(
                store=store, partitioner=partitioner, mpi_comm=comm
            )

    @dace_inhibitor
    def store(self, time: Union[datetime, timedelta], state: DriverState):
        if len(self.names) > 0:
            zarr_state = {"time": time}
            for name in self.names:
                try:
                    quantity = getattr(state.dycore_state, name)
                except AttributeError:
                    quantity = getattr(state.physics_state, name)
                zarr_state[name] = quantity
            derived_state = self._get_derived_state(state)
            zarr_state.update(derived_state)
            assert time is not None
            self.monitor.store(zarr_state)

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


class NullDiagnostics(Diagnostics):
    """Diagnostics that do nothing."""

    def store(self, time: Union[datetime, timedelta], state: DriverState):
        pass

    def store_grid(
        self, grid_data: pace.util.grid.GridData, metadata: QuantityMetadata
    ):
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
    if len(q_in.shape) < 3:
        assert ShapeError(f"{name} does not have vertical levels.")
    column_integral = pace.util.Quantity(
        sum(
            q_in.data[:, :, k] * delp.data[:, :, k]
            for k in range(q_in.metadata.extent[2])
        ),
        dims=("x", "y"),
        origin=q_in.metadata.origin[0:2],
        extent=(q_in.metadata.extent[0], q_in.metadata.extent[1]),
        units="kg/m**2",
    )
    return column_integral
