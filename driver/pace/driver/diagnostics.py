import abc
import dataclasses
from datetime import datetime
from typing import List, Optional

import zarr.storage

import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from pace.util.quantity import QuantityMetadata

from .state import DriverState


class Diagnostics(abc.ABC):
    @abc.abstractmethod
    def store(self, time: datetime, state: DriverState):
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
        names: diagnostics to save
    """

    path: Optional[str] = None
    names: List[str] = dataclasses.field(default_factory=list)

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
                path=self.path, names=self.names, partitioner=partitioner, comm=comm
            )


class ZarrDiagnostics(Diagnostics):
    """Diagnostics that saves to a zarr store."""

    def __init__(
        self,
        path: str,
        names: List[str],
        partitioner: pace.util.CubedSpherePartitioner,
        comm,
    ):
        self.names = names
        store = zarr.storage.DirectoryStore(path=path)
        self.monitor = pace.util.ZarrMonitor(
            store=store, partitioner=partitioner, mpi_comm=comm
        )

    def store(self, time: datetime, state: DriverState):
        if len(self.names) > 0:
            zarr_state = {"time": time}
            for name in self.names:
                try:
                    quantity = getattr(state.dycore_state, name)
                except AttributeError:
                    quantity = getattr(state.physics_state, name)
                zarr_state[name] = quantity
            assert time is not None
            self.monitor.store(zarr_state)

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

    def store(self, time: datetime, state: DriverState):
        pass

    def store_grid(
        self, grid_data: pace.util.grid.GridData, metadata: QuantityMetadata
    ):
        pass
