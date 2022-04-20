import dataclasses
from datetime import datetime
from typing import List

import zarr.storage

import pace.driver
import pace.dsl
import pace.stencils
import pace.util
import pace.util.grid
from pace.util.quantity import QuantityMetadata

from .state import DriverState


@dataclasses.dataclass(frozen=True)
class DiagnosticsConfig:
    path: str
    names: List[str] = dataclasses.field(default_factory=list)


class Diagnostics:
    def __init__(
        self,
        config: DiagnosticsConfig,
        partitioner: pace.util.CubedSpherePartitioner,
        comm,
    ):
        self.config = config
        store = zarr.storage.DirectoryStore(path=self.config.path)
        self.monitor = pace.util.ZarrMonitor(
            store=store, partitioner=partitioner, mpi_comm=comm
        )

    def store(self, time: datetime, state: DriverState):
        if len(self.config.names) > 0:
            zarr_state = {"time": time}
            for name in self.config.names:
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
