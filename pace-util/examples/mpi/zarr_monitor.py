from datetime import timedelta

import cftime
import numpy as np
import zarr
from mpi4py import MPI

import pace.util


OUTPUT_PATH = "output/zarr_monitor.zarr"


def get_example_state(time):
    sizer = pace.util.SubtileGridSizer(
        nx=48, ny=48, nz=70, n_halo=3, extra_dim_lengths={}
    )
    allocator = pace.util.QuantityFactory(sizer, np)
    air_temperature = allocator.zeros(
        [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM], units="degK"
    )
    air_temperature.view[:] = np.random.randn(*air_temperature.extent)
    return {"time": time, "air_temperature": air_temperature}


if __name__ == "__main__":
    size = MPI.COMM_WORLD.Get_size()
    # assume square tile faces
    ranks_per_edge = int((size // 6) ** 0.5)
    layout = (ranks_per_edge, ranks_per_edge)

    store = zarr.storage.DirectoryStore(OUTPUT_PATH)
    partitioner = pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout))
    monitor = pace.util.ZarrMonitor(store, partitioner, mpi_comm=MPI.COMM_WORLD)

    time = cftime.DatetimeJulian(2020, 1, 1)
    timestep = timedelta(hours=1)

    for i in range(10):
        state = get_example_state(time)
        monitor.store(state)
        time += timestep
