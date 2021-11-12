import fv3gfs.util
from mpi4py import MPI
import numpy as np
from datetime import timedelta
import cftime
import zarr

OUTPUT_PATH = "output/zarr_monitor.zarr"


def get_example_state(time):
    sizer = fv3gfs.util.SubtileGridSizer(
        nx=48, ny=48, nz=70, n_halo=3, extra_dim_lengths={}
    )
    allocator = fv3gfs.util.QuantityFactory(sizer, np)
    air_temperature = allocator.zeros(
        [fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM], units="degK"
    )
    air_temperature.view[:] = np.random.randn(*air_temperature.extent)
    return {"time": time, "air_temperature": air_temperature}


if __name__ == "__main__":
    size = MPI.COMM_WORLD.Get_size()
    # assume square tile faces
    ranks_per_edge = int((size // 6) ** 0.5)
    layout = (ranks_per_edge, ranks_per_edge)

    store = zarr.storage.DirectoryStore(OUTPUT_PATH)
    partitioner = fv3gfs.util.CubedSpherePartitioner(
        fv3gfs.util.TilePartitioner(layout)
    )
    monitor = fv3gfs.util.ZarrMonitor(store, partitioner, mpi_comm=MPI.COMM_WORLD)

    time = cftime.DatetimeJulian(2020, 1, 1)
    timestep = timedelta(hours=1)

    for i in range(10):
        state = get_example_state(time)
        monitor.store(state)
        time += timestep
