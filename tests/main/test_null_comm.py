import fv3gfs.util
from fv3core.utils.null_comm import NullComm


def test_can_create_cube_communicator():
    rank = 2
    total_ranks = 24
    mpi_comm = NullComm(rank, total_ranks)
    layout = (2, 2)
    partitioner = fv3gfs.util.CubedSpherePartitioner(
        fv3gfs.util.TilePartitioner(layout)
    )
    communicator = fv3gfs.util.CubedSphereCommunicator(mpi_comm, partitioner)
    communicator.tile.partitioner
