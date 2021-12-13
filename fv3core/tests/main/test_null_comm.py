import pace.util
from fv3core.utils.null_comm import NullComm


def test_can_create_cube_communicator():
    rank = 2
    total_ranks = 24
    mpi_comm = NullComm(rank, total_ranks)
    layout = (2, 2)
    partitioner = pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout))
    communicator = pace.util.CubedSphereCommunicator(mpi_comm, partitioner)
    communicator.tile.partitioner
