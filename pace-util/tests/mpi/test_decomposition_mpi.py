import unittest.mock

import pytest
from mpi_comm import MPI

from pace.util.decomposition import block_waiting_for_compilation, unblock_waiting_tiles


@pytest.mark.parallel
@pytest.mark.skipif(
    MPI is None or MPI.COMM_WORLD.Get_size() != 6,
    reason="mpi4py is not available or pytest was not run in parallel",
)
def test_unblock_waiting_tiles():
    comm = MPI.COMM_WORLD
    compilation_config = unittest.mock.MagicMock(compiling_equivalent=0)
    rank = comm.Get_rank()
    if rank != 0:
        block_waiting_for_compilation(comm, compilation_config)
    if rank == 0:
        unblock_waiting_tiles(comm)
