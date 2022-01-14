import numpy
import pytest

from pace.util import LocalComm


@pytest.fixture
def total_ranks():
    return 2


@pytest.fixture
def tags():
    return [1, 2]


@pytest.fixture
def local_communicator_list(total_ranks):
    shared_buffer = {}
    return_list = []
    for rank in range(total_ranks):
        return_list.append(
            LocalComm(rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer)
        )
    return return_list


def test_local_comm_simple(local_communicator_list):
    for comm in local_communicator_list:
        rank = comm.Get_rank()
        size = comm.Get_size()
        data = numpy.asarray([rank], dtype=numpy.int)
        if rank % 2 == 0:
            comm.Send(data, dest=(rank + 1) % size)
        else:
            comm.Recv(data, source=(rank - 1) % size)
            assert data == (rank - 1) % size


@pytest.mark.parametrize("tags", [(0, 1, 2), (2, 1, 0), (2, 0, 1)])
def test_local_comm_tags(local_communicator_list, tags):
    for comm in local_communicator_list:
        rank = comm.Get_rank()
        size = comm.Get_size()
        data = numpy.array([[rank], [rank + 1], [rank + 2]])
        if rank % 2 == 0:
            for i in range(len(tags)):
                comm.Isend(data[i], dest=(rank + 1) % size, tag=tags[i])
        else:
            rec_buffer = numpy.array([[-1], [-1], [-1]])
            for i in range(len(tags)):
                recv = comm.Irecv(rec_buffer[i], source=(rank - 1) % size, tag=i)
                recv.wait()
            assert (rec_buffer[list(tags)] == data - 1).all()
