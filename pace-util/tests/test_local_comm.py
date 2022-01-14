import numpy
import pytest

from pace.util import local_communicator


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
            local_communicator.LocalComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
            )
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
        data0 = numpy.asarray([rank], dtype=numpy.int)
        data1 = numpy.asarray([rank + 1], dtype=numpy.int)
        data2 = numpy.asarray([rank + 2], dtype=numpy.int)
        if rank % 2 == 0:
            comm.Isend(data0, dest=(rank + 1) % size, tag=tags[0])
            comm.Isend(data1, dest=(rank + 1) % size, tag=tags[1])
            comm.Isend(data2, dest=(rank + 1) % size, tag=tags[2])
        else:
            result_order_list = [None, None, None]
            recv1 = comm.Irecv(data1, source=(rank - 1) % size, tag=tags[1])
            recv1.wait()
            result_order_list[1] = data1[0]
            recv0 = comm.Irecv(data0, source=(rank - 1) % size, tag=tags[0])
            recv0.wait()
            result_order_list[0] = data0[0]
            recv2 = comm.Irecv(data2, source=(rank - 1) % size, tag=tags[2])
            recv2.wait()
            result_order_list[2] = data2[0]
            assert result_order_list == [rank - 1, rank, rank + 1]
