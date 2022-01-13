import numpy
import pytest

import pace.util
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
    buffer = pace.util.Buffer
    for comm in local_communicator_list:
        rank = comm.Get_rank()
        size = comm.Get_size()
        data0 = numpy.asarray([rank], dtype=numpy.int)
        if rank % 2 == 0:
            comm.Send(data0, dest=(rank + 1) % size)
        else:
            comm.Recv(data0, source=(rank - 1) % size)
            assert data0 == (rank - 1) % size


def test_local_comm_tags_inorder(local_communicator_list):
    buffer = pace.util.Buffer
    for comm in local_communicator_list:
        rank = comm.Get_rank()
        size = comm.Get_size()
        data0 = numpy.asarray([rank], dtype=numpy.int)
        data1 = numpy.asarray([rank + 1], dtype=numpy.int)
        data2 = numpy.asarray([rank + 2], dtype=numpy.int)
        if rank % 2 == 0:
            comm.Isend(data0, dest=(rank + 1) % size, tag=0)
            comm.Isend(data1, dest=(rank + 1) % size, tag=1)
            comm.Isend(data2, dest=(rank + 1) % size, tag=2)
        else:
            result_order_list = []
            comm.Irecv(data0, source=(rank - 1) % size, tag=0)
            comm.Irecv(data0, source=(rank - 1) % size, tag=1)
            comm.Irecv(data0, source=(rank - 1) % size, tag=2)
            result_order_list = (data0[0], data1[0], data2[0])
            assert result_order_list == [rank - 1, rank, rank + 1]


def test_local_comm_tags_reversed(local_communicator_list):
    buffer = pace.util.Buffer
    for comm in local_communicator_list:
        rank = comm.Get_rank()
        size = comm.Get_size()
        data0 = numpy.asarray([rank], dtype=numpy.int)
        data1 = numpy.asarray([rank + 1], dtype=numpy.int)
        data2 = numpy.asarray([rank + 2], dtype=numpy.int)
        if rank % 2 == 0:
            comm.Isend(data0, dest=(rank + 1) % size, tag=0)
            comm.Isend(data1, dest=(rank + 1) % size, tag=1)
            comm.Isend(data2, dest=(rank + 1) % size, tag=2)
        else:
            result_order_list = []
            comm.Irecv(data0, source=(rank - 1) % size, tag=2)
            comm.Irecv(data0, source=(rank - 1) % size, tag=1)
            comm.Irecv(data0, source=(rank - 1) % size, tag=0)
            result_order_list = (data0[0], data1[0], data2[0])
            assert result_order_list == [rank + 1, rank, rank - 1]
