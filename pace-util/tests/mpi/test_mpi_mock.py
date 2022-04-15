import numpy as np
import pytest
from mpi_comm import MPI

import pace.util


worker_function_list = []

MAX_WORKER_ITERATIONS = 16


def worker(rank_order=range):
    def decorator(func):
        func.rank_order = rank_order
        worker_function_list.append(func)
        return func

    return decorator


@worker()
def return_constant(comm):
    return 1


@worker()
def send_recv(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = numpy.asarray([rank], dtype=numpy.int)

    if rank < size - 1:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"sending data from {rank} to {rank + 1}")
        comm.Send(data, dest=rank + 1)
    if rank > 0:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"recieving data from {rank - 1} to {rank}")
        comm.Recv(data, source=rank - 1)
    return data


@worker()
def send_recv_big_data(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = numpy.ones([5, 3, 96], dtype=numpy.float64) * rank

    if rank < size - 1:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"sending data from {rank} to {rank + 1}")
        comm.Send(data, dest=rank + 1)
    if rank > 0:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"recieving data from {rank - 1} to {rank}")
        comm.Recv(data, source=rank - 1)
    return data


def data_send(data, to_rank):
    new_array = data.copy()
    return comm.Isend(new_array, dest=to_rank, tag=0)


@worker()
def send_recv_multiple_async_calls(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    shape = [50, 3, 48]
    data = numpy.ones(shape, dtype=numpy.float64) * rank
    recv_data = numpy.zeros([size] + shape, dtype=numpy.float64) - 1

    req_list = []

    for to_rank in range(size):
        if to_rank != rank:
            req_list.append(data_send(data, dest=to_rank))

    for from_rank in range(size):
        if from_rank != rank:
            with pace.util.recv_buffer(numpy, recv_data[from_rank, :]) as recvbuf:
                comm.Recv(recvbuf, source=from_rank, tag=0)
    for req in req_list:
        req.wait()
    return recv_data


@worker()
def send_f_contiguous_buffer(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    numpy.random.seed(rank)
    data = numpy.random.uniform(size=[2, 3]).T

    if rank < size - 1:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"sending data from {rank} to {rank + 1}")
        comm.Send(data, dest=rank + 1)
    if rank > 0:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"recieving data from {rank - 1} to {rank}")
        comm.Recv(data, source=rank - 1)
    return data


@worker()
def send_non_contiguous_buffer(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    numpy.random.seed(rank)
    data = numpy.random.uniform(size=[2, 3, 4]).transpose(2, 0, 1)
    recv_buffer = numpy.zeros([4, 2, 3])

    if rank < size - 1:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"sending data from {rank} to {rank + 1}")
        comm.Send(data, dest=rank + 1)
    if rank > 0:
        pass  # sends will raise exceptions, so we don't want to recv
    return recv_buffer


@worker()
def send_subarray(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    numpy.random.seed(rank)
    data = numpy.random.uniform(size=[4, 4, 4])
    recv_buffer = numpy.zeros([2, 2, 2])

    if rank < size - 1:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"sending data from {rank} to {rank + 1}")
        comm.Send(data[1:-1, 1:-1, 1:-1], dest=rank + 1)
    if rank > 0:
        pass  # sends will raise exceptions, so we don't want to recv
    return recv_buffer


@worker()
def recv_to_subarray(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    numpy.random.seed(rank)
    data = numpy.random.uniform(size=[2, 2, 2])
    recv_buffer = numpy.zeros([4, 4, 4])
    contiguous_recv_buffer = numpy.zeros([2, 2, 2])
    return_value = recv_buffer

    if rank < size - 1:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"sending data from {rank} to {rank + 1}")
        comm.Send(data, dest=rank + 1)
    if rank > 0:
        if isinstance(comm, pace.util.testing.DummyComm):
            print(f"recieving data from {rank - 1} to {rank}")
        try:
            comm.Recv(recv_buffer[1:-1, 1:-1, 1:-1], source=rank - 1)
        except Exception as err:
            return_value = err
        # must complete the MPI transaction for politeness to subsequent tests
        comm.Recv(contiguous_recv_buffer, source=rank - 1)
    return return_value


@worker()
def scatter(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    recvbuf = numpy.array([-1])
    if rank == 0:
        data = numpy.arange(size)[:, None]
    else:
        data = None
    comm.Scatter(data, recvbuf)
    assert recvbuf[0] == rank
    return recvbuf


@worker(rank_order=lambda total_ranks: range(total_ranks - 1, -1, -1))
def gather(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    sendbuf = numpy.array([rank])
    if rank == 0:
        recvbuf = numpy.ones([size], dtype=sendbuf.dtype)[:, None] * -1
    else:
        recvbuf = None
    comm.Gather(sendbuf, recvbuf)
    if rank == 0:
        assert numpy.all(recvbuf == numpy.arange(size)[:, None])
        return list(recvbuf)
    else:
        return recvbuf


@worker()
def isend_irecv(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = numpy.asarray([rank], dtype=numpy.int)
    if rank < size - 1:
        req = comm.Isend(data, dest=(rank + 1) % size)
        req.wait()
    if rank > 0:
        req = comm.Irecv(data, source=(rank - 1) % size)
        req.wait()
    return data


@worker()
def asynchronous_and_synchronous_send_recv(comm, numpy):
    rank = comm.Get_rank()
    size = comm.Get_size()
    data_async = numpy.asarray([rank], dtype=numpy.int)
    data_sync = numpy.asarray([-rank], dtype=numpy.int)
    if rank < size - 1:
        req = comm.Isend(data_async, dest=(rank + 1) % size)
        req.wait()
        comm.Send(data_sync, dest=(rank + 1) % size)
    if rank > 0:
        comm.Recv(data_sync, source=(rank - 1) % size)
        req = comm.Irecv(data_async, source=(rank - 1) % size)
        req.wait()
    return (data_async, data_sync)


@pytest.fixture(params=worker_function_list)
def worker_function(request):
    return request.param


def gather_decorator(worker_function):
    def wrapped(comm, numpy):
        try:
            result = worker_function(comm, numpy)
        except Exception as err:
            result = err
        return comm.gather(result, root=0)

    return wrapped


@pytest.fixture
def total_ranks():
    return MPI.COMM_WORLD.Get_size()


@pytest.fixture
def dummy_list(total_ranks):
    shared_buffer = {}
    return_list = []
    for rank in range(total_ranks):
        return_list.append(
            pace.util.testing.DummyComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
            )
        )
    return return_list


@pytest.fixture
def comm(worker_function, total_ranks):
    return MPI.COMM_WORLD


@pytest.fixture
def mpi_results(comm, worker_function, numpy):
    return gather_decorator(worker_function)(comm, numpy)


@pytest.fixture
def dummy_results(worker_function, dummy_list, numpy):
    print("Getting dummy results")
    result_list = [None] * len(dummy_list)
    done = False
    iter_count = 0
    while not done:
        iter_count += 1
        done = True
        for i in worker_function.rank_order(len(dummy_list)):
            comm = dummy_list[i]
            try:
                result_list[i] = worker_function(comm, numpy)
            except pace.util.testing.ConcurrencyError as err:
                if iter_count >= MAX_WORKER_ITERATIONS:
                    result_list[i] = err
                else:
                    done = False
            except Exception as err:
                result_list[i] = err
    return result_list


@pytest.mark.skipif(
    MPI is None, reason="mpi4py is not available or pytest was not run in parallel"
)
def test_worker(comm, dummy_results, mpi_results, numpy):
    comm.barrier()  # synchronize the test "dots" across ranks
    if comm.Get_rank() == 0:
        assert len(dummy_results) == len(mpi_results)
        for dummy, mpi in zip(dummy_results, mpi_results):
            if isinstance(mpi, numpy.ndarray):
                numpy.testing.assert_array_equal(np.asarray(dummy), np.asarray(mpi))
            elif isinstance(mpi, Exception):
                assert type(dummy) == type(mpi)
                assert dummy.args == mpi.args
            else:
                assert dummy == mpi
