import tempfile
import inspect
import sys
import pytest
import numpy as np
import fv3util
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if MPI is not None and MPI.COMM_WORLD.Get_size() == 1:
    # not run as a parallel test, disable MPI tests
    MPI.Finalize()
    MPI = None


worker_function_list = []


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
def send_recv_worker(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.asarray([rank], dtype=np.int)

    if rank < size - 1:
        if isinstance(comm, fv3util.testing.DummyComm):
            print(f"sending data from {rank} to {rank + 1}")
        comm.Send(data, dest=rank + 1)
    if rank > 0:
        if isinstance(comm, fv3util.testing.DummyComm):
            print(f"recieving data from {rank - 1} to {rank}")
        comm.Recv(data, source=rank - 1)
    return data


@worker()
def scatter_worker(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    recvbuf = np.array([-1])
    if rank == 0:
        data = np.arange(size)[:, None]
    else:
        data = None
    comm.Scatter(data, recvbuf)
    assert recvbuf[0] == rank
    return recvbuf


@worker(rank_order=lambda total_ranks: range(total_ranks - 1, -1, -1))
def gather_worker(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    sendbuf = np.array([rank])
    if rank == 0:
        recvbuf = np.ones(size, dtype=sendbuf.dtype)[:, None] * -1
    else:
        recvbuf = None
    comm.Gather(sendbuf, recvbuf)
    if rank == 0:
        assert np.all(recvbuf == np.arange(size)[:, None])
        return list(recvbuf)
    else:
        return recvbuf


@worker()
def isend_irecv_worker(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = np.asarray([rank], dtype=np.int)
    if rank < size - 1:
        comm.Isend(data, dest=(rank + 1) % size)
    if rank > 0:
        req = comm.Irecv(data, source=(rank - 1) % size)
        req.wait()
    return data


@worker()
def asynchronous_and_synchronous_send_recv(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    data_async = np.asarray([rank], dtype=np.int)
    data_sync = np.asarray([-rank], dtype=np.int)
    if rank < size - 1:
        comm.Isend(data_async, dest=(rank + 1) % size)
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
    def wrapped(comm):
        result = worker_function(comm)
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
            fv3util.testing.DummyComm(
                rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
            )
        )
    return return_list


@pytest.fixture
def comm(worker_function, total_ranks):
    return MPI.COMM_WORLD


@pytest.fixture
def mpi_results(comm, worker_function):
    return gather_decorator(worker_function)(comm)


@pytest.fixture
def dummy_results(worker_function, dummy_list):
    print('Getting dummy results')
    result_list = [None] * len(dummy_list)
    for i in worker_function.rank_order(len(dummy_list)):
        comm = dummy_list[i]
        result_list[i] = worker_function(comm)
    print('done getting dummy results')
    return result_list


@pytest.mark.skipif(
    MPI is None, reason='mpi4py is not available or pytest was not run in parallel'
)
def test_worker(comm, dummy_results, mpi_results):
    comm.barrier()  # synchronize the test "dots" across ranks
    if comm.Get_rank() == 0:
        assert dummy_results == mpi_results
