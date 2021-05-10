""" Test of the GPU to GPU communication strategy.

Those test use halo_update but are separated from the entire
"""
import pytest
import numpy as np
import fv3gfs.util
import contextlib
import functools

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request, fast):
    if fast and request.param == (1, 1):
        pytest.skip("running in fast mode")
    else:
        return request.param


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture
def tile_partitioner(layout):
    return fv3gfs.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return fv3gfs.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture
def cpu_communicators(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            fv3gfs.util.CubedSphereCommunicator(
                comm=fv3gfs.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                force_cpu=True,
                partitioner=cube_partitioner,
                timer=fv3gfs.util.Timer(),
            )
        )
    return return_list


@pytest.fixture
def gpu_communicators(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            fv3gfs.util.CubedSphereCommunicator(
                comm=fv3gfs.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                force_cpu=False,
                timer=fv3gfs.util.Timer(),
            )
        )
    return return_list


# To record the calls to cp.empty/np.empty we use a global
# dict indexed on the functions
global N_EMPTY_CALLS
N_EMPTY_CALLS = {}


@contextlib.contextmanager
def module_count_calls_to_empty(module):
    global N_EMPTY_CALLS
    N_EMPTY_CALLS[module.empty] = 0

    def count_calls(func):
        """Count func call"""

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            global N_EMPTY_CALLS
            N_EMPTY_CALLS[func] = N_EMPTY_CALLS[func] + 1
            return func(*args, **kwargs)

        return wrapped

    try:
        original = module.empty
        module.empty = count_calls(module.empty)
        yield
    finally:
        module.empty = original


@pytest.mark.parametrize("backend", ["cupy", "gt4py_cupy"], indirect=True)
def test_halo_update_only_communicate_on_gpu(backend, gpu_communicators):
    with module_count_calls_to_empty(np), module_count_calls_to_empty(cp):
        shape = (10, 10, 79)
        dims = (fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM)
        data = cp.empty(shape, dtype=float)
        quantity = fv3gfs.util.Quantity(
            data, dims=dims, units="m", origin=(3, 3, 0), extent=(3, 3, 0),
        )
        req_list = []
        for communicator in gpu_communicators:
            req = communicator.start_halo_update(quantity, 3)
            req_list.append(req)
        for req in req_list:
            req.wait()

    # We expect no np calls and several cp calls
    global N_EMPTY_CALLS
    assert N_EMPTY_CALLS[cp.empty] > 0
    assert N_EMPTY_CALLS[np.empty] == 0


@pytest.mark.parametrize("backend", ["cupy", "gt4py_cupy"], indirect=True)
def test_halo_update_communicate_though_cpu(backend, cpu_communicators):
    with module_count_calls_to_empty(np), module_count_calls_to_empty(cp):
        shape = (10, 10, 79)
        data = cp.empty(shape, dtype=float)
        quantity = fv3gfs.util.Quantity(
            data,
            dims=(fv3gfs.util.X_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM,),
            units="m",
            origin=(3, 3, 0),
            extent=(3, 3, 0),
        )
        req_list = []
        for communicator in cpu_communicators:
            req = communicator.start_halo_update(quantity, 3)
            req_list.append(req)
        for req in req_list:
            req.wait()

    # We expect several np calls and several cp calls
    global N_EMPTY_CALLS
    assert N_EMPTY_CALLS[np.empty] > 0
    assert N_EMPTY_CALLS[cp.empty] > 0
