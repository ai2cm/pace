""" Test of the GPU to GPU communication strategy.

Those test use halo_update but are separated from the entire
"""
import contextlib
import functools

import numpy as np
import pytest

import pace.util


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
    return pace.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return pace.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture
def cpu_communicators(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            pace.util.CubedSphereCommunicator(
                comm=pace.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                force_cpu=True,
                partitioner=cube_partitioner,
                timer=pace.util.Timer(),
            )
        )
    return return_list


@pytest.fixture
def gpu_communicators(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            pace.util.CubedSphereCommunicator(
                comm=pace.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                force_cpu=False,
                timer=pace.util.Timer(),
            )
        )
    return return_list


# To record the calls to cp.ZEROS/np.ZEROS we use a global
# dict indexed on the functions
global N_ZEROS_CALLS
N_ZEROS_CALLS = {}


@contextlib.contextmanager
def module_count_calls_to_zeros(module):
    global N_ZEROS_CALLS
    N_ZEROS_CALLS[module.zeros] = 0

    def count_calls(func):
        """Count func call"""

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            global N_ZEROS_CALLS
            N_ZEROS_CALLS[func] = N_ZEROS_CALLS[func] + 1
            return func(*args, **kwargs)

        return wrapped

    try:
        original = module.zeros
        module.zeros = count_calls(module.zeros)
        yield
    finally:
        module.zeros = original


@pytest.mark.parametrize("backend", ["cupy", "gt4py_cupy"], indirect=True)
def test_halo_update_only_communicate_on_gpu(backend, gpu_communicators):
    with module_count_calls_to_zeros(np), module_count_calls_to_zeros(cp):
        shape = (10, 10, 79)
        dims = (pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM)
        data = cp.ones(shape, dtype=float)
        quantity = pace.util.Quantity(
            data,
            dims=dims,
            units="m",
            origin=(3, 3, 1),
            extent=(3, 3, 1),
        )
        halo_updater_list = []
        for communicator in gpu_communicators:
            halo_updater = communicator.start_halo_update(quantity, 3)
            halo_updater_list.append(halo_updater)
        for halo_updater in halo_updater_list:
            halo_updater.wait()

    # We expect no np calls and several cp calls
    global N_ZEROS_CALLS
    print(f"Results {N_ZEROS_CALLS}")
    assert N_ZEROS_CALLS[cp.zeros] > 0
    assert N_ZEROS_CALLS[np.zeros] == 0


@pytest.mark.parametrize("backend", ["cupy", "gt4py_cupy"], indirect=True)
def test_halo_update_communicate_though_cpu(backend, cpu_communicators):
    with module_count_calls_to_zeros(np), module_count_calls_to_zeros(cp):
        shape = (10, 10, 79)
        data = cp.ones(shape, dtype=float)
        quantity = pace.util.Quantity(
            data,
            dims=(
                pace.util.X_DIM,
                pace.util.Y_DIM,
                pace.util.Z_DIM,
            ),
            units="m",
            origin=(3, 3, 0),
            extent=(3, 3, 0),
        )
        halo_updater_list = []
        for communicator in cpu_communicators:
            halo_updater = communicator.start_halo_update(quantity, 3)
            halo_updater_list.append(halo_updater)
        for halo_updater in halo_updater_list:
            halo_updater.wait()

    # We expect several np calls and several cp calls
    global N_ZEROS_CALLS
    assert N_ZEROS_CALLS[np.zeros] > 0
    assert N_ZEROS_CALLS[cp.zeros] == 0
