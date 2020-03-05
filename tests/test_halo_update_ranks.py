import copy
import pytest
import fv3util
import utils
import numpy as np


@pytest.fixture
def numpy():
    return np


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture(params=[(1, 1)])
def layout(request):
    return request.param


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture
def shape(nz, ny, nx, dims, n_ghost):
    return (3, 3, 3)
    return_list = []
    length_dict = {
        fv3util.X_DIM: 2 * n_ghost + nx,
        fv3util.X_INTERFACE_DIM: 2 * n_ghost + nx + 1,
        fv3util.Y_DIM: 2 * n_ghost + ny,
        fv3util.Y_INTERFACE_DIM: 2 * n_ghost + ny + 1,
        fv3util.Z_DIM: nz,
        fv3util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def origin(n_ghost, dims):
    return_list = []
    origin_dict = {
        fv3util.X_DIM: n_ghost,
        fv3util.X_INTERFACE_DIM: n_ghost,
        fv3util.Y_DIM: n_ghost,
        fv3util.Y_INTERFACE_DIM: n_ghost,
        fv3util.Z_DIM: 0,
        fv3util.Z_INTERFACE_DIM: 0,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture
def extent(n_ghost, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        fv3util.X_DIM: nx,
        fv3util.X_INTERFACE_DIM: nx + 1,
        fv3util.Y_DIM: ny,
        fv3util.Y_INTERFACE_DIM: ny + 1,
        fv3util.Z_DIM: nz,
        fv3util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


@pytest.fixture
def grid():
    return fv3util.HorizontalGridSpec((1, 1))


@pytest.fixture
def cube_partitioner(grid):
    return fv3util.CubedSpherePartitioner(grid)


@pytest.fixture()
def communicator_list(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            fv3util.CubedSphereCommunicator(
                comm=utils.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
            )
        )
    return return_list


@pytest.fixture
def rank_quantity_list(total_ranks, numpy, dtype):
    quantity_list = []
    for rank in range(total_ranks):
        data = numpy.ones((3, 3), dtype=dtype) * np.nan
        data[1, 1] = rank
        quantity = fv3util.Quantity(
            data,
            dims=(fv3util.Y_DIM, fv3util.X_DIM),
            units='m',
            origin=(1, 1),
            extent=(1, 1),
        )
        quantity_list.append(quantity)
    return quantity_list


@pytest.mark.filterwarnings("ignore:invalid value encountered in remainder")
def test_correct_rank_layout(
        rank_quantity_list, communicator_list, subtests, numpy):
    for communicator, quantity in zip(communicator_list, rank_quantity_list):
        communicator.start_halo_update(quantity, 1)
    for communicator, quantity in zip(communicator_list, rank_quantity_list):
        communicator.finish_halo_update(quantity, 1)
    for rank, quantity in enumerate(rank_quantity_list):
        with subtests.test(rank=rank):
            if rank % 2 == 0:
                target_data = np.array([
                    [np.nan, rank - 1, np.nan],
                    [rank - 2, rank, rank + 1],
                    [np.nan, rank + 2, np.nan]
                ]) % 6
            else:
                target_data = np.array([
                    [np.nan, rank - 2, np.nan],
                    [rank - 1, rank, rank + 2],
                    [np.nan, rank + 1, np.nan]
                ]) % 6
            numpy.testing.assert_array_equal(quantity.data, target_data)
