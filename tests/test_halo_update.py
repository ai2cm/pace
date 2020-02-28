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


@pytest.fixture(params=[(1, 1), (2, 2), (3, 3)])
def layout(request):
    return request.param


@pytest.fixture
def nx_rank():
    return 3


@pytest.fixture
def ny_rank():
    return 3


@pytest.fixture
def nz():
    return 3


@pytest.fixture
def ny(ny_rank, layout):
    return ny_rank * layout[0]


@pytest.fixture
def nx(nx_rank, layout):
    return nx_rank * layout[1]


@pytest.fixture(params=[1, 3])
def n_ghost(request):
    return request.param


@pytest.fixture(params=['fewer', 'more', 'same'])
def n_ghost_update(request, n_ghost):
    return n_ghost + {'fewer': -1, 'more': 1, 'same': 0}[request.param]


@pytest.fixture(
    params=[
        pytest.param((fv3util.Y_DIM, fv3util.X_DIM), id='center'),
        pytest.param((fv3util.Z_DIM, fv3util.Y_DIM, fv3util.X_DIM), id='center_3d'),
        pytest.param((fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM), id='center_3d_reverse'),
        pytest.param((fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM), id='interface'),
        pytest.param((fv3util.Z_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM), id='interface_3d'),
    ]
)
def dims(request):
    return request.param


@pytest.fixture
def units():
    return 'm'


@pytest.fixture
def total_ranks(layout):
    return 6 * layout[0] * layout[1]


@pytest.fixture
def shape(nz, ny, nx, dims, n_ghost):
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
def zeros_quantity_list(total_ranks, dims, units, origin, extent, shape, numpy, dtype):
    return_list = []
    for rank in range(total_ranks):
        data = numpy.ones(shape, dtype=dtype)
        quantity = fv3util.Quantity(
            data,
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
        )
        quantity.view[:] = 0.
        return_list.append(quantity)
    return return_list


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
def grid(ny, nx, layout):
    return fv3util.HorizontalGridSpec(ny, nx, layout)


@pytest.fixture
def cube_partitioner(grid):
    return fv3util.CubedSpherePartitioner(grid)


@pytest.fixture
def updated_slice(ny, nx, dims, n_ghost, n_ghost_update):
    n_ghost_remain = n_ghost - n_ghost_update
    return_list = []
    length_dict = {
        fv3util.X_DIM: slice(n_ghost_remain, n_ghost + nx + n_ghost_update),
        fv3util.X_INTERFACE_DIM: slice(n_ghost_remain, n_ghost + nx + 1 + n_ghost_update),
        fv3util.Y_DIM: slice(n_ghost_remain, n_ghost + ny + n_ghost_update),
        fv3util.Y_INTERFACE_DIM: slice(n_ghost_remain, n_ghost + ny + 1 + n_ghost_update),
        fv3util.Z_DIM: slice(None, None),
        fv3util.Z_INTERFACE_DIM: slice(None, None),
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def remaining_ones(nz, ny, nx, n_ghost, n_ghost_update):
    width = n_ghost - n_ghost_update
    return (2 * nx + 2 * ny + 4 * width) * width


def test_zeros_halo_update(
        zeros_quantity_list, communicator_list, n_ghost_update, n_ghost, numpy, subtests):
    """test that zeros from adjacent domains get written over ones on local halo"""
    if 0 < n_ghost_update <= n_ghost:
        for communicator, quantity in zip(communicator_list, zeros_quantity_list):
            communicator.start_halo_update(quantity, n_ghost_update)
        for communicator, quantity in zip(communicator_list, zeros_quantity_list):
            communicator.finish_halo_update(quantity, n_ghost_update)
        if len(communicator_list) == 6:  # all subtile corners are tile corners
            for rank, quantity in enumerate(zeros_quantity_list):
                for side in ('left', 'right', 'top', 'bottom'):
                    with subtests.test(f'rank={rank}, side={side}'):
                        numpy.testing.assert_array_equal(
                            quantity.boundary_data(side, n_ghost_update, interior=False), 0.
                        )
    else:
        pytest.skip('invalid combination of n_ghost and n_ghost_update')
