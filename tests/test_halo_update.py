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


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request):
    return request.param


@pytest.fixture
def nx_rank(n_ghost):
    return max(3, n_ghost * 2 - 1)


@pytest.fixture
def ny_rank(nx_rank):
    return nx_rank


@pytest.fixture
def nz():
    return 2


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
    update = n_ghost + {'fewer': -1, 'more': 1, 'same': 0}[request.param]
    if update > n_ghost:
        pytest.skip()
    else:
        return update


@pytest.fixture(
    params=[
        pytest.param((fv3util.Y_DIM, fv3util.X_DIM), id='center'),
        pytest.param((fv3util.Z_DIM, fv3util.Y_DIM, fv3util.X_DIM), id='center_3d'),
        pytest.param((fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM), id='center_3d_reverse'),
        pytest.param((fv3util.X_DIM, fv3util.Z_DIM, fv3util.Y_DIM), id='center_3d_shuffle'),
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
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


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
def tile_partitioner(layout):
    return fv3util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return fv3util.CubedSpherePartitioner(tile_partitioner)


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


@pytest.fixture
def boundary_dict(ranks_per_tile):
    if ranks_per_tile == 1:
        return {
            0: fv3util.EDGE_BOUNDARY_TYPES
        }
    elif ranks_per_tile == 4:
        return {
            0: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_LEFT, fv3util.TOP_RIGHT, fv3util.BOTTOM_RIGHT),
            1: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_LEFT, fv3util.TOP_RIGHT, fv3util.BOTTOM_LEFT),
            2: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_RIGHT, fv3util.BOTTOM_LEFT, fv3util.BOTTOM_RIGHT),
            3: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_LEFT, fv3util.BOTTOM_LEFT, fv3util.BOTTOM_RIGHT),
        }
    elif ranks_per_tile == 9:
        return {
            0: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_LEFT, fv3util.TOP_RIGHT, fv3util.BOTTOM_RIGHT),
            1: fv3util.BOUNDARY_TYPES,
            2: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_LEFT, fv3util.TOP_RIGHT, fv3util.BOTTOM_LEFT),
            3: fv3util.BOUNDARY_TYPES,
            4: fv3util.BOUNDARY_TYPES,
            5: fv3util.BOUNDARY_TYPES,
            6: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_RIGHT, fv3util.BOTTOM_LEFT, fv3util.BOTTOM_RIGHT),
            7: fv3util.BOUNDARY_TYPES,
            8: fv3util.EDGE_BOUNDARY_TYPES + (fv3util.TOP_LEFT, fv3util.BOTTOM_LEFT, fv3util.BOTTOM_RIGHT),
        }
    else:
        raise NotImplementedError(ranks_per_tile)


@pytest.fixture
def depth_quantity_list(total_ranks, dims, units, origin, extent, shape, numpy, dtype, n_ghost):
    """A list of quantities whose value indicates the distance from the computational
    domain boundary."""
    return_list = []
    for rank in range(total_ranks):
        data = numpy.zeros(shape, dtype=dtype) + numpy.nan

        for n_inside in range(n_ghost - 1, -1, -1):
            for i, dim in enumerate(dims):
                if dim in fv3util.HORIZONTAL_DIMS:
                    pos = [slice(None, None)] * len(dims)
                    pos[i] = origin[i] + n_inside
                    data[tuple(pos)] = n_inside
                    pos[i] = origin[i] + extent[i] - 1 - n_inside
                    data[tuple(pos)] = n_inside
        for n_outside in range(1, n_ghost + 1):
            for i, dim in enumerate(dims):
                if dim in fv3util.HORIZONTAL_DIMS:
                    pos = [slice(None, None)] * len(dims)
                    pos[i] = origin[i] - n_outside
                    data[tuple(pos)] = numpy.nan
                    pos[i] = origin[i] + extent[i] + n_outside - 1
                    data[tuple(pos)] = numpy.nan
        quantity = fv3util.Quantity(
            data,
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
        )
        return_list.append(quantity)
    return return_list


def test_depth_halo_update(
        depth_quantity_list, communicator_list, n_ghost_update, n_ghost, numpy,
        subtests, boundary_dict, ranks_per_tile):
    """test that written values have the correct orientation"""
    sample_quantity = depth_quantity_list[0]
    dims = sample_quantity.dims
    y_dim, x_dim = get_horizontal_dims(sample_quantity.dims)
    y_index = sample_quantity.dims.index(y_dim)
    x_index = sample_quantity.dims.index(x_dim)
    y_extent = sample_quantity.extent[y_index]
    x_extent = sample_quantity.extent[x_index]
    if 0 < n_ghost_update <= n_ghost:
        for communicator, quantity in zip(communicator_list, depth_quantity_list):
            communicator.start_halo_update(quantity, n_ghost_update)
        for communicator, quantity in zip(communicator_list, depth_quantity_list):
            communicator.finish_halo_update(quantity, n_ghost_update)
        if dims.index(y_dim) < dims.index(x_dim):
            for rank, quantity in enumerate(depth_quantity_list):
                with subtests.test(rank=rank, quantity=quantity):
                    for dim, extent in ((y_dim, y_extent), (x_dim, x_extent)):
                        assert numpy.all(quantity.sel(**{dim: -1}) == 0)
                        assert numpy.all(quantity.sel(**{dim: extent}) == 0)
                        if n_ghost_update >= 2:
                            assert numpy.all(quantity.sel(**{dim: -2}) <= 1)
                            assert numpy.all(quantity.sel(**{dim: extent + 1}) <= 1)
                        if n_ghost_update >= 3:
                            assert numpy.all(quantity.sel(**{dim: -3}) <= 2)
                            assert numpy.all(quantity.sel(**{dim: extent + 2}) <= 2)
                        if n_ghost_update > 3:
                            raise NotImplementedError(n_ghost_update)


@pytest.fixture
def zeros_quantity_list(total_ranks, dims, units, origin, extent, shape, numpy, dtype):
    """A list of quantities whose values are 0 in the computational domain and 1
    outside of it."""
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


def test_zeros_halo_update(
        zeros_quantity_list, communicator_list, n_ghost_update, n_ghost, numpy,
        subtests, boundary_dict, ranks_per_tile):
    """test that zeros from adjacent domains get written over ones on local halo"""
    if 0 < n_ghost_update <= n_ghost:
        for communicator, quantity in zip(communicator_list, zeros_quantity_list):
            communicator.start_halo_update(quantity, n_ghost_update)
        for communicator, quantity in zip(communicator_list, zeros_quantity_list):
            communicator.finish_halo_update(quantity, n_ghost_update)
        for rank, quantity in enumerate(zeros_quantity_list):
            boundaries = boundary_dict[rank % ranks_per_tile]
            for boundary in boundaries:
                boundary_slice = fv3util.boundary._get_boundary_slice(
                    quantity.dims, quantity.origin, quantity.extent,
                    boundary, n_ghost_update, interior=False
                )
                with subtests.test(quantity=quantity, rank=rank, boundary=boundary, boundary_slice=boundary_slice):
                    numpy.testing.assert_array_equal(
                        quantity.data[tuple(boundary_slice)], 0.
                    )


def get_horizontal_dims(dims):
    for dim in fv3util.X_DIMS:
        if dim in dims:
            x_dim = dim
            break
    else:
        raise ValueError(f'no x dimension in {dims}')
    for dim in fv3util.Y_DIMS:
        if dim in dims:
            y_dim = dim
            break
    else:
        raise ValueError(f'no y dimension in {dims}')
    return y_dim, x_dim

