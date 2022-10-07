import copy

import pytest
from mpi_comm import MPI

import pace.util


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture
def layout():
    if MPI is not None:
        size = MPI.COMM_WORLD.Get_size()
        ranks_per_tile = size // 6
        ranks_per_edge = int(ranks_per_tile ** 0.5)
        return (ranks_per_edge, ranks_per_edge)
    else:
        return (1, 1)


@pytest.fixture
def nz():
    return 70


@pytest.fixture
def ny(n_points, layout):
    ny_rank = max(12, n_points * 2 - 1)
    return ny_rank * layout[0]


@pytest.fixture
def nx(n_points, layout):
    nx_rank = max(12, n_points * 2 - 1)
    return nx_rank * layout[1]


@pytest.fixture(params=[1, 3])
def n_points(request):
    return request.param


@pytest.fixture(params=["fewer", "same"])
def n_points_update(request, n_points):
    update = n_points + {"fewer": -1, "same": 0}[request.param]
    if update > n_points:
        pytest.skip("cannot update more points than exist in the halo")
    elif update <= 0:
        pytest.skip("cannot update fewer than 1 points")
    else:
        return update


@pytest.fixture(
    params=[
        pytest.param((pace.util.Y_DIM, pace.util.X_DIM), id="center"),
        pytest.param(
            (pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM), id="center_3d"
        ),
        pytest.param(
            (pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM),
            id="center_3d_reverse",
        ),
        pytest.param(
            (pace.util.X_DIM, pace.util.Z_DIM, pace.util.Y_DIM),
            id="center_3d_shuffle",
        ),
        pytest.param(
            (pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM), id="interface"
        ),
        pytest.param(
            (
                pace.util.Z_INTERFACE_DIM,
                pace.util.Y_INTERFACE_DIM,
                pace.util.X_INTERFACE_DIM,
            ),
            id="interface_3d",
        ),
    ]
)
def dims(request):
    return request.param


@pytest.fixture
def units():
    return "m"


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture(params=[0, 1])
def n_buffer(request):
    return request.param


@pytest.fixture
def shape(nz, ny, nx, dims, n_points, n_buffer):
    return_list = []
    length_dict = {
        pace.util.X_DIM: 2 * n_points + nx + n_buffer,
        pace.util.X_INTERFACE_DIM: 2 * n_points + nx + 1 + n_buffer,
        pace.util.Y_DIM: 2 * n_points + ny + n_buffer,
        pace.util.Y_INTERFACE_DIM: 2 * n_points + ny + 1 + n_buffer,
        pace.util.Z_DIM: nz + n_buffer,
        pace.util.Z_INTERFACE_DIM: nz + 1 + n_buffer,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def origin(n_points, dims):
    return_list = []
    origin_dict = {
        pace.util.X_DIM: n_points,
        pace.util.X_INTERFACE_DIM: n_points,
        pace.util.Y_DIM: n_points,
        pace.util.Y_INTERFACE_DIM: n_points,
        pace.util.Z_DIM: 0,
        pace.util.Z_INTERFACE_DIM: 0,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture
def extent(n_points, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        pace.util.X_DIM: nx,
        pace.util.X_INTERFACE_DIM: nx + 1,
        pace.util.Y_DIM: ny,
        pace.util.Y_INTERFACE_DIM: ny + 1,
        pace.util.Z_DIM: nz,
        pace.util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


@pytest.fixture()
def communicator(cube_partitioner):
    return pace.util.CubedSphereCommunicator(
        comm=MPI.COMM_WORLD,
        partitioner=cube_partitioner,
    )


@pytest.fixture(params=[0.1, 1.0])
def edge_interior_ratio(request):
    return request.param


@pytest.fixture
def tile_partitioner(layout, edge_interior_ratio: float):
    return pace.util.TilePartitioner(layout, edge_interior_ratio=edge_interior_ratio)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return pace.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture
def updated_slice(ny, nx, dims, n_points, n_points_update):
    n_points_remain = n_points - n_points_update
    return_list = []
    length_dict = {
        pace.util.X_DIM: slice(n_points_remain, n_points + nx + n_points_update),
        pace.util.X_INTERFACE_DIM: slice(
            n_points_remain, n_points + nx + 1 + n_points_update
        ),
        pace.util.Y_DIM: slice(n_points_remain, n_points + ny + n_points_update),
        pace.util.Y_INTERFACE_DIM: slice(
            n_points_remain, n_points + ny + 1 + n_points_update
        ),
        pace.util.Z_DIM: slice(None, None),
        pace.util.Z_INTERFACE_DIM: slice(None, None),
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def remaining_ones(nz, ny, nx, n_points, n_points_update):
    width = n_points - n_points_update
    return (2 * nx + 2 * ny + 4 * width) * width


@pytest.fixture
def boundary_dict(ranks_per_tile):
    if ranks_per_tile == 1:
        return {0: pace.util.EDGE_BOUNDARY_TYPES}
    elif ranks_per_tile == 4:
        return {
            0: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHWEST, pace.util.NORTHEAST, pace.util.SOUTHEAST),
            1: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHWEST, pace.util.NORTHEAST, pace.util.SOUTHWEST),
            2: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHEAST, pace.util.SOUTHWEST, pace.util.SOUTHEAST),
            3: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHWEST, pace.util.SOUTHWEST, pace.util.SOUTHEAST),
        }
    elif ranks_per_tile == 9:
        return {
            0: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHWEST, pace.util.NORTHEAST, pace.util.SOUTHEAST),
            1: pace.util.BOUNDARY_TYPES,
            2: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHWEST, pace.util.NORTHEAST, pace.util.SOUTHWEST),
            3: pace.util.BOUNDARY_TYPES,
            4: pace.util.BOUNDARY_TYPES,
            5: pace.util.BOUNDARY_TYPES,
            6: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHEAST, pace.util.SOUTHWEST, pace.util.SOUTHEAST),
            7: pace.util.BOUNDARY_TYPES,
            8: pace.util.EDGE_BOUNDARY_TYPES
            + (pace.util.NORTHWEST, pace.util.SOUTHWEST, pace.util.SOUTHEAST),
        }
    else:
        raise NotImplementedError(ranks_per_tile)


@pytest.fixture
def depth_quantity(
    dims, units, origin, extent, shape, numpy, dtype, n_points, n_buffer
):
    """A quantity whose value indicates the distance from the computational
    domain boundary."""
    data = numpy.zeros(shape, dtype=dtype)
    data[:] = numpy.nan
    for n_inside in range(max(n_points, max(extent) // 2), -1, -1):
        for i, dim in enumerate(dims):
            if (n_inside <= extent[i] // 2) and (dim in pace.util.HORIZONTAL_DIMS):
                pos = [slice(None, None)] * len(dims)
                pos[i] = origin[i] + n_inside
                data[tuple(pos)] = n_inside
                pos[i] = origin[i] + extent[i] - 1 - n_inside
                data[tuple(pos)] = n_inside
    for n_outside in range(1, n_points + 1 + n_buffer):
        for i, dim in enumerate(dims):
            if dim in pace.util.HORIZONTAL_DIMS:
                pos = [slice(None, None)] * len(dims)
                pos[i] = origin[i] - n_outside
                data[tuple(pos)] = numpy.nan
                pos[i] = origin[i] + extent[i] + n_outside - 1
                data[tuple(pos)] = numpy.nan
    quantity = pace.util.Quantity(
        data,
        dims=dims,
        units=units,
        origin=origin,
        extent=extent,
    )
    return quantity


@pytest.mark.skipif(
    MPI is None, reason="mpi4py is not available or pytest was not run in parallel"
)
def test_depth_halo_update(
    depth_quantity,
    communicator,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that written values have the correct orientation"""
    y_dim, x_dim = get_horizontal_dims(depth_quantity.dims)
    y_index = depth_quantity.dims.index(y_dim)
    x_index = depth_quantity.dims.index(x_dim)
    y_extent = depth_quantity.extent[y_index]
    x_extent = depth_quantity.extent[x_index]
    quantity = depth_quantity
    if 0 < n_points_update <= n_points:
        communicator.halo_update(quantity, n_points_update)
        for dim, extent in ((y_dim, y_extent), (x_dim, x_extent)):
            assert numpy.all(quantity.sel(**{dim: -1}) <= 1)
            assert numpy.all(quantity.sel(**{dim: extent}) <= 1)
            if n_points_update >= 2:
                assert numpy.all(quantity.sel(**{dim: -2}) <= 2)
                assert numpy.all(quantity.sel(**{dim: extent + 1}) <= 2)
            if n_points_update >= 3:
                assert numpy.all(quantity.sel(**{dim: -3}) <= 3)
                assert numpy.all(quantity.sel(**{dim: extent + 2}) <= 3)
            if n_points_update > 3:
                raise NotImplementedError(n_points_update)


@pytest.fixture
def zeros_quantity(dims, units, origin, extent, shape, numpy, dtype):
    """A list of quantities whose values are 0 in the computational domain and 1
    outside of it."""
    data = numpy.ones(shape, dtype=dtype)
    quantity = pace.util.Quantity(
        data,
        dims=dims,
        units=units,
        origin=origin,
        extent=extent,
    )
    quantity.view[:] = 0.0
    return quantity


@pytest.mark.skipif(
    MPI is None, reason="mpi4py is not available or pytest was not run in parallel"
)
def test_zeros_halo_update(
    zeros_quantity,
    communicator,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that zeros from adjacent domains get written over ones on local halo"""
    quantity = zeros_quantity
    if 0 < n_points_update <= n_points:
        communicator.halo_update(quantity, n_points_update)
        boundaries = boundary_dict[communicator.rank % ranks_per_tile]
        for boundary in boundaries:
            boundary_slice = pace.util._boundary_utils.get_boundary_slice(
                quantity.dims,
                quantity.origin,
                quantity.extent,
                quantity.data.shape,
                boundary,
                n_points_update,
                interior=False,
            )
            with subtests.test(
                quantity=quantity,
                rank=communicator.rank,
                boundary=boundary,
                boundary_slice=boundary_slice,
            ):
                numpy.testing.assert_array_equal(
                    quantity.data[tuple(boundary_slice)], 0.0
                )


@pytest.mark.skipif(
    MPI is None, reason="mpi4py is not available or pytest was not run in parallel"
)
def test_zeros_vector_halo_update(
    zeros_quantity,
    communicator,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that zeros from adjacent domains get written over ones on local halo"""
    x_quantity = zeros_quantity
    y_quantity = copy.deepcopy(x_quantity)
    if 0 < n_points_update <= n_points:
        communicator.vector_halo_update(y_quantity, x_quantity, n_points_update)
        boundaries = boundary_dict[communicator.rank % ranks_per_tile]
        for boundary in boundaries:
            boundary_slice = pace.util._boundary_utils.get_boundary_slice(
                x_quantity.dims,
                x_quantity.origin,
                x_quantity.extent,
                x_quantity.data.shape,
                boundary,
                n_points_update,
                interior=False,
            )
            with subtests.test(
                x_quantity=x_quantity,
                rank=communicator.rank,
                boundary=boundary,
                boundary_slice=boundary_slice,
            ):
                for quantity in y_quantity, x_quantity:
                    numpy.testing.assert_array_equal(
                        quantity.data[tuple(boundary_slice)], 0.0
                    )


def get_horizontal_dims(dims):
    for dim in pace.util.X_DIMS:
        if dim in dims:
            x_dim = dim
            break
    else:
        raise ValueError(f"no x dimension in {dims}")
    for dim in pace.util.Y_DIMS:
        if dim in dims:
            y_dim = dim
            break
    else:
        raise ValueError(f"no y dimension in {dims}")
    return y_dim, x_dim
