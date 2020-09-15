import pytest
import fv3gfs.util


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
def shape(nz, ny, nx, dims, n_points):
    return (3, 3, 3)
    return_list = []
    length_dict = {
        fv3gfs.util.X_DIM: 2 * n_points + nx,
        fv3gfs.util.X_INTERFACE_DIM: 2 * n_points + nx + 1,
        fv3gfs.util.Y_DIM: 2 * n_points + ny,
        fv3gfs.util.Y_INTERFACE_DIM: 2 * n_points + ny + 1,
        fv3gfs.util.Z_DIM: nz,
        fv3gfs.util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def origin(n_points, dims):
    return_list = []
    origin_dict = {
        fv3gfs.util.X_DIM: n_points,
        fv3gfs.util.X_INTERFACE_DIM: n_points,
        fv3gfs.util.Y_DIM: n_points,
        fv3gfs.util.Y_INTERFACE_DIM: n_points,
        fv3gfs.util.Z_DIM: 0,
        fv3gfs.util.Z_INTERFACE_DIM: 0,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture
def extent(n_points, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        fv3gfs.util.X_DIM: nx,
        fv3gfs.util.X_INTERFACE_DIM: nx + 1,
        fv3gfs.util.Y_DIM: ny,
        fv3gfs.util.Y_INTERFACE_DIM: ny + 1,
        fv3gfs.util.Z_DIM: nz,
        fv3gfs.util.Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


@pytest.fixture
def tile_partitioner(layout):
    return fv3gfs.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return fv3gfs.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture()
def communicator_list(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            fv3gfs.util.CubedSphereCommunicator(
                comm=fv3gfs.util.testing.DummyComm(
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
        data = numpy.empty((3, 3), dtype=dtype)
        data[:] = numpy.nan
        data[1, 1] = rank
        quantity = fv3gfs.util.Quantity(
            data,
            dims=(fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM),
            units="m",
            origin=(1, 1),
            extent=(1, 1),
        )
        quantity_list.append(quantity)
    return quantity_list


@pytest.mark.filterwarnings("ignore:invalid value encountered in remainder")
def test_correct_rank_layout(rank_quantity_list, communicator_list, subtests, numpy):
    req_list = []
    for communicator, quantity in zip(communicator_list, rank_quantity_list):
        req = communicator.start_halo_update(quantity, 1)
        req_list.append(req)
    for req in req_list:
        req.wait()
    for rank, quantity in enumerate(rank_quantity_list):
        with subtests.test(rank=rank):
            if rank % 2 == 0:
                target_data = (
                    numpy.array(
                        [
                            [numpy.nan, rank - 1, numpy.nan],
                            [rank - 2, rank, rank + 1],
                            [numpy.nan, rank + 2, numpy.nan],
                        ]
                    )
                    % 6
                )
            else:
                target_data = (
                    numpy.array(
                        [
                            [numpy.nan, rank - 2, numpy.nan],
                            [rank - 1, rank, rank + 2],
                            [numpy.nan, rank + 1, numpy.nan],
                        ]
                    )
                    % 6
                )
            numpy.testing.assert_array_equal(quantity.data, target_data)
