import copy

import pytest

import pace.util
from pace.util.buffer import BUFFER_CACHE


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request, fast):
    if fast and request.param == (1, 1):
        pytest.skip("running in fast mode")
    else:
        return request.param


@pytest.fixture
def nz():
    return 2


@pytest.fixture
def ny(n_points, layout):
    ny_rank = max(12, n_points * 2 - 1)
    return ny_rank * layout[0]


@pytest.fixture
def nx(n_points, layout):
    nx_rank = max(12, n_points * 2 - 1)
    return nx_rank * layout[1]


@pytest.fixture(params=[1, 3])
def n_points(request, fast):
    if fast and request.param == 1:
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture(params=["fewer", "more", "same"])
def n_points_update(request, n_points, fast):
    if fast and request.param == "same":
        pytest.skip("running in fast mode")
    return n_points + {"fewer": -1, "more": 1, "same": 0}[request.param]


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
def dims(request, fast):
    if fast and request.param in (
        (pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM),
        (
            pace.util.Z_INTERFACE_DIM,
            pace.util.Y_INTERFACE_DIM,
            pace.util.X_INTERFACE_DIM,
        ),
    ):
        pytest.skip("running in fast mode")
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
def origin(n_points, dims, n_buffer):
    return_list = []
    origin_dict = {
        pace.util.X_DIM: n_points + n_buffer,
        pace.util.X_INTERFACE_DIM: n_points + n_buffer,
        pace.util.Y_DIM: n_points + n_buffer,
        pace.util.Y_INTERFACE_DIM: n_points + n_buffer,
        pace.util.Z_DIM: n_buffer,
        pace.util.Z_INTERFACE_DIM: n_buffer,
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


@pytest.fixture
def communicator_list(cube_partitioner, total_ranks):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            pace.util.CubedSphereCommunicator(
                comm=pace.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                timer=pace.util.Timer(),
            )
        )
    return return_list


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
def depth_quantity_list(
    total_ranks, dims, units, origin, extent, shape, numpy, dtype, n_points
):
    """A list of quantities whose value indicates the distance from the computational
    domain boundary."""
    return_list = []
    for rank in range(total_ranks):
        data = numpy.empty(shape, dtype=dtype)
        data[:] = numpy.nan
        for n_inside in range(max(n_points, max(extent) // 2), -1, -1):
            for i, dim in enumerate(dims):
                if (n_inside <= extent[i] // 2) and (dim in pace.util.HORIZONTAL_DIMS):
                    pos = [slice(None, None)] * len(dims)
                    pos[i] = origin[i] + n_inside
                    data[tuple(pos)] = n_inside
                    pos[i] = origin[i] + extent[i] - 1 - n_inside
                    data[tuple(pos)] = n_inside
        for n_outside in range(1, n_points + 1):
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
        return_list.append(quantity)
    return return_list


@pytest.mark.parametrize(
    "layout, n_points, n_points_update, dims",
    [[(1, 1), 3, "same", [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM]]],
    indirect=True,
)
def test_halo_update_timer(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """
    test that halo update produces nonzero timings for all expected labels
    """
    halo_updater_list = []
    for communicator, quantity in zip(communicator_list, zeros_quantity_list):
        halo_updater = communicator.start_halo_update(quantity, n_points_update)
        halo_updater_list.append(halo_updater)
    for halo_updater in halo_updater_list:
        halo_updater.wait()
    required_times_keys = ("pack", "unpack", "Isend", "Irecv", "wait")
    for communicator in communicator_list:
        with subtests.test(rank=communicator.rank):
            assert isinstance(communicator.timer, pace.util.Timer)
            times = communicator.timer.times
            missing_keys = set(required_times_keys).difference(times.keys())
            assert len(missing_keys) == 0
            extra_keys = set(times.keys()).difference(required_times_keys)
            assert len(extra_keys) == 0
            for key in required_times_keys:
                assert times[key] > 0.0
                assert isinstance(times[key], float)


def test_depth_halo_update(
    depth_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that written values have the correct orientation"""
    sample_quantity = depth_quantity_list[0]
    y_dim, x_dim = get_horizontal_dims(sample_quantity.dims)
    y_index = sample_quantity.dims.index(y_dim)
    x_index = sample_quantity.dims.index(x_dim)
    y_extent = sample_quantity.extent[y_index]
    x_extent = sample_quantity.extent[x_index]
    halo_updater_list = []
    if 0 < n_points_update <= n_points:
        for communicator, quantity in zip(communicator_list, depth_quantity_list):
            halo_updater = communicator.start_halo_update(quantity, n_points_update)
            halo_updater_list.append(halo_updater)
        for halo_updater in halo_updater_list:
            halo_updater.wait()
        for rank, quantity in enumerate(depth_quantity_list):
            with subtests.test(rank=rank, quantity=quantity):
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
def zeros_quantity_list(total_ranks, dims, units, origin, extent, shape, numpy, dtype):
    """A list of quantities whose values are 0 in the computational domain and 1
    outside of it."""
    return_list = []
    for rank in range(total_ranks):
        data = numpy.ones(shape, dtype=dtype)
        quantity = pace.util.Quantity(
            data,
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
        )
        quantity.view[:] = 0.0
        return_list.append(quantity)
    return return_list


@pytest.mark.parametrize(
    "n_points, n_points_update, n_buffer", [(2, "more", 0)], indirect=True
)
def test_too_many_points_requested(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """
    test that an exception is raised when trying to update more halo points than exist
    """
    for communicator, quantity in zip(communicator_list, zeros_quantity_list):
        with pytest.raises(pace.util.OutOfBoundsError):
            communicator.start_halo_update(quantity, n_points_update)


def test_zeros_halo_update(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that zeros from adjacent domains get written over ones on local halo"""
    halo_updater_list = []
    if 0 < n_points_update <= n_points:
        for communicator, quantity in zip(communicator_list, zeros_quantity_list):
            halo_updater = communicator.start_halo_update(quantity, n_points_update)
            halo_updater_list.append(halo_updater)
        for halo_updater in halo_updater_list:
            halo_updater.wait()
        for rank, quantity in enumerate(zeros_quantity_list):
            boundaries = boundary_dict[rank % ranks_per_tile]
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
                    rank=rank,
                    boundary=boundary,
                    boundary_slice=boundary_slice,
                ):
                    numpy.testing.assert_array_equal(
                        quantity.data[tuple(boundary_slice)], 0.0
                    )


def test_zeros_vector_halo_update(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """test that zeros from adjacent domains get written over ones on local halo"""
    x_list = zeros_quantity_list
    y_list = copy.deepcopy(x_list)
    if 0 < n_points_update <= n_points:
        halo_updater_list = []
        for communicator, y_quantity, x_quantity in zip(
            communicator_list, y_list, x_list
        ):
            halo_updater_list.append(
                communicator.start_vector_halo_update(
                    y_quantity, x_quantity, n_points_update
                )
            )
        for halo_updater in halo_updater_list:
            halo_updater.wait()
        for rank, (y_quantity, x_quantity) in enumerate(zip(y_list, x_list)):
            boundaries = boundary_dict[rank % ranks_per_tile]
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
                    rank=rank,
                    boundary=boundary,
                    boundary_slice=boundary_slice,
                ):
                    for quantity in y_quantity, x_quantity:
                        numpy.testing.assert_array_equal(
                            quantity.data[tuple(boundary_slice)], 0.0
                        )


@pytest.mark.parametrize(
    "layout, n_points, n_points_update, dims",
    [[(1, 1), 3, "same", [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM]]],
    indirect=True,
)
def test_vector_halo_update_timer(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """
    test that halo update produces nonzero timings for all expected labels
    """
    x_list = zeros_quantity_list
    y_list = copy.deepcopy(x_list)
    halo_updater_list = []
    for communicator, y_quantity, x_quantity in zip(communicator_list, y_list, x_list):
        halo_updater_list.append(
            communicator.start_vector_halo_update(
                y_quantity, x_quantity, n_points_update
            )
        )
    for halo_updater in halo_updater_list:
        halo_updater.wait()
    required_times_keys = ("pack", "unpack", "Isend", "Irecv", "wait")
    for communicator in communicator_list:
        with subtests.test(rank=communicator.rank):
            assert isinstance(communicator.timer, pace.util.Timer)
            times = communicator.timer.times
            missing_keys = set(required_times_keys).difference(times.keys())
            assert len(missing_keys) == 0
            extra_keys = set(times.keys()).difference(required_times_keys)
            assert len(extra_keys) == 0
            for key in required_times_keys:
                assert times[key] > 0.0
                assert isinstance(times[key], float)


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


@pytest.mark.parametrize(
    "layout, n_points, n_points_update, n_buffer",
    [((1, 1), 2, "more", 0)],
    indirect=True,
)
def test_halo_updater_stability(
    zeros_quantity_list,
    communicator_list,
    n_points_update,
    n_points,
    numpy,
    subtests,
    boundary_dict,
    ranks_per_tile,
):
    """
    Test that that halo_updater.start()/wait() is consistent through multiple execution.
    Test the internal buffers are re-used properly and re-cached properly.
    """
    BUFFER_CACHE.clear()
    halo_updaters = []
    for communicator, quantity in zip(communicator_list, zeros_quantity_list):
        specification = pace.util.QuantityHaloSpec(
            n_points,
            quantity.data.strides,
            quantity.data.itemsize,
            quantity.data.shape,
            quantity.origin,
            quantity.extent,
            quantity.dims,
            quantity.np,
            quantity.metadata.dtype,
        )
        halo_updater = pace.util.HaloUpdater.from_scalar_specifications(
            comm=communicator,
            numpy_like_module=quantity.np,
            specifications=[specification],
            boundaries=communicator.boundaries.values(),
            tag=0,
        )
        halo_updaters.append(halo_updater)

    # Caches must be created before we run (e.g. cache line != 0
    # and no caches in cache line since they are used)
    assert len(BUFFER_CACHE) == 1
    assert len(next(iter(BUFFER_CACHE.values()))) == 0

    # First run
    for halo_updater in halo_updaters:
        halo_updater.start([quantity])
    for halo_updater in halo_updaters:
        halo_updater.wait()

    # Copy the exchanged buffer and trigger multiple runs
    # The buffer should stay stable since we are exchanging the same information
    exchanged_once_quantity = copy.deepcopy(quantity)
    for halo_updater in halo_updaters:
        halo_updater.start([quantity])
    for halo_updater in halo_updaters:
        halo_updater.wait()
    for halo_updater in halo_updaters:
        halo_updater.start([quantity])
    for halo_updater in halo_updaters:
        halo_updater.wait()
    assert (quantity.data == exchanged_once_quantity.data).all()

    # All caches are still in use
    assert len(BUFFER_CACHE) == 1
    assert len(next(iter(BUFFER_CACHE.values()))) == 0

    # Manually call finalize on the transfomers
    # This should recache all the buffers
    # DSL-816 will refactor that behavior out
    for halo_updater in halo_updaters:
        for transformer in halo_updater._transformers.values():
            transformer.finalize()

    # With the layout constrained we will have
    # 6 (ranks) * 4 (boundaries) * 2 (send&recv) buffers recached.
    assert len(BUFFER_CACHE) == 1
    assert (
        len(next(iter(BUFFER_CACHE.values())))
        == len(communicator_list) * len(communicator.boundaries.values()) * 2
    )
