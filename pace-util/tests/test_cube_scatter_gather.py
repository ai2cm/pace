import copy
import datetime

import pytest

import pace.util


try:
    import gt4py
except ImportError:
    gt4py = None


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request):
    return request.param


@pytest.fixture(params=[0, 1, 3])
def n_rank_halo(request):
    return request.param


@pytest.fixture(params=[0, 3])
def n_tile_halo(request):
    return request.param


@pytest.fixture(params=["x,y", "y,x", "xi,y", "x,y,z", "z,y,x", "y,z,x"])
def dims(request, fast):
    if request.param == "x,y":
        return [pace.util.X_DIM, pace.util.Y_DIM]
    elif request.param == "y,x":
        if fast:
            pytest.skip("running in fast mode")
        else:
            return [pace.util.Y_DIM, pace.util.X_DIM]
    elif request.param == "xi,y":
        return [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM]
    elif request.param == "x,y,z":
        return [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM]
    elif request.param == "z,y,x":
        if fast:
            pytest.skip("running in fast mode")
        else:
            return [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM]
    elif request.param == "y,z,x":
        return [pace.util.Y_DIM, pace.util.Z_DIM, pace.util.X_DIM]
    else:
        raise NotImplementedError()


@pytest.fixture
def units():
    return "m/s"


@pytest.fixture
def time():
    return datetime.datetime(2000, 1, 1)


def assert_quantity_equals(result, reference):
    assert result.dims == reference.dims
    assert result.units == reference.units
    assert result.extent == reference.extent
    assert isinstance(result.data, type(reference.data))
    reference.np.testing.assert_array_equal(result.view[:], reference.view[:])


@pytest.fixture()
def dim_lengths(layout):
    return {
        pace.util.X_DIM: 2 * layout[1],
        pace.util.X_INTERFACE_DIM: 2 * layout[1] + 1,
        pace.util.Y_DIM: 2 * layout[0],
        pace.util.Y_INTERFACE_DIM: 2 * layout[0] + 1,
        pace.util.Z_DIM: 3,
        pace.util.Z_INTERFACE_DIM: 4,
    }


@pytest.fixture()
def communicator_list(layout):
    total_ranks = 6 * layout[0] * layout[1]
    shared_buffer = {}
    return_list = []
    for rank in range(total_ranks):
        return_list.append(
            pace.util.CubedSphereCommunicator(
                pace.util.testing.DummyComm(rank, total_ranks, shared_buffer),
                pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout)),
                timer=pace.util.Timer(),
            )
        )
    return return_list


@pytest.fixture
def tile_extent(dims, dim_lengths):
    return_list = []
    for dim in dims:
        return_list.append(dim_lengths[dim])
    return tuple(return_list)


@pytest.fixture
def cube_quantity(dims, units, dim_lengths, tile_extent, n_tile_halo, numpy):
    return get_cube_quantity(dims, units, dim_lengths, tile_extent, n_tile_halo, numpy)


@pytest.fixture
def scattered_quantities(cube_quantity, layout, n_rank_halo, numpy):
    tile_ranks = layout[0] * layout[1]
    return_list = []
    partitioner = pace.util.TilePartitioner(layout)
    for i_tile in range(6):
        for rank in range(tile_ranks):
            # partitioner is tested in other tests, here we assume it works
            subtile_slice = partitioner.subtile_slice(
                rank=rank,
                global_dims=cube_quantity.dims[1:],
                global_extent=cube_quantity.extent[1:],
                overlap=True,
            )
            subtile_view = cube_quantity.view[(i_tile,) + subtile_slice]
            subtile_quantity = get_quantity(
                cube_quantity.dims[1:],
                cube_quantity.units,
                subtile_view.shape,
                n_rank_halo,
                numpy,
            )
            subtile_quantity.view[:] = subtile_view
            return_list.append(subtile_quantity)
    return return_list


def get_cube_quantity(dims, units, dim_lengths, tile_extent, n_halo, numpy):
    extent = [6] + [dim_lengths[dim] for dim in dims]
    dims = [pace.util.TILE_DIM] + dims
    quantity = get_quantity(dims, units, extent, n_halo, numpy)
    quantity.view[:] = numpy.random.randn(*quantity.extent)
    return quantity


def get_quantity(dims, units, extent, n_halo, numpy):
    shape = list(copy.deepcopy(extent))
    origin = [0 for dim in dims]
    for i, dim in enumerate(dims):
        if dim in pace.util.HORIZONTAL_DIMS:
            origin[i] += n_halo
            shape[i] += 2 * n_halo
    return pace.util.Quantity(
        numpy.zeros(shape),
        dims,
        units,
        origin=tuple(origin),
        extent=tuple(extent),
    )


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
@pytest.mark.parametrize("dims, layout", [["x,y,z", (2, 2)]], indirect=True)
def test_gathered_quantity_has_storage(
    scattered_quantities, communicator_list, time, backend
):
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        result = communicator.gather(send_quantity=rank_quantity)
        if communicator.rank == 0:
            assert isinstance(result.storage, gt4py.storage.storage.Storage)
        else:
            assert result is None


@pytest.mark.parametrize("backend", ["gt4py_numpy", "gt4py_cupy"], indirect=True)
@pytest.mark.parametrize("dims, layout", [["x,y,z", (2, 2)]], indirect=True)
def test_scattered_quantity_has_storage(
    cube_quantity, communicator_list, time, backend
):
    result_list = []
    for communicator in communicator_list:
        if communicator.rank == 0:
            result_list.append(communicator.scatter(send_quantity=cube_quantity))
        else:
            result_list.append(communicator.scatter())
    for rank, result in enumerate(result_list):
        assert isinstance(result.storage, gt4py.storage.storage.Storage)


def test_cube_gather_state(
    cube_quantity, scattered_quantities, communicator_list, time, backend
):
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        state = {"time": time, "air_temperature": rank_quantity}
        out = communicator.gather_state(send_state=state)
        if communicator.rank == 0:
            result_state = out
        else:
            assert out is None
    assert result_state["time"] == time
    result = result_state["air_temperature"]
    assert_quantity_equals(result, cube_quantity)


def test_cube_gather_state_with_recv_state(
    cube_quantity, scattered_quantities, communicator_list, time
):
    recv_state = {"time": time, "air_temperature": copy.deepcopy(cube_quantity)}
    recv_state["air_temperature"].data[:] = -1
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        state = {"time": time, "air_temperature": rank_quantity}
        if communicator.rank == 0:
            communicator.gather_state(send_state=state, recv_state=recv_state)
        else:
            communicator.gather_state(send_state=state)
    assert recv_state["time"] == time
    result = recv_state["air_temperature"]
    assert_quantity_equals(result, cube_quantity)


def test_cube_gather_no_recv_quantity(
    cube_quantity, scattered_quantities, communicator_list
):
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        result = communicator.gather(send_quantity=rank_quantity)
        if communicator.rank != 0:
            assert result is None
    assert_quantity_equals(result, cube_quantity)


def test_cube_scatter_no_recv_quantity(
    cube_quantity, scattered_quantities, communicator_list
):
    result_list = []
    for communicator in communicator_list:
        if communicator.rank == 0:
            result_list.append(communicator.scatter(send_quantity=cube_quantity))
        else:
            result_list.append(communicator.scatter())
    for rank, (result, scattered) in enumerate(zip(result_list, scattered_quantities)):
        assert_quantity_equals(result, scattered)


def test_cube_scatter_with_recv_quantity(
    cube_quantity, scattered_quantities, communicator_list
):
    recv_quantities = copy.deepcopy(scattered_quantities)
    for q in recv_quantities:
        q.data[:] = 0.0
    for recv, communicator in zip(recv_quantities, communicator_list):
        if communicator.rank == 0:
            result = communicator.scatter(
                send_quantity=cube_quantity, recv_quantity=recv
            )
        else:
            result = communicator.scatter(recv_quantity=recv)
        assert result is recv
    for rank, (result, scattered) in enumerate(
        zip(recv_quantities, scattered_quantities)
    ):
        assert_quantity_equals(result, scattered)


def test_cube_gather_with_recv_quantity(
    cube_quantity, scattered_quantities, communicator_list
):
    recv_quantity = copy.deepcopy(cube_quantity)
    recv_quantity.data[:] = -1
    for communicator, rank_quantity in reversed(
        list(zip(communicator_list, scattered_quantities))
    ):
        if communicator.rank == 0:
            result = communicator.gather(
                send_quantity=rank_quantity, recv_quantity=recv_quantity
            )
        else:
            result = communicator.gather(send_quantity=rank_quantity)
            assert result is None
    assert_quantity_equals(recv_quantity, cube_quantity)


def test_cube_scatter_state(
    cube_quantity, scattered_quantities, communicator_list, time
):
    state = {"time": time, "air_temperature": cube_quantity}
    result_list = []
    for communicator in communicator_list:
        if communicator.rank == 0:
            result_list.append(communicator.scatter_state(send_state=state))
        else:
            result_list.append(communicator.scatter_state())
    for result_state, scattered in zip(result_list, scattered_quantities):
        assert result_state["time"] == time
        result = result_state["air_temperature"]
        assert_quantity_equals(result, scattered)


def test_cube_scatter_state_with_recv_state(
    cube_quantity, scattered_quantities, communicator_list, time
):
    tile_state = {"time": time, "air_temperature": cube_quantity}
    recv_quantities = copy.deepcopy(scattered_quantities)
    for q in recv_quantities:
        q.data[:] = 0.0
    for recv, communicator in zip(recv_quantities, communicator_list):
        state = {
            "time": time - datetime.timedelta(hours=1),
            "air_temperature": recv,
        }
        if communicator.rank == 0:
            result = communicator.scatter_state(send_state=tile_state, recv_state=state)
        else:
            result = communicator.scatter_state(recv_state=state)
        assert result["time"] == time
        assert result["air_temperature"] is recv
    for rank, (result, scattered) in enumerate(
        zip(recv_quantities, scattered_quantities)
    ):
        assert_quantity_equals(result, scattered)
