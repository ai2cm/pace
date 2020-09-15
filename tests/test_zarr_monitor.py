import tempfile
import zarr
import cftime
from datetime import timedelta
import pytest
import xarray as xr
import copy
import fv3gfs.util
import logging
from fv3gfs.util.testing import DummyComm


logger = logging.getLogger("test_zarr_monitor")


@pytest.fixture(params=["one_step", "three_steps"])
def n_times(request, fast):
    if request.param == "one_step":
        if fast:
            pytest.skip("running in fast mode")
        else:
            return 1
    elif request.param == "three_steps":
        return 3


@pytest.fixture(
    params=[cftime.DatetimeJulian, cftime.Datetime360Day, cftime.DatetimeNoLeap]
)
def start_time(request):
    date_type = request.param
    return date_type(2010, 1, 1)


@pytest.fixture
def time_step():
    return timedelta(hours=1)


@pytest.fixture
def ny():
    return 4


@pytest.fixture
def nx():
    return 4


@pytest.fixture
def nz():
    return 5


@pytest.fixture
def layout():
    return (1, 1)


@pytest.fixture
def tile_partitioner(layout):
    return fv3gfs.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return fv3gfs.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture(params=["empty", "one_var_2d", "one_var_3d", "two_vars"])
def base_state(request, nz, ny, nx, numpy):
    if request.param == "empty":
        return {}
    elif request.param == "one_var_2d":
        return {
            "var1": fv3gfs.util.Quantity(
                numpy.ones([ny, nx]), dims=("y", "x"), units="m",
            )
        }
    elif request.param == "one_var_3d":
        return {
            "var1": fv3gfs.util.Quantity(
                numpy.ones([nz, ny, nx]), dims=("z", "y", "x"), units="m",
            )
        }
    elif request.param == "two_vars":
        return {
            "var1": fv3gfs.util.Quantity(
                numpy.ones([ny, nx]), dims=("y", "x"), units="m",
            ),
            "var2": fv3gfs.util.Quantity(
                numpy.ones([nz, ny, nx]), dims=("z", "y", "x"), units="degK",
            ),
        }
    else:
        raise NotImplementedError()


@pytest.fixture
def state_list(base_state, n_times, start_time, time_step, numpy):
    state_list = []
    for i in range(n_times):
        new_state = copy.deepcopy(base_state)
        for name in set(new_state.keys()).difference(["time"]):
            new_state[name].view[:] = numpy.random.randn(*new_state[name].extent)
        state_list.append(new_state)
        new_state["time"] = start_time + i * time_step
    return state_list


def test_monitor_file_store(state_list, cube_partitioner, numpy, start_time):
    with tempfile.TemporaryDirectory(suffix=".zarr") as tempdir:
        monitor = fv3gfs.util.ZarrMonitor(tempdir, cube_partitioner)
        for state in state_list:
            monitor.store(state)
        validate_store(state_list, tempdir, numpy, start_time)
        validate_xarray_can_open(tempdir)


def validate_xarray_can_open(dirname):
    # just checking there are no crashes, validate_group checks data
    xr.open_zarr(dirname)


def validate_store(states, filename, numpy, start_time):
    nt = len(states)
    calendar = start_time.calendar

    def assert_no_missing_names(store, state):
        missing_names = set(states[0].keys()).difference(store.array_keys())
        assert len(missing_names) == 0, missing_names

    def validate_array_shape(name, array):
        if name == "time":
            assert array.shape == (nt,)
        else:
            assert array.shape == (nt, 6) + states[0][name].extent

    def validate_array_dimensions_and_attributes(name, array):
        if name == "time":
            target_attrs = {
                "_ARRAY_DIMENSIONS": ["time"],
                "units": "seconds since 2010-01-01 00:00:00",
                "calendar": calendar,
            }
        else:
            target_attrs = states[0][name].attrs
            target_attrs["_ARRAY_DIMENSIONS"] = ["time", "tile"] + list(
                states[0][name].dims
            )
        assert dict(array.attrs) == target_attrs

    def validate_array_values(name, array):
        if name == "time":
            for i, s in enumerate(states):
                value = cftime.num2date(
                    array[i],
                    units="seconds since 2010-01-01 00:00:00",
                    calendar=calendar,
                )
                assert value == s["time"]
        else:
            for i, s in enumerate(states):
                numpy.testing.assert_array_equal(array[i, 0, :], s[name].view[:])

    store = zarr.open_group(filename, mode="r")
    assert_no_missing_names(
        store, states[0]
    )  # states in test all have same names defined
    for name, array in store.arrays():
        validate_array_shape(name, array)
        validate_array_dimensions_and_attributes(name, array)
        validate_array_values(name, array)


@pytest.mark.parametrize("layout", [(1, 1), (1, 2), (2, 2), (4, 4)])
@pytest.mark.parametrize("nt", [1, 3])
@pytest.mark.parametrize(
    "shape, ny_rank_add, nx_rank_add, dims",
    [
        ((5, 4, 4), 0, 0, ("z", "y", "x")),
        ((5, 4, 4), 1, 1, ("z", "y_interface", "x_interface")),
        ((5, 4, 4), 0, 1, ("z", "y", "x_interface")),
    ],
)
def test_monitor_file_store_multi_rank_state(
    layout, nt, tmpdir_factory, shape, ny_rank_add, nx_rank_add, dims, numpy
):
    units = "m"
    tmpdir = tmpdir_factory.mktemp("data.zarr")
    nz, ny, nx = shape
    ny_rank = int(ny / layout[0] + ny_rank_add)
    nx_rank = int(nx / layout[1] + nx_rank_add)
    grid = fv3gfs.util.TilePartitioner(layout)
    time = cftime.DatetimeJulian(2010, 6, 20, 6, 0, 0)
    timestep = timedelta(hours=1)
    total_ranks = 6 * layout[0] * layout[1]
    partitioner = fv3gfs.util.CubedSpherePartitioner(grid)
    store = zarr.storage.DirectoryStore(tmpdir)
    shared_buffer = {}
    monitor_list = []
    for rank in range(total_ranks):
        monitor_list.append(
            fv3gfs.util.ZarrMonitor(
                store,
                partitioner,
                "w",
                mpi_comm=DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
            )
        )
    for i_t in range(nt):
        for rank in range(total_ranks):
            state = {
                "time": time + i_t * timestep,
                "var1": fv3gfs.util.Quantity(
                    numpy.ones([nz, ny_rank, nx_rank]), dims=dims, units=units,
                ),
            }
            monitor_list[rank].store(state)
    group = zarr.hierarchy.open_group(store=store, mode="r")
    assert "var1" in group
    assert group["var1"].shape == (nt, 6, nz, ny + ny_rank_add, nx + nx_rank_add)
    numpy.testing.assert_array_equal(group["var1"], 1.0)


@pytest.mark.parametrize(
    "layout, tile_array_shape, array_dims, target",
    [
        pytest.param(
            (1, 1),
            (7, 6, 6),
            [fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (7, 6, 6),
            id="single_chunk_tile_3d",
        ),
        pytest.param(
            (1, 1),
            (6, 6),
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (6, 6),
            id="single_chunk_tile_2d",
        ),
        pytest.param(
            (1, 1), (6,), [fv3gfs.util.Y_DIM], (6,), id="single_chunk_tile_1d"
        ),
        pytest.param(
            (1, 1),
            (7, 6, 6),
            [
                fv3gfs.util.Z_DIM,
                fv3gfs.util.Y_INTERFACE_DIM,
                fv3gfs.util.X_INTERFACE_DIM,
            ],
            (7, 5, 5),
            id="single_chunk_tile_3d_interfaces",
        ),
        pytest.param(
            (2, 2),
            (7, 6, 6),
            [fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (7, 3, 3),
            id="2_by_2_tile_3d",
        ),
        pytest.param(
            (2, 2),
            (6, 16, 6),
            [fv3gfs.util.Y_DIM, fv3gfs.util.Z_DIM, fv3gfs.util.X_DIM],
            (3, 16, 3),
            id="2_by_2_tile_3d_odd_dim_order",
        ),
        pytest.param(
            (2, 2),
            (7, 7, 7),
            [
                fv3gfs.util.Z_DIM,
                fv3gfs.util.Y_INTERFACE_DIM,
                fv3gfs.util.X_INTERFACE_DIM,
            ],
            (7, 3, 3),
            id="2_by_2_tile_3d_interfaces",
        ),
    ],
)
def test_array_chunks(layout, tile_array_shape, array_dims, target):
    result = fv3gfs.util.zarr_monitor.array_chunks(layout, tile_array_shape, array_dims)
    assert result == target


def _assert_no_nulls(dataset: xr.Dataset):
    number_of_null = dataset["var"].isnull().sum().item()
    total_size = dataset["var"].size

    assert (
        number_of_null == 0
    ), f"Number of nulls {number_of_null}. Size of data {total_size}"


@pytest.mark.parametrize("mask_and_scale", [True, False])
def test_open_zarr_without_nans(cube_partitioner, numpy, backend, mask_and_scale):

    store = {}

    # initialize store
    monitor = fv3gfs.util.ZarrMonitor(store, cube_partitioner)
    zero_quantity = fv3gfs.util.Quantity(
        numpy.zeros([10, 10]), dims=("y", "x"), units="m"
    )
    monitor.store({"var": zero_quantity})

    # open w/o dask using chunks=None
    dataset = xr.open_zarr(store, chunks=None, mask_and_scale=mask_and_scale)
    _assert_no_nulls(dataset.sel(tile=0))


def test_values_preserved(cube_partitioner, numpy):
    dims = ("y", "x")
    units = "m"

    store = {}

    # initialize store
    monitor = fv3gfs.util.ZarrMonitor(store, cube_partitioner)
    quantity = fv3gfs.util.Quantity(
        numpy.random.uniform(size=(10, 10)), dims=dims, units=units
    )
    monitor.store({"var": quantity})

    # open w/o dask using chunks=None
    dataset = xr.open_zarr(store, chunks=None)
    numpy.testing.assert_array_almost_equal(
        dataset["var"][0, 0, :, :].values, quantity.data
    )
    assert dataset["var"].shape[:2] == (1, 6)
    assert dataset["var"].attrs["units"] == units
    assert dataset["var"].dims[2:] == dims


@pytest.fixture
def state_list_with_inconsistent_calendars(base_state, numpy):
    state_list = []
    state_times = [cftime.DatetimeNoLeap(2000, 1, 1), cftime.Datetime360Day(2000, 1, 2)]
    for i in range(2):
        new_state = copy.deepcopy(base_state)
        for name in set(new_state.keys()).difference(["time"]):
            new_state[name].view[:] = numpy.random.randn(*new_state[name].extent)
        state_list.append(new_state)
        new_state["time"] = state_times[i]
    return state_list


def test_monitor_file_store_inconsistent_calendars(
    state_list_with_inconsistent_calendars, cube_partitioner, numpy
):
    with tempfile.TemporaryDirectory(suffix=".zarr") as tempdir:
        monitor = fv3gfs.util.ZarrMonitor(tempdir, cube_partitioner)
        initial_state, final_state = state_list_with_inconsistent_calendars
        monitor.store(initial_state)
        with pytest.raises(ValueError, match="Calendar type"):
            monitor.store(final_state)
