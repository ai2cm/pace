import tempfile
import zarr
import numpy as np
from datetime import datetime, timedelta
import pytest
import xarray as xr
import copy
import fv3util
import logging
from utils import DummyComm


logger = logging.getLogger('test_zarr_monitor')


@pytest.fixture(params=["one_step", "three_steps"])
def n_times(request):
    if request.param == "one_step":
        return 1
    elif request.param == "three_steps":
        return 3


@pytest.fixture
def start_time():
    return datetime(2010, 1, 1)


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


@pytest.fixture(params=["empty", "one_var_2d", "one_var_3d", "two_vars"])
def base_state(request, nz, ny, nx):
    if request.param == 'empty':
        return {}
    elif request.param == 'one_var_2d':
        return {
            'var1': xr.DataArray(
                np.ones([ny, nx]),
                dims=('y', 'x'),
                attrs={'units': 'm'},
            )
        }
    elif request.param == 'one_var_3d':
        return {
            'var1': xr.DataArray(
                np.ones([nz, ny, nx]),
                dims=('z', 'y', 'x'),
                attrs={'units': 'm'},
            )
        }
    elif request.param == 'two_vars':
        return {
            'var1': xr.DataArray(
                np.ones([ny, nx]),
                dims=('y', 'x'),
                attrs={'units': 'm'},
            ),
            'var2': xr.DataArray(
                np.ones([nz, ny, nx]),
                dims=('z', 'y', 'x'),
                attrs={'units': 'm'},
            )
        }
    else:
        raise NotImplementedError()


@pytest.fixture
def state_list(base_state, n_times, start_time, time_step):
    state_list = []
    for i in range(n_times):
        new_state = copy.deepcopy(base_state)
        for name in set(new_state.keys()).difference(['time']):
            new_state[name].values = np.random.randn(*new_state[name].shape)
        state_list.append(new_state)
        new_state["time"] = start_time + i * time_step
    return state_list


def test_monitor_file_store(state_list, nz, ny, nx):
    domain = fv3util.CubedSpherePartitioner(ny=ny, nx=nx, layout=(1, 1))
    with tempfile.TemporaryDirectory(suffix='.zarr') as tempdir:
        monitor = fv3util.ZarrMonitor(tempdir, domain)
        for state in state_list:
            monitor.store(state)
        validate_group(state_list, tempdir)
        validate_xarray_can_open(tempdir)


def validate_xarray_can_open(dirname):
    # just checking there are no crashes, validate_group checks data
    xr.open_zarr(dirname)


def validate_group(states, dirname):
    nt = len(states)

    def assert_no_missing_names(group, state):
        missing_names = set(states[0].keys()).difference(group.array_keys())
        assert len(missing_names) == 0, missing_names

    def validate_array_shape(name, array):
        if name == 'time':
            assert array.shape == (nt,)
        else:
            assert array.shape == (nt, 6) + states[0][name].shape

    def validate_array_dimensions_and_attributes(name, array):
        if name == 'time':
            target_attrs = {"_ARRAY_DIMENSIONS": ['time']}
        else:
            target_attrs = states[0][name].attrs
            target_attrs["_ARRAY_DIMENSIONS"] = ['time', 'tile'] + list(states[0][name].dims)
        assert dict(array.attrs) == target_attrs

    def validate_array_values(name, array):
        if name == 'time':
            for i, s in enumerate(states):
                assert array[i] == np.datetime64(s['time'])
        else:
            for i, s in enumerate(states):
                np.testing.assert_array_equal(array[i, 0, :], s[name].values)

    group = zarr.open_group(dirname, mode='r')
    assert_no_missing_names(group, states[0])  # states in test all have same names defined
    for name, array in group.arrays():
        validate_array_shape(name, array)
        validate_array_dimensions_and_attributes(name, array)
        validate_array_values(name, array)


@pytest.mark.parametrize(
    'layout', [(1, 1), (1, 2), (2, 2), (4, 4)]
)
@pytest.mark.parametrize(
    'nt', [1, 3]
)
@pytest.mark.parametrize(
    'shape, ny_rank_add, nx_rank_add, dims', [
        ((5, 4, 4), 0, 0, ('z', 'y', 'x')),
        ((5, 4, 4), 1, 1, ('z', 'y_interface', 'x_interface')),
        ((5, 4, 4), 0, 1, ('z', 'y', 'x_interface')),
    ]
)
def test_monitor_file_store_multi_rank_state(
        layout, nt, tmpdir_factory, shape, ny_rank_add, nx_rank_add, dims):
    tmpdir = tmpdir_factory.mktemp("data.zarr")
    nz, ny, nx = shape
    time = datetime(2010, 6, 20, 6, 0, 0)
    timestep = timedelta(hours=1)
    total_ranks = 6 * layout[0] * layout[1]
    partitioner = fv3util.CubedSpherePartitioner(ny=ny, nx=nx, layout=layout)
    ny_rank = partitioner.ny_rank + ny_rank_add
    nx_rank = partitioner.nx_rank + nx_rank_add
    store = zarr.storage.DirectoryStore(tmpdir)
    shared_buffer = {}
    monitor_list = []
    for rank in range(total_ranks):
        monitor_list.append(fv3util.ZarrMonitor(
            store,
            partitioner,
            "w",
            mpi_comm=DummyComm(rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer)
        ))
    for i_t in range(nt):
        for rank in range(total_ranks):
            state = {
                'time': time + i_t * timestep,
                'var1': xr.DataArray(
                    np.ones([nz, ny_rank, nx_rank]),
                    dims=dims,
                    attrs={'units': 'm'}
                )
            }
            monitor_list[rank].store(state)
    group = zarr.hierarchy.open_group(store=store, mode='r')
    assert 'var1' in group
    assert group['var1'].shape == (nt, 6, nz, ny + ny_rank_add, nx + nx_rank_add)
    np.testing.assert_array_equal(group['var1'], 1.0)

