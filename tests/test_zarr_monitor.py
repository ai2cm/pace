import tempfile
import zarr
import numpy as np
from datetime import datetime, timedelta
import pytest
import xarray as xr
import copy
import fv3util


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


@pytest.fixture(params=["empty", "one_var", "one_var_2d", "two_vars"])
def base_state(request):
    if request.param == 'empty':
        return {}
    elif request.param == 'one_var':
        return {
            'var1': xr.DataArray(
                np.ones([5]),
                dims=('dim1',),
                attrs={'units': 'm'},
            )
        }
    elif request.param == 'one_var':
        return {
            'var1': xr.DataArray(
                np.ones([3]),
                dims=('dim1'),
                attrs={'units': 'm'},
            )
        }
    elif request.param == 'one_var_2d':
        return {
            'var1': xr.DataArray(
                np.ones([3, 3]),
                dims=('dim1', 'dim2'),
                attrs={'units': 'm'},
            )
        }
    elif request.param == 'two_vars':
        return {
            'var1': xr.DataArray(
                np.ones([5]),
                dims=('dim1',),
                attrs={'units': 'm'},
            ),
            'var2': xr.DataArray(
                np.ones([5]),
                dims=('dim1',),
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


def test_monitor_file_store(state_list):
    with tempfile.TemporaryDirectory(suffix='.zarr') as tempdir:
        monitor = fv3util.ZarrMonitor(tempdir)
        for state in state_list:
            monitor.store(state)
        validate_store(state_list, tempdir)


def validate_store(states, filename):
    store = zarr.open_group(filename, mode='r')
    assert set(store.array_keys()) == set(states[0].keys())
    nt = len(states)
    for name, array in store.arrays():
        if name == 'time':
            assert array.shape == (nt,)
        else:
            assert array.shape == (nt, 1) + states[0][name].shape
        if name == 'time':
            target_attrs = {"_ARRAY_DIMENSIONS": ['time']}
        else:
            target_attrs = states[0][name].attrs
            target_attrs["_ARRAY_DIMENSIONS"] = ['time', 'rank'] + list(states[0][name].dims)
        assert dict(array.attrs) == target_attrs
        if name == 'time':
            for i, s in enumerate(states):
                assert array[i] == np.datetime64(s['time'])
        else:
            for i, s in enumerate(states):
                np.testing.assert_array_equal(array[i, 0, :], s[name].values)
