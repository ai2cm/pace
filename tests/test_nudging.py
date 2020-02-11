import fv3util
from datetime import timedelta
import pytest
import xarray as xr
import numpy as np
import copy


@pytest.fixture(params=["empty", "one_var", "two_vars"])
def state(request):
    if request.param == 'empty':
        return {}
    elif request.param == 'one_var':
        return {
            'var1': xr.DataArray(
                np.ones([5]),
                dims=['dim1'],
                attrs={'units': 'm'},
            )
        }
    elif request.param == 'two_vars':
        return {
            'var1': xr.DataArray(
                np.ones([5]),
                dims=['dim1'],
                attrs={'units': 'm'},
            ),
            'var2': xr.DataArray(
                np.ones([5]),
                dims=['dim1'],
                attrs={'units': 'm'},
            )
        }
    else:
        raise NotImplementedError()


@pytest.fixture(params=["equal", "plus_one", "extra_var"])
def reference_difference(request):
    return request.param


@pytest.fixture
def reference_state(reference_difference, state):
    if reference_difference == "equal":
        reference_state = copy.deepcopy(state)
    elif reference_difference == "extra_var":
        reference_state = copy.deepcopy(state)
        reference_state['extra_var'] = xr.DataArray(
            np.ones([5]),
            dims=['dim1'],
            attrs={'units': 'm'},
        )
    elif reference_difference == "plus_one":
        reference_state = copy.deepcopy(state)
        for array in reference_state.values():
            array += 1.0
    else:
        raise NotImplementedError()
    return reference_state


@pytest.fixture(params=[0.1, 0.5, 1.0])
def multiple_of_timestep(request):
    return request.param


@pytest.fixture
def nudging_timescales(state, timestep, multiple_of_timestep):
    return_dict = {}
    for name in state.keys():
        return_dict[name] = timedelta(
            seconds=multiple_of_timestep * timestep.total_seconds()
        )
    return return_dict


@pytest.fixture
def final_state(reference_difference, state, multiple_of_timestep):
    if reference_difference in ("equal", "extra_var"):
        final_state = copy.deepcopy(state)
    elif reference_difference == "plus_one":
        final_state = copy.deepcopy(state)
        for name, array in final_state.items():
            array.values += 1.0 / multiple_of_timestep
    else:
        raise NotImplementedError()
    return final_state


@pytest.fixture
def nudging_tendencies(reference_difference, state, nudging_timescales):
    if reference_difference in ("equal", "extra_var"):
        tendencies = copy.deepcopy(state)
        for array in tendencies.values():
            array.values[:] = 0.
            array.attrs['units'] = array.attrs['units'] + ' s^-1'
    elif reference_difference == "plus_one":
        tendencies = copy.deepcopy(state)
        for name, array in tendencies.items():
            array.values[:] = 1.0 / nudging_timescales[name].total_seconds()
            array.attrs['units'] = array.attrs['units'] + ' s^-1'
    else:
        raise NotImplementedError()
    return tendencies


@pytest.fixture(params=["one_second", "one_hour", "30_seconds"])
def timestep(request):
    if request.param == "one_second":
        return timedelta(seconds=1)
    if request.param == "one_hour":
        return timedelta(hours=1)
    if request.param == "30_seconds":
        return timedelta(seconds=30)
    else:
        raise NotImplementedError


def test_apply_nudging_equals(
        state, reference_state, nudging_timescales, timestep,
        final_state, nudging_tendencies):
    result = fv3util.apply_nudging(
        state, reference_state, nudging_timescales, timestep)
    for name, tendency in nudging_tendencies.items():
        xr.testing.assert_equal(result[name], tendency)
        assert result[name].attrs['units'] == tendency.attrs['units']
    for name, reference_array in final_state.items():
        xr.testing.assert_equal(state[name], reference_array)
        assert state[name].attrs['units'] == reference_array.attrs['units']


def test_get_nudging_tendencies_equals(
        state, reference_state, nudging_timescales, nudging_tendencies):
    result = fv3util.get_nudging_tendencies(
        state, reference_state, nudging_timescales)
    for name, tendency in nudging_tendencies.items():
        xr.testing.assert_equal(result[name], tendency)
        assert result[name].attrs['units'] == tendency.attrs['units']


def test_get_nudging_tendencies_half_timescale(
        state, reference_state, nudging_timescales, nudging_tendencies):
    for name, timescale in nudging_timescales.items():
        nudging_timescales[name] = timedelta(seconds=0.5 * timescale.total_seconds())
    result = fv3util.get_nudging_tendencies(
        state, reference_state, nudging_timescales)
    for name, tendency in nudging_tendencies.items():
        xr.testing.assert_equal(result[name], 2.0 * tendency)
        assert result[name].attrs['units'] == tendency.attrs['units']

