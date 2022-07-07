import copy
from datetime import timedelta

import pytest

import pace.util


@pytest.fixture(params=["empty", "one_var", "two_vars"])
def state(request, numpy):
    if request.param == "empty":
        return {}
    elif request.param == "one_var":
        return {
            "var1": pace.util.Quantity(
                numpy.ones([5]),
                dims=["dim1"],
                units="m",
            )
        }
    elif request.param == "two_vars":
        return {
            "var1": pace.util.Quantity(
                numpy.ones([5]),
                dims=["dim1"],
                units="m",
            ),
            "var2": pace.util.Quantity(
                numpy.ones([5]),
                dims=["dim_2"],
                units="m",
            ),
        }
    else:
        raise NotImplementedError()


@pytest.fixture(params=["equal", "plus_one", "extra_var"])
def reference_difference(request):
    return request.param


@pytest.fixture
def reference_state(reference_difference, state, numpy):
    if reference_difference == "equal":
        reference_state = copy.deepcopy(state)
    elif reference_difference == "extra_var":
        reference_state = copy.deepcopy(state)
        reference_state["extra_var"] = pace.util.Quantity(
            numpy.ones([5]),
            dims=["dim1"],
            units="m",
        )
    elif reference_difference == "plus_one":
        reference_state = copy.deepcopy(state)
        for array in reference_state.values():
            array.data[:] += 1.0
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
            array.data[:] += 1.0 / multiple_of_timestep
    else:
        raise NotImplementedError()
    return final_state


@pytest.fixture
def nudging_tendencies(reference_difference, state, nudging_timescales):
    if reference_difference in ("equal", "extra_var"):
        tendencies = copy.deepcopy(state)
        for array in tendencies.values():
            array.data[:] = 0.0
            array.metadata.units = array.units + " s^-1"
    elif reference_difference == "plus_one":
        tendencies = copy.deepcopy(state)
        for name, array in tendencies.items():
            array.data[:] = 1.0 / nudging_timescales[name].total_seconds()
            array.metadata.units = array.units + " s^-1"
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
    state,
    reference_state,
    nudging_timescales,
    timestep,
    final_state,
    nudging_tendencies,
    numpy,
):
    result = pace.util.apply_nudging(
        state, reference_state, nudging_timescales, timestep
    )
    for name, tendency in nudging_tendencies.items():
        numpy.testing.assert_array_equal(result[name].data, tendency.data)
        assert result[name].dims == tendency.dims
        assert result[name].units == tendency.units
    for name, reference_array in final_state.items():
        numpy.testing.assert_array_equal(state[name].data, reference_array.data)
        assert state[name].dims == reference_array.dims
        assert state[name].units == reference_array.units


def test_get_nudging_tendencies_equals(
    state, reference_state, nudging_timescales, nudging_tendencies, numpy
):
    result = pace.util.get_nudging_tendencies(
        state, reference_state, nudging_timescales
    )
    for name, tendency in nudging_tendencies.items():
        numpy.testing.assert_array_equal(result[name].data, tendency.data)
        assert result[name].dims == tendency.dims
        assert result[name].attrs["units"] == tendency.attrs["units"]


def test_get_nudging_tendencies_half_timescale(
    state, reference_state, nudging_timescales, nudging_tendencies, numpy
):
    for name, timescale in nudging_timescales.items():
        nudging_timescales[name] = timedelta(seconds=0.5 * timescale.total_seconds())
    result = pace.util.get_nudging_tendencies(
        state, reference_state, nudging_timescales
    )
    for name, tendency in nudging_tendencies.items():
        numpy.testing.assert_array_equal(result[name].data, 2.0 * tendency.data)
        assert result[name].dims == tendency.dims
        assert result[name].attrs["units"] == tendency.attrs["units"]
