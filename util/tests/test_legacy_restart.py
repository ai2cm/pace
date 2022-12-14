import os
import tempfile

import cftime


try:
    import xarray as xr
except ModuleNotFoundError:
    xr = None
import numpy as np
import pytest

import pace.util
import pace.util._legacy_restart
from pace.util.testing import DummyComm


requires_xarray = pytest.mark.skipif(xr is None, reason="xarray is not installed")

TEST_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(TEST_DIRECTORY, "data")


@pytest.fixture(params=[(1, 1)])
def layout(request):
    return request.param


@requires_xarray
def get_c12_restart_state_list(layout, only_names, tracer_properties):
    total_ranks = 6 * layout[0] * layout[1]
    shared_buffer = {}
    communicator_list = []
    for rank in range(total_ranks):
        communicator = pace.util.CubedSphereCommunicator(
            DummyComm(rank, total_ranks, shared_buffer),
            pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout)),
        )
        communicator_list.append(communicator)
    state_list = []
    for communicator in communicator_list:
        state_list.append(
            pace.util.open_restart(
                os.path.join(DATA_DIRECTORY, "c12_restart"),
                communicator,
                only_names=only_names,
                tracer_properties=tracer_properties,
            )
        )
    return state_list


@pytest.mark.parametrize("layout", [(1, 1), (3, 3)])
@pytest.mark.cpu_only
@requires_xarray
def test_open_c12_restart(layout):
    tracer_properties = {}
    only_names = None
    c12_restart_state_list = get_c12_restart_state_list(
        layout, only_names, tracer_properties
    )
    # C12 has 12 gridcells along each tile side, we divide this across processors
    ny = 12 / layout[0]
    nx = 12 / layout[1]
    for state in c12_restart_state_list:
        assert "time" in state.keys()
        assert len(state.keys()) == 63
        for name, value in state.items():
            if name == "time":
                assert isinstance(value, cftime.DatetimeJulian)
            else:
                assert isinstance(value, pace.util.Quantity)
                assert np.sum(np.isnan(value.view[:])) == 0
                for dim, extent in zip(value.dims, value.extent):
                    if dim == pace.util.X_DIM:
                        assert extent == nx
                    elif dim == pace.util.X_INTERFACE_DIM:
                        assert extent == nx + 1
                    elif dim == pace.util.Y_DIM:
                        assert extent == ny
                    elif dim == pace.util.Y_INTERFACE_DIM:
                        assert extent == ny + 1


@pytest.mark.parametrize(
    "tracer_properties",
    [
        {
            "specific_humidity": {
                "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
                "units": "kg/kg",
                "restart_name": "sphum",
            },
        },
        {
            "specific_humidity_by_another_name": {
                "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
                "units": "kg/kg",
                "restart_name": "sphum",
            },
        },
        {
            "specific_humidity": {
                "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
                "units": "kg/kg",
                "restart_name": "sphum",
            },
        },
        {
            "specific_humidity": {
                "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
                "units": "kg/kg",
                "restart_name": "sphum",
            },
            "snow_water_mixing_ratio": {
                "dims": [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
                "units": "kg/kg",
                "restart_name": "snowwat",
            },
        },
    ],
)
@requires_xarray
@pytest.mark.cpu_only
def test_open_c12_restart_tracer_properties(layout, tracer_properties):
    only_names = None
    c12_restart_state_list = get_c12_restart_state_list(
        layout, only_names, tracer_properties
    )
    for state in c12_restart_state_list:
        for name, properties in tracer_properties.items():
            assert name in state.keys()
            assert state[name].dims == tuple(properties["dims"])
            assert state[name].attrs["units"] == properties["units"]
            assert properties["restart_name"] not in state


@pytest.mark.parametrize("layout", [(1, 1), (3, 3)])
@pytest.mark.cpu_only
@requires_xarray
def test_open_c12_restart_empty_to_state_without_crashing(layout):
    total_ranks = 6 * layout[0] * layout[1]
    ny = 12 / layout[0]
    nx = 12 / layout[1]
    shared_buffer = {}
    communicator_list = []
    for rank in range(total_ranks):
        communicator = pace.util.CubedSphereCommunicator(
            DummyComm(rank, total_ranks, shared_buffer),
            pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout)),
        )
        communicator_list.append(communicator)
    state_list = []
    for communicator in communicator_list:
        state_list.append({})
        pace.util.open_restart(
            os.path.join(DATA_DIRECTORY, "c12_restart"),
            communicator,
            to_state=state_list[-1],
        )
    for state in state_list:
        assert "time" in state.keys()
        assert len(state.keys()) == 63
        for name, value in state.items():
            if name == "time":
                assert isinstance(value, cftime.DatetimeJulian)
            else:
                assert isinstance(value, pace.util.Quantity)
                assert np.sum(np.isnan(value.view[:])) == 0
                for dim, extent in zip(value.dims, value.extent):
                    if dim == pace.util.X_DIM:
                        assert extent == nx
                    elif dim == pace.util.X_INTERFACE_DIM:
                        assert extent == nx + 1
                    elif dim == pace.util.Y_DIM:
                        assert extent == ny
                    elif dim == pace.util.Y_INTERFACE_DIM:
                        assert extent == ny + 1


@pytest.mark.parametrize("layout", [(1, 1), (3, 3)])
@pytest.mark.cpu_only
@requires_xarray
def test_open_c12_restart_to_allocated_state_without_crashing(layout):
    total_ranks = 6 * layout[0] * layout[1]
    ny = 12 / layout[0]
    nx = 12 / layout[1]
    shared_buffer = {}
    communicator_list = []
    for rank in range(total_ranks):
        communicator = pace.util.CubedSphereCommunicator(
            DummyComm(rank, total_ranks, shared_buffer),
            pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout)),
        )
        communicator_list.append(communicator)
    state_list = []
    for communicator in communicator_list:
        state_list.append(
            pace.util.open_restart(
                os.path.join(DATA_DIRECTORY, "c12_restart"), communicator
            )
        )
    for state in state_list:
        for name, value in state.items():
            if name != "time":
                value.view[:] = np.nan
    for state, communicator in zip(state_list, communicator_list):
        pace.util.open_restart(
            os.path.join(DATA_DIRECTORY, "c12_restart"), communicator, to_state=state
        )

    for state in state_list:
        assert "time" in state.keys()
        assert len(state.keys()) == 63
        for name, value in state.items():
            if name == "time":
                assert isinstance(value, cftime.DatetimeJulian)
            else:
                assert isinstance(value, pace.util.Quantity)
                assert np.sum(np.isnan(value.view[:])) == 0
                for dim, extent in zip(value.dims, value.extent):
                    if dim == pace.util.X_DIM:
                        assert extent == nx
                    elif dim == pace.util.X_INTERFACE_DIM:
                        assert extent == nx + 1
                    elif dim == pace.util.Y_DIM:
                        assert extent == ny
                    elif dim == pace.util.Y_INTERFACE_DIM:
                        assert extent == ny + 1


@pytest.fixture(
    params=[
        ("coupler_julian.res", cftime.DatetimeJulian),
        ("coupler_thirty_day.res", cftime.Datetime360Day),
        ("coupler_noleap.res", cftime.DatetimeNoLeap),
    ],
    ids=["julian", "thirty_day", "noleap"],
)
def coupler_res_file_and_time(request):
    file, expected_date_type = request.param
    return (
        os.path.join(DATA_DIRECTORY, file),
        expected_date_type(2016, 8, 3),
    )


@pytest.mark.cpu_only
def test_get_current_date_from_coupler_res(coupler_res_file_and_time):
    filename, current_time = coupler_res_file_and_time
    with open(filename, "r") as f:
        result = pace.util.io.get_current_date_from_coupler_res(f)
    assert result == current_time


@pytest.fixture
def data_array():
    return xr.DataArray(np.random.randn(2, 3), dims=["x", "y"], attrs={"units": "m"})


@pytest.fixture(params=["empty", "1_dim", "2_dims"])
def new_dims(request):
    if request.param == "empty":
        return ()
    elif request.param == "1_dim":
        return ("dim1",)
    elif request.param == "2_dims":
        return ("dim_2", "dim_1")
    else:
        raise NotImplementedError()


@pytest.fixture
def result_dims(data_array, new_dims):
    kept_dims = len(data_array.dims) - len(new_dims)
    return tuple(list(data_array.dims[:kept_dims]) + list(new_dims))


@pytest.mark.cpu_only
@requires_xarray
def test_apply_dims(data_array, new_dims, result_dims):
    result = pace.util._legacy_restart._apply_dims(data_array, new_dims)
    np.testing.assert_array_equal(result.values, data_array.values)
    assert result.dims == result_dims
    assert result.attrs == data_array.attrs


@pytest.mark.parametrize(
    "old_dict, key_mapping, new_dict",
    [
        pytest.param(
            {},
            {},
            {},
            id="empty_dict",
        ),
        pytest.param(
            {"key1": 1, "key2": 2},
            {},
            {"key1": 1, "key2": 2},
            id="empty_map",
        ),
        pytest.param(
            {"key1": 1, "key2": 2},
            {"key1": "key_1"},
            {"key_1": 1, "key2": 2},
            id="one_item_map",
        ),
        pytest.param(
            {"key1": 1, "key2": 2},
            {"key3": "key_3"},
            {"key1": 1, "key2": 2},
            id="map_not_in_dict",
        ),
        pytest.param(
            {"key1": 1, "key2": 2},
            {"key1": "key_1", "key2": "key_2"},
            {"key_1": 1, "key_2": 2},
            id="two_item_map",
        ),
    ],
)
@pytest.mark.cpu_only
def test_map_keys(old_dict, key_mapping, new_dict):
    result = pace.util._legacy_restart.map_keys(old_dict, key_mapping)
    assert result == new_dict


@pytest.mark.parametrize(
    "rank, total_ranks, suffix",
    [
        pytest.param(
            0,
            6,
            ".tile1.nc",
            id="first_tile",
        ),
        pytest.param(
            2,
            6,
            ".tile3.nc",
            id="third_tile",
        ),
        pytest.param(
            2,
            24,
            ".tile1.nc.0002",
            id="third_subtile",
        ),
        pytest.param(
            6,
            24,
            ".tile2.nc.0002",
            id="third_subtile_second_tile",
        ),
    ],
)
@pytest.mark.cpu_only
def test_get_rank_suffix(rank, total_ranks, suffix):
    result = pace.util._legacy_restart.get_rank_suffix(rank, total_ranks)
    assert result == suffix


@pytest.mark.parametrize("invalid_total_ranks", [5, 7, 9, 23])
@pytest.mark.cpu_only
def test_get_rank_suffix_invalid_total_ranks(invalid_total_ranks):
    with pytest.raises(ValueError):
        # total_ranks should be multiple of 6
        pace.util._legacy_restart.get_rank_suffix(0, invalid_total_ranks)


@pytest.mark.cpu_only
@requires_xarray
def test_read_state_incorrectly_encoded_time():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nc") as file:
        state_ds = xr.DataArray(0.0, name="time").to_dataset()
        state_ds.to_netcdf(file.name)
        with pytest.raises(ValueError, match="Time in stored state"):
            pace.util.io.read_state(file.name)


@pytest.mark.cpu_only
@requires_xarray
def test_read_state_non_scalar_time():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".nc") as file:
        state_ds = xr.DataArray([0.0, 1.0], dims=["T"], name="time").to_dataset()
        state_ds.to_netcdf(file.name)
        with pytest.raises(ValueError, match="scalar time"):
            pace.util.io.read_state(file.name)


@pytest.mark.parametrize(
    "only_names",
    [["time", "air_temperature"], ["air_temperature"]],
    ids=lambda x: f"{x}",
)
@requires_xarray
def test_open_c12_restart_only_names(layout, only_names):
    tracer_properties = {}
    c12_restart_state_list = get_c12_restart_state_list(
        layout, only_names, tracer_properties
    )
    for state in c12_restart_state_list:
        assert set(only_names) == set(state.keys())
