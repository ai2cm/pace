import pytest
import fv3util
import fv3util.domain
import xarray as xr
import numpy as np
from utils import DummyComm


rank_list = []
total_rank_list = []
tile_index_list = []

for ranks_per_tile in (1, 4):
    total_ranks = 6 * ranks_per_tile
    rank = 0
    for tile in range(6):
        for subtile in range(ranks_per_tile):
            rank_list.append(rank)
            total_rank_list.append(total_ranks)
            tile_index_list.append(tile)
            rank += 1


@pytest.mark.parametrize(
    "rank, total_ranks, tile_index",
    zip(rank_list, total_rank_list, tile_index_list)
)
def test_get_tile_number(rank, total_ranks, tile_index):
    tile = fv3util.get_tile_number(rank, total_ranks)
    assert tile == tile_index + 1


@pytest.mark.parametrize(
    "rank, total_ranks, tile_index",
    zip(rank_list, total_rank_list, tile_index_list)
)
def test_get_tile_index(rank, total_ranks, tile_index):
    tile = fv3util.get_tile_index(rank, total_ranks)
    assert tile == tile_index


# initialize: rank, total_ranks, ny, nx, layout
# out: nx_rank, ny_rank, ranks_per_tile, subtile_index
# array_shape -> tile_extent
# array_dims -> subtile_slice

rank_list = []
layout_list = []
subtile_index_list = []

for layout in ((1, 1), (1, 2), (2, 2), (2, 3)):
    rank = 0
    for tile in range(6):
        for y_subtile in range(layout[0]):
            for x_subtile in range(layout[1]):
                rank_list.append(rank)
                layout_list.append(layout)
                subtile_index_list.append((y_subtile, x_subtile))
                rank += 1


@pytest.mark.parametrize(
    "rank, layout, subtile_index",
    zip(rank_list, layout_list, subtile_index_list))
def test_subtile_index(rank, layout, subtile_index):
    nz = 60
    ny = 49
    nx = 49
    partitioner = fv3util.Partitioner(nz, ny, nx, layout)
    assert partitioner.subtile_index(rank) == subtile_index


@pytest.mark.parametrize(
    "nz, ny, nx, array_dims, extent",
    [
        (8, 16, 32, ([fv3util.Y_DIM, fv3util.X_INTERFACE_DIM]), (16, 33)),
        (8, 16, 32, ([fv3util.Y_DIM, fv3util.X_DIM]), (16, 32)),
        (8, 16, 32, ([fv3util.Y_INTERFACE_DIM, fv3util.X_DIM]), (17, 32)),
        (8, 16, 32, ([fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM]), (17, 33)),
        (8, 16, 32, ([fv3util.Z_DIM, fv3util.Y_DIM, fv3util.X_INTERFACE_DIM]), (8, 16, 33)),
        (8, 16, 32, ([fv3util.Z_DIM, fv3util.Y_DIM, fv3util.X_DIM]), (8, 16, 32)),
        (8, 16, 32, ([fv3util.Z_DIM, fv3util.Y_INTERFACE_DIM, fv3util.X_DIM]), (8, 17, 32)),
        (8, 16, 32, ([fv3util.Z_DIM, fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM]), (8, 17, 33)),
        (8, 16, 32, ([fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM, fv3util.X_INTERFACE_DIM]), (9, 16, 33)),
        (8, 16, 32, ([fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM, fv3util.X_DIM]), (9, 16, 32)),
        (8, 16, 32, ([fv3util.Z_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.X_DIM]), (9, 17, 32)),
        (8, 16, 32, ([fv3util.Z_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM]), (9, 17, 33)),
    ],
)
def test_tile_extent(nz, ny, nx, array_dims, extent):
    result = fv3util.domain.tile_extent(nz, ny, nx, array_dims)
    assert result == extent


@pytest.mark.parametrize(
    'array_dims, nz, ny_rank, nx_rank, layout, subtile_index, subtile_slice, overlap',
    [
        pytest.param(
            [fv3util.Y_DIM, fv3util.X_DIM], 10, 8, 8, (1, 1), (0, 0), (slice(0, 8), slice(0, 8)), False,
            id='6_rank_centered'
        ),
        pytest.param(
            [fv3util.Z_DIM, fv3util.Y_DIM, fv3util.X_DIM], 10, 8, 8, (1, 1), (0, 0), (slice(0, 10), slice(0, 8), slice(0, 8)), False,
            id='6_rank_centered_3d'
        ),
        pytest.param(
            [fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM, fv3util.X_DIM], 10, 8, 8, (1, 1), (0, 0), (slice(0, 11), slice(0, 8), slice(0, 8)), False,
            id='6_rank_centered_z_interface'
        ),
        pytest.param(
            [fv3util.Y_INTERFACE_DIM, fv3util.X_DIM], 10, 8, 8, (1, 1), (0, 0), (slice(0, 9), slice(0, 8)), True,
            id='6_rank_y_interface'
        ),
        pytest.param(
            [fv3util.Y_DIM, fv3util.X_INTERFACE_DIM], 10, 8, 8, (1, 1), (0, 0), (slice(0, 8), slice(0, 9)), True,
            id='6_rank_x_interface'
        ),
        pytest.param(
            [fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM], 10, 8, 8, (1, 1), (0, 0), (slice(0, 9), slice(0, 9)), False,
            id='6_rank_both_interface'
        ),
        pytest.param(
            [fv3util.Y_DIM, fv3util.X_DIM], 10, 4, 4, (2, 2), (0, 0), (slice(0, 4), slice(0, 4)), True,
            id='24_rank_centered_left'
        ),
        pytest.param(
            [fv3util.Y_DIM, fv3util.X_DIM], 10, 4, 4, (2, 2), (1, 1), (slice(4, 8), slice(4, 8)), False,
            id='24_rank_centered_right'
        ),
        pytest.param(
            [fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM], 10, 4, 4, (2, 2), (0, 0), (slice(0, 4), slice(0, 4)), False,
            id='24_rank_interface_left_no_overlap'
        ),
        pytest.param(
            [fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM], 10, 4, 4, (2, 2), (1, 1), (slice(4, 9), slice(4, 9)), False,
            id='24_rank_interface_right_no_overlap'
        ),
        pytest.param(
            [fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM], 10, 4, 4, (2, 2), (0, 0), (slice(0, 5), slice(0, 5)), True,
            id='24_rank_interface_left_overlap',
        ),
        pytest.param(
            [fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM], 10, 4, 4, (2, 2), (1, 1), (slice(4, 9), slice(4, 9)), True,
            id='24_rank_interface_right_overlap'
        ),
        pytest.param(
            [fv3util.Y_DIM, fv3util.X_DIM], 10, 4, 2, (1, 2), (0, 0), (slice(0, 4), slice(0, 2)), True,
            id='24_rank_interface_right_overlap'
        ),
        pytest.param(
            [fv3util.Y_DIM, fv3util.X_DIM], 10, 4, 2, (1, 2), (0, 1), (slice(0, 4), slice(2, 4)), True,
            id='24_rank_interface_right_overlap'
        ),
    ]
)
def test_subtile_slice(array_dims, nz, ny_rank, nx_rank, layout, subtile_index, subtile_slice, overlap):
    result = fv3util.domain.subtile_slice(
        array_dims, nz, ny_rank, nx_rank, layout, subtile_index, overlap
    )
    assert result == subtile_slice


def get_metadata(array):
    return fv3util.ArrayMetadata(dims=array.dims, units=array.attrs['units'], dtype=array.dtype)


@pytest.mark.parametrize(
    'layout', [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
)
def test_centered_state_one_item_per_rank_scatter_tile(layout):
    nz = 5
    ny = layout[0]
    nx = layout[1]
    total_ranks = layout[0] * layout[1]
    state = {
        'rank': xr.DataArray(
            np.empty([layout[0], layout[1]]),
            dims=[fv3util.Y_DIM, fv3util.X_DIM],
            attrs={'units': 'dimensionless'}
        ),
        'rank_pos_j': xr.DataArray(
            np.empty([layout[0], layout[1]]),
            dims=[fv3util.Y_DIM, fv3util.X_DIM],
            attrs={'units': 'dimensionless'}
        ),
        'rank_pos_i': xr.DataArray(
            np.empty([layout[0], layout[1]]),
            dims=[fv3util.Y_DIM, fv3util.X_DIM],
            attrs={'units': 'dimensionless'}
        ),
    }
    
    partitioner = fv3util.Partitioner(nz, ny, nx, layout)
    for rank in range(total_ranks):
        state['rank'].values[np.unravel_index(rank, state['rank'].shape)] = rank
        j, i = partitioner.subtile_index(rank)
        state['rank_pos_j'].values[np.unravel_index(rank, state['rank_pos_j'].shape)] = j
        state['rank_pos_i'].values[np.unravel_index(rank, state['rank_pos_i'].shape)] = i

    shared_buffer = {}
    tile_comm_list = []
    for rank in range(total_ranks):
        tile_comm_list.append(
            DummyComm(rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer)
        )
    for rank, tile_comm in enumerate(tile_comm_list):
        if rank == 0:
            array = state['rank']
        else:
            array = None
        metadata = get_metadata(state['rank'])
        print(state['rank'])
        rank_array = partitioner.scatter_tile(tile_comm, array, metadata)
        assert rank_array.shape == (1, 1)
        assert rank_array[0, 0] == rank
        assert rank_array.dtype == state['rank'].dtype


@pytest.mark.parametrize(
    'layout', [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3)]
)
def test_interface_state_two_by_two_per_rank_scatter_tile(layout):
    nz = 5
    ny = layout[0]
    nx = layout[1]
    total_ranks = layout[0] * layout[1]
    state = {
        'pos_j': xr.DataArray(
            np.empty([layout[0] + 1, layout[1] + 1]),
            dims=[fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM],
            attrs={'units': 'dimensionless'}
        ),
        'pos_i': xr.DataArray(
            np.empty([layout[0] + 1, layout[1] + 1], dtype=np.int32),
            dims=[fv3util.Y_INTERFACE_DIM, fv3util.X_INTERFACE_DIM],
            attrs={'units': 'dimensionless'}
        ),
    }
    
    state['pos_j'][:, :] = np.arange(0, layout[0] + 1)[:, None]
    state['pos_i'][:, :] = np.arange(0, layout[1] + 1)[None, :]

    shared_buffer = {}
    tile_comm_list = []
    for rank in range(total_ranks):
        tile_comm_list.append(
            DummyComm(rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer)
        )
    partitioner = fv3util.Partitioner(nz, ny, nx, layout)
    for rank, tile_comm in enumerate(tile_comm_list):
        if rank == 0:
            array = state['pos_j']
        else:
            array = None
        metadata = get_metadata(state['pos_j'])
        rank_array = partitioner.scatter_tile(tile_comm, array, metadata)
        assert rank_array.shape == (2, 2)
        j, i = partitioner.subtile_index(rank)
        assert rank_array[0, 0] == j
        assert rank_array[0, 1] == j
        assert rank_array[1, 0] == j + 1
        assert rank_array[1, 1] == j + 1
        assert rank_array.dtype == state['pos_j'].dtype

    for rank, tile_comm in enumerate(tile_comm_list):
        if rank == 0:
            array = state['pos_i']
        else:
            array = None
        metadata = get_metadata(state['pos_i'])
        rank_array = partitioner.scatter_tile(tile_comm, array, metadata)
        assert rank_array.shape == (2, 2)
        j, i = partitioner.subtile_index(rank)
        assert rank_array[0, 0] == i
        assert rank_array[1, 0] == i
        assert rank_array[0, 1] == i + 1
        assert rank_array[1, 1] == i + 1
        assert rank_array.dtype == state['pos_i'].dtype
