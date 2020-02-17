import pytest
import fv3util
import fv3util.domain


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
# array_dims -> subtile_range

rank_list = []
total_rank_list = []
layout_list = []
subtile_index_list = []

for layout in ((1, 1), (2, 2), (2, 3)):
    rank = 0
    total_ranks = layout[0] * layout[1] * 6
    for tile in range(6):
        for y_subtile in range(layout[1]):
            for x_subtile in range(layout[0]):
                rank_list.append(rank)
                total_rank_list.append(total_ranks)
                layout_list.append(layout)
                subtile_index_list.append((y_subtile, x_subtile))
                rank += 1


@pytest.mark.parametrize(
    "rank, total_ranks, layout, subtile_index",
    zip(rank_list, total_rank_list, layout_list, subtile_index_list))
def test_subtile_index(rank, total_ranks, layout, subtile_index):
    nz = 60
    ny = 49
    nx = 49
    grid_2d = fv3util.Grid2D(rank, total_ranks, nz, ny, nx, layout)
    assert grid_2d.subtile_index == subtile_index


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
    'array_dims, nz, ny_rank, nx_rank, layout, subtile_index, subtile_range, overlap',
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
    ]
)
def test_subtile_range(array_dims, nz, ny_rank, nx_rank, layout, subtile_index, subtile_range, overlap):
    result = fv3util.domain.subtile_range(
        array_dims, nz, ny_rank, nx_rank, layout, subtile_index, overlap
    )
    assert result == subtile_range

