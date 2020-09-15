import pytest
import fv3gfs.util
import fv3gfs.util.partitioner


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
    "rank, total_ranks, tile_index", zip(rank_list, total_rank_list, tile_index_list)
)
@pytest.mark.cpu_only
def test_get_tile_number(rank, total_ranks, tile_index):
    tile = fv3gfs.util.get_tile_number(rank, total_ranks)
    assert tile == tile_index + 1


@pytest.mark.parametrize(
    "rank, total_ranks, tile_index", zip(rank_list, total_rank_list, tile_index_list)
)
@pytest.mark.cpu_only
def test_get_tile_index(rank, total_ranks, tile_index):
    tile = fv3gfs.util.get_tile_index(rank, total_ranks)
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
    "rank, layout, subtile_index", zip(rank_list, layout_list, subtile_index_list)
)
@pytest.mark.cpu_only
def test_subtile_index(rank, layout, subtile_index):
    partitioner = fv3gfs.util.TilePartitioner(layout)
    assert partitioner.subtile_index(rank) == subtile_index


@pytest.mark.parametrize(
    "array_extent, array_dims, layout, tile_extent",
    [
        ((16, 32), (fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM), (1, 1), (16, 32)),
        ((16, 32), (fv3gfs.util.Y_DIM, fv3gfs.util.X_INTERFACE_DIM), (1, 1), (16, 32)),
        ((16, 32), (fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_DIM), (1, 1), (16, 32)),
        (
            (16, 32),
            (fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_INTERFACE_DIM),
            (1, 1),
            (16, 32),
        ),
        (
            (8, 16, 32),
            (fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM),
            (1, 1),
            (8, 16, 32),
        ),
        ((2, 2), (fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM), (2, 2), (4, 4)),
        ((3, 2), (fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_DIM), (2, 2), (5, 4)),
        ((2, 3), (fv3gfs.util.Y_DIM, fv3gfs.util.X_INTERFACE_DIM), (2, 2), (4, 5)),
        (
            (4, 2, 3),
            (
                fv3gfs.util.Z_INTERFACE_DIM,
                fv3gfs.util.Y_DIM,
                fv3gfs.util.X_INTERFACE_DIM,
            ),
            (2, 2),
            (4, 4, 5),
        ),
    ],
)
@pytest.mark.cpu_only
def test_tile_extent_from_rank_metadata(array_extent, array_dims, layout, tile_extent):
    result = fv3gfs.util.partitioner.tile_extent_from_rank_metadata(
        array_dims, array_extent, layout
    )
    assert result == tile_extent


@pytest.mark.parametrize(
    "array_dims, tile_extent, layout, subtile_index, subtile_slice, overlap",
    [
        pytest.param(
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (8, 8),
            (1, 1),
            (0, 0),
            (slice(0, 8), slice(0, 8)),
            False,
            id="6_rank_centered",
        ),
        pytest.param(
            [fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (10, 8, 8),
            (1, 1),
            (0, 0),
            (slice(0, 10), slice(0, 8), slice(0, 8)),
            False,
            id="6_rank_centered_3d",
        ),
        pytest.param(
            [fv3gfs.util.Z_INTERFACE_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (11, 8, 8),
            (1, 1),
            (0, 0),
            (slice(0, 11), slice(0, 8), slice(0, 8)),
            False,
            id="6_rank_centered_z_interface",
        ),
        pytest.param(
            [fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_DIM],
            (9, 8),
            (1, 1),
            (0, 0),
            (slice(0, 9), slice(0, 8)),
            True,
            id="6_rank_y_interface",
        ),
        pytest.param(
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_INTERFACE_DIM],
            (8, 9),
            (1, 1),
            (0, 0),
            (slice(0, 8), slice(0, 9)),
            True,
            id="6_rank_x_interface",
        ),
        pytest.param(
            [fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_INTERFACE_DIM],
            (9, 9),
            (1, 1),
            (0, 0),
            (slice(0, 9), slice(0, 9)),
            False,
            id="6_rank_both_interface",
        ),
        pytest.param(
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (8, 8),
            (2, 2),
            (0, 0),
            (slice(0, 4), slice(0, 4)),
            True,
            id="24_rank_centered_left",
        ),
        pytest.param(
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (8, 8),
            (2, 2),
            (1, 1),
            (slice(4, 8), slice(4, 8)),
            False,
            id="24_rank_centered_right",
        ),
        pytest.param(
            [fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            (0, 0),
            (slice(0, 4), slice(0, 4)),
            False,
            id="24_rank_interface_left_no_overlap",
        ),
        pytest.param(
            [fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            (1, 1),
            (slice(4, 9), slice(4, 9)),
            False,
            id="24_rank_interface_right_no_overlap",
        ),
        pytest.param(
            [fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            (0, 0),
            (slice(0, 5), slice(0, 5)),
            True,
            id="24_rank_interface_left_overlap",
        ),
        pytest.param(
            [fv3gfs.util.Y_INTERFACE_DIM, fv3gfs.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            (1, 1),
            (slice(4, 9), slice(4, 9)),
            True,
            id="24_rank_interface_right_overlap",
        ),
        pytest.param(
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (4, 4),
            (1, 2),
            (0, 0),
            (slice(0, 4), slice(0, 2)),
            True,
            id="24_rank_interface_right_overlap",
        ),
        pytest.param(
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (4, 4),
            (1, 2),
            (0, 1),
            (slice(0, 4), slice(2, 4)),
            True,
            id="24_rank_interface_right_overlap",
        ),
        pytest.param(
            [fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM],
            (4, 4),
            (1, 2),
            (0, 1),
            (slice(0, 4), slice(2, 4)),
            False,
            id="24_rank_centered_right_no_overlap_rectangle_layout",
        ),
    ],
)
@pytest.mark.cpu_only
def test_subtile_slice(
    array_dims, tile_extent, layout, subtile_index, subtile_slice, overlap
):
    result = fv3gfs.util.partitioner.subtile_slice(
        array_dims, tile_extent, layout, subtile_index, overlap
    )
    assert result == subtile_slice
