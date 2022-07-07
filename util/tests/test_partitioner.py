import numpy as np
import pytest

import pace.util
import pace.util.partitioner


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
    tile = pace.util.get_tile_number(rank, total_ranks)
    assert tile == tile_index + 1


@pytest.mark.parametrize(
    "rank, total_ranks, tile_index", zip(rank_list, total_rank_list, tile_index_list)
)
@pytest.mark.cpu_only
def test_get_tile_index(rank, total_ranks, tile_index):
    tile = pace.util.get_tile_index(rank, total_ranks)
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
    partitioner = pace.util.TilePartitioner(layout)
    assert partitioner.subtile_index(rank) == subtile_index


@pytest.mark.parametrize(
    "array_extent, array_dims, layout, tile_extent",
    [
        ((16, 32), (pace.util.Y_DIM, pace.util.X_DIM), (1, 1), (16, 32)),
        ((16, 32), (pace.util.Y_DIM, pace.util.X_INTERFACE_DIM), (1, 1), (16, 32)),
        ((16, 32), (pace.util.Y_INTERFACE_DIM, pace.util.X_DIM), (1, 1), (16, 32)),
        (
            (16, 32),
            (pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM),
            (1, 1),
            (16, 32),
        ),
        (
            (8, 16, 32),
            (pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM),
            (1, 1),
            (8, 16, 32),
        ),
        ((2, 2), (pace.util.Y_DIM, pace.util.X_DIM), (2, 2), (4, 4)),
        ((3, 2), (pace.util.Y_INTERFACE_DIM, pace.util.X_DIM), (2, 2), (5, 4)),
        ((2, 3), (pace.util.Y_DIM, pace.util.X_INTERFACE_DIM), (2, 2), (4, 5)),
        (
            (4, 2, 3),
            (
                pace.util.Z_INTERFACE_DIM,
                pace.util.Y_DIM,
                pace.util.X_INTERFACE_DIM,
            ),
            (2, 2),
            (4, 4, 5),
        ),
    ],
)
@pytest.mark.cpu_only
def test_tile_extent_from_rank_metadata(array_extent, array_dims, layout, tile_extent):
    result = pace.util.partitioner.tile_extent_from_rank_metadata(
        array_dims, array_extent, layout
    )
    assert result == tile_extent


@pytest.mark.parametrize(
    (
        "array_dims, tile_extent, layout, rank, subtile_slice, "
        "overlap, edge_interior_ratio"
    ),
    [
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (8, 8),
            (1, 1),
            0,
            (slice(0, 8), slice(0, 8)),
            False,
            1.0,
            id="6_rank_centered",
        ),
        pytest.param(
            [pace.util.Z_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (10, 8, 8),
            (1, 1),
            0,
            (slice(0, 10), slice(0, 8), slice(0, 8)),
            False,
            1.0,
            id="6_rank_centered_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (11, 8, 8),
            (1, 1),
            0,
            (slice(0, 11), slice(0, 8), slice(0, 8)),
            False,
            1.0,
            id="6_rank_centered_z_interface",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_DIM],
            (9, 8),
            (1, 1),
            0,
            (slice(0, 9), slice(0, 8)),
            True,
            1.0,
            id="6_rank_y_interface",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_INTERFACE_DIM],
            (8, 9),
            (1, 1),
            0,
            (slice(0, 8), slice(0, 9)),
            True,
            1.0,
            id="6_rank_x_interface",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (9, 9),
            (1, 1),
            0,
            (slice(0, 9), slice(0, 9)),
            False,
            1.0,
            id="6_rank_both_interface",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (8, 8),
            (2, 2),
            0,
            (slice(0, 4), slice(0, 4)),
            True,
            1.0,
            id="24_rank_centered_left",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (8, 8),
            (2, 2),
            3,
            (slice(4, 8), slice(4, 8)),
            False,
            1.0,
            id="24_rank_centered_right",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            0,
            (slice(0, 4), slice(0, 4)),
            False,
            1.0,
            id="24_rank_interface_left_no_overlap",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            3,
            (slice(4, 9), slice(4, 9)),
            False,
            1.0,
            id="24_rank_interface_right_no_overlap",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            0,
            (slice(0, 5), slice(0, 5)),
            True,
            1.0,
            id="24_rank_interface_left_overlap",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (9, 9),
            (2, 2),
            3,
            (slice(4, 9), slice(4, 9)),
            True,
            1.0,
            id="24_rank_interface_right_overlap",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (4, 4),
            (1, 2),
            0,
            (slice(0, 4), slice(0, 2)),
            True,
            1.0,
            id="12_rank_no_interface_right_overlap",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (4, 4),
            (1, 2),
            1,
            (slice(0, 4), slice(2, 4)),
            True,
            1.0,
            id="12_rank_no_interface_right_overlap",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (4, 4),
            (1, 2),
            1,
            (slice(0, 4), slice(2, 4)),
            False,
            1.0,
            id="12_rank_centered_right_no_overlap_rectangle_layout",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (1, 3),
            0,
            (slice(0, 5), slice(0, 4), slice(0, 1)),
            False,
            0.5,
            id="18_rank_left_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (1, 3),
            1,
            (slice(0, 5), slice(0, 4), slice(1, 3)),
            False,
            0.5,
            id="18_rank_mid_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (1, 3),
            2,
            (slice(0, 5), slice(0, 4), slice(3, 4)),
            False,
            0.5,
            id="18_rank_right_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (2, 3),
            0,
            (slice(0, 5), slice(0, 2), slice(0, 1)),
            False,
            0.5,
            id="36_rank_botleft_right_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (2, 3),
            1,
            (slice(0, 5), slice(0, 2), slice(1, 3)),
            False,
            0.5,
            id="36_rank_botmid_right_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (2, 3),
            2,
            (slice(0, 5), slice(0, 2), slice(3, 4)),
            False,
            0.5,
            id="36_rank_botright_right_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (2, 3),
            3,
            (slice(0, 5), slice(2, 4), slice(0, 1)),
            False,
            0.5,
            id="36_rank_topleft_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (2, 3),
            4,
            (slice(0, 5), slice(2, 4), slice(1, 3)),
            False,
            0.5,
            id="36_rank_topmid_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 4, 4),
            (2, 3),
            5,
            (slice(0, 5), slice(2, 4), slice(3, 4)),
            False,
            0.5,
            id="36_rank_topright_no_overlap_rectangle_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            0,
            (slice(0, 5), slice(0, 2), slice(0, 2)),
            False,
            0.5,
            id="54_rank_botleft_no_overlap_square_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            1,
            (slice(0, 5), slice(0, 2), slice(2, 6)),
            False,
            0.5,
            id="54_rank_botmid_no_overlap_square_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            4,
            (slice(0, 5), slice(2, 6), slice(2, 6)),
            False,
            0.5,
            id="54_rank_midmid_no_overlap_square_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            5,
            (slice(0, 5), slice(2, 6), slice(6, 8)),
            False,
            0.5,
            id="54_rank_midright_no_overlap_square_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            8,
            (slice(0, 5), slice(6, 8), slice(6, 8)),
            False,
            0.5,
            id="54_rank_topright_no_overlap_square_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            0,
            (slice(0, 5), slice(0, 1), slice(0, 1)),
            False,
            float(1.0 / 6),
            id="54_rank_botleft_no_overlap_square_layout_sixth_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            1,
            (slice(0, 5), slice(0, 1), slice(1, 7)),
            False,
            float(1.0 / 6),
            id="54_rank_botmid_no_overlap_square_layout_sixth_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            4,
            (slice(0, 5), slice(1, 7), slice(1, 7)),
            False,
            float(1.0 / 6),
            id="54_rank_midmid_no_overlap_square_layout_sixth_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            5,
            (slice(0, 5), slice(1, 7), slice(7, 8)),
            False,
            float(1.0 / 6),
            id="54_rank_midright_no_overlap_square_layout_sixth_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 8, 8),
            (3, 3),
            8,
            (slice(0, 5), slice(7, 8), slice(7, 8)),
            False,
            float(1.0 / 6),
            id="54_rank_topright_no_overlap_square_layout_half_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 16, 16),
            (4, 4),
            0,
            (slice(0, 5), slice(0, 2), slice(0, 2)),
            False,
            float(1.0 / 3),
            id="96_rank_farbotfarleft_no_overlap_square_layout_third_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 16, 16),
            (4, 4),
            1,
            (slice(0, 5), slice(0, 2), slice(2, 8)),
            False,
            float(1.0 / 3),
            id="96_rank_farbotcloseleft_no_overlap_square_layout_third_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 16, 16),
            (4, 4),
            6,
            (slice(0, 5), slice(2, 8), slice(8, 14)),
            False,
            float(1.0 / 3),
            id=(
                "96_rank_closebotcloseright_right_no_overlap_"
                "square_layout_third_edge_tiles_3d"
            ),
        ),
        pytest.param(
            [pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM, pace.util.X_DIM],
            (5, 16, 16),
            (4, 4),
            14,
            (slice(0, 5), slice(14, 16), slice(8, 14)),
            False,
            float(1.0 / 3),
            id="96_rank_fartopcloseright_no_overlap_square_layout_third_edge_tiles_3d",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (13, 13),
            (2, 4),
            1,
            (slice(0, 7), slice(3, 7)),
            True,
            1.0,
            id="48_rank_botcloseleft_interface_right_overlap",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (13, 13),
            (2, 4),
            1,
            (slice(0, 7), slice(2, 7)),
            True,
            0.5,
            id="48_rank_botcloseleft_interface_overlap_half_edge",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_INTERFACE_DIM],
            (13, 13),
            (2, 4),
            7,
            (slice(6, 13), slice(10, 13)),
            True,
            0.5,
            id="48_rank_topfarright_interface_overlap_half_edge",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (12, 12),
            (2, 4),
            0,
            (slice(0, 6), slice(0, 2)),
            True,
            0.5,
            id="48_rank_botfarleft_overlap_half_edge",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_INTERFACE_DIM],
            (12, 13),
            (3, 4),
            0,
            (slice(0, 3), slice(0, 2)),
            False,
            0.5,
            id="72_rank_botfarleft_x_interface_half_edge",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_INTERFACE_DIM],
            (12, 13),
            (3, 4),
            1,
            (slice(0, 3), slice(2, 6)),
            False,
            0.5,
            id="72_rank_botcloseleft_x_interface_half_edge",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_INTERFACE_DIM],
            (12, 13),
            (3, 4),
            11,
            (slice(9, 12), slice(10, 13)),
            False,
            0.5,
            id="72_rank_topfarright_x_interface_half_edge",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_DIM],
            (13, 12),
            (3, 4),
            0,
            (slice(0, 3), slice(0, 2)),
            False,
            0.5,
            id="72_rank_botfarleft_y_interface_half_edge",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_DIM],
            (13, 12),
            (3, 4),
            1,
            (slice(0, 3), slice(2, 6)),
            False,
            0.5,
            id="72_rank_botcloseleft_y_interface_half_edge",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_DIM],
            (13, 12),
            (3, 4),
            8,
            (slice(9, 13), slice(0, 2)),
            False,
            0.5,
            id="72_rank_topfarleft_y_interface_half_edge",
        ),
        pytest.param(
            [pace.util.Y_INTERFACE_DIM, pace.util.X_DIM],
            (13, 12),
            (3, 4),
            11,
            (slice(9, 13), slice(10, 12)),
            False,
            0.5,
            id="72_rank_topfarright_y_interface_half_edge",
        ),
    ],
)
@pytest.mark.cpu_only
def test_subtile_slice(
    array_dims, tile_extent, layout, rank, subtile_slice, overlap, edge_interior_ratio
):
    partitioner = pace.util.TilePartitioner(layout, edge_interior_ratio)
    result = partitioner.subtile_slice(rank, array_dims, tile_extent, overlap)
    assert result == subtile_slice


@pytest.mark.parametrize(
    (
        "array_dims, tile_extent, layout, rank, subtile_slice, "
        "overlap, edge_interior_ratio"
    ),
    [
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (16, 16),
            (5, 5),
            0,
            (slice(0, 2), slice(0, 2)),
            False,
            1.0,
            id="150_rank_yx_botfarleft",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (16, 16),
            (5, 5),
            1,
            (slice(0, 2), slice(2, 6)),
            False,
            1.0,
            id="150_rank_yx_botcloseleft",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (16, 16),
            (5, 5),
            8,
            (slice(2, 6), slice(10, 14)),
            False,
            1.0,
            id="150_rank_yx_closebotcloseright",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (16, 16),
            (5, 5),
            14,
            (slice(6, 10), slice(14, 16)),
            False,
            1.0,
            id="150_rank_yx_midfarright",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (16, 16),
            (5, 5),
            24,
            (slice(14, 16), slice(14, 16)),
            False,
            1.0,
            id="150_rank_yx_fartopfarright",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Y_DIM],
            (16, 16),
            (5, 5),
            0,
            (slice(0, 2), slice(0, 2)),
            False,
            1.0,
            id="150_rank_xy_botfarleft",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Y_DIM],
            (16, 16),
            (5, 5),
            1,
            (slice(2, 6), slice(0, 2)),
            False,
            1.0,
            id="150_rank_xy_botcloseleft",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Y_DIM],
            (16, 16),
            (5, 5),
            8,
            (slice(10, 14), slice(2, 6)),
            False,
            1.0,
            id="150_rank_xy_closebotcloseright",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Y_DIM],
            (16, 16),
            (5, 5),
            14,
            (slice(14, 16), slice(6, 10)),
            False,
            1.0,
            id="150_rank_xy_midfarright",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Y_DIM],
            (16, 16),
            (5, 5),
            24,
            (slice(14, 16), slice(14, 16)),
            False,
            1.0,
            id="150_rank_xy_fartopfarright",
        ),
    ],
)
@pytest.mark.cpu_only
def test_subtile_slice_even_grid_odd_layout(
    array_dims, tile_extent, layout, rank, subtile_slice, overlap, edge_interior_ratio
):
    partitioner = pace.util.TilePartitioner(layout, edge_interior_ratio)
    result = partitioner.subtile_slice(rank, array_dims, tile_extent, overlap)
    assert result == subtile_slice


@pytest.mark.parametrize(
    (
        "array_dims, tile_extent, layout, rank, expected_error_string, "
        "overlap, edge_interior_ratio"
    ),
    [
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (13, 19),
            (4, 4),
            24,
            (
                "Cannot find valid decomposition for odd \\(13\\) gridpoints "
                "along an even count \\(4\\) of ranks."
            ),
            False,
            0.5,
            id="48_rank_odd_grid_even_layout_y",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (12, 19),
            (4, 2),
            24,
            (
                "Cannot find valid decomposition for odd \\(19\\) gridpoints "
                "along an even count \\(2\\) of ranks."
            ),
            False,
            1.0,
            id="48_rank_odd_grid_even_layout_x",
        ),
    ],
)
@pytest.mark.cpu_only
def test_subtile_slice_odd_grid_even_layout_no_interface(
    array_dims,
    tile_extent,
    layout,
    rank,
    expected_error_string,
    overlap,
    edge_interior_ratio,
):
    partitioner = pace.util.TilePartitioner(layout, edge_interior_ratio)
    with pytest.raises(ValueError, match=expected_error_string):
        partitioner.subtile_slice(rank, array_dims, tile_extent, overlap)


@pytest.mark.parametrize(
    "array_dims, tile_extent, layout, edge_interior_ratio, rank_extent",
    [
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (12, 12),
            (2, 3),
            1.0,
            ((6, 4), (6, 4)),
            id="36_rank_full_edge_tiles",
        ),
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (12, 12),
            (2, 3),
            0.5,
            ((6, 6), (6, 3)),
            id="36_rank_half_edge_tiles",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Y_DIM],
            (12, 12),
            (3, 4),
            0.5,
            ((4, 6), (2, 3)),
            id="72_rank_half_edge_tiles",
        ),
        pytest.param(
            [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM],
            (13, 12),
            (3, 4),
            0.5,
            ((4, 6), (2, 3)),
            id="72_rank_half_edge_tiles_x_interface",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
            (12, 13),
            (3, 4),
            0.5,
            ((4, 6), (2, 3)),
            id="72_rank_half_edge_tiles_y_interface",
        ),
        pytest.param(
            [pace.util.X_DIM, pace.util.Z_DIM, pace.util.Y_DIM],
            (12, 5, 12),
            (3, 4),
            0.5,
            ((4, 5, 6), (2, 5, 3)),
            id="72_rank_3d_half_edge_tiles",
        ),
        pytest.param(
            [
                pace.util.TILE_DIM,
                pace.util.X_DIM,
                pace.util.Z_DIM,
                pace.util.Y_DIM,
            ],
            (6, 12, 5, 12),
            (3, 4),
            0.5,
            ((6, 4, 5, 6), (6, 2, 5, 3)),
            id="72_rank_3d_with_tile_dim_half_edge_tiles",
        ),
    ],
)
@pytest.mark.cpu_only
def test_subtile_extents_from_tile_metadata(
    array_dims, tile_extent, layout, edge_interior_ratio, rank_extent
):
    result = pace.util.partitioner._subtile_extents_from_tile_metadata(
        array_dims, tile_extent, layout, edge_interior_ratio
    )
    assert result == rank_extent
    result = pace.util.partitioner._subtile_extents_from_tile_metadata(
        dims=array_dims,
        tile_extent=tile_extent,
        layout=layout,
        edge_interior_ratio=edge_interior_ratio,
    )
    assert result == rank_extent


@pytest.mark.parametrize(
    (
        "array_dims, tile_extent, layout, full_edge_interior_ratio, "
        "half_edge_interior_ratio, expected_slice, expected_extent, "
        "expected_error_string"
    ),
    [
        pytest.param(
            [pace.util.Y_DIM, pace.util.X_DIM],
            (12, 12),
            (3, 4),
            1.0,
            0.5,
            (slice(0, 3, None), slice(0, 2, None)),
            (9, 8),
            (
                "Only equal sized subdomains are supported, "
                "was given an edge_interior_ratio of 0.5"
            ),
            id="72_rank_half_edge_tiles",
        ),
    ],
)
@pytest.mark.cpu_only
def test_tile_extent_from_metadata(
    array_dims,
    tile_extent,
    layout,
    full_edge_interior_ratio,
    half_edge_interior_ratio,
    expected_slice,
    expected_extent,
    expected_error_string,
):
    partitioner = pace.util.TilePartitioner(layout, half_edge_interior_ratio)
    subtile_slice = partitioner.subtile_slice(0, array_dims, tile_extent, False)
    assert subtile_slice == expected_slice
    slice_extent = (
        subtile_slice[0].stop - subtile_slice[0].start,
        subtile_slice[1].stop - subtile_slice[1].start,
    )
    rank_extent = pace.util.partitioner.tile_extent_from_rank_metadata(
        array_dims, slice_extent, layout, full_edge_interior_ratio
    )
    assert rank_extent == expected_extent
    with pytest.raises(NotImplementedError, match=expected_error_string):
        pace.util.partitioner.tile_extent_from_rank_metadata(
            array_dims, slice_extent, layout, half_edge_interior_ratio
        )


@pytest.mark.parametrize(
    (
        "array_dims, tile_extent, layout, edge_interior_ratio, rank, "
        "tile_expected, cubedsphere_expected"
    ),
    [
        pytest.param(
            [
                pace.util.TILE_DIM,
                pace.util.X_DIM,
                pace.util.Z_DIM,
                pace.util.Y_DIM,
            ],
            (6, 12, 5, 12),
            (3, 4),
            0.5,
            0,
            (2, 5, 3),
            (2, 5, 3),
            id="72_rank_tile_x_z_y_half_edge_tiles",
        ),
        pytest.param(
            [
                pace.util.TILE_DIM,
                pace.util.Y_DIM,
                pace.util.Z_DIM,
                pace.util.X_DIM,
            ],
            (6, 12, 5, 12),
            (3, 4),
            0.5,
            0,
            (3, 5, 2),
            (3, 5, 2),
            id="72_rank_tile_y_z_x_half_edge_tiles",
        ),
        pytest.param(
            [
                pace.util.Z_DIM,
                pace.util.Y_DIM,
                pace.util.TILE_DIM,
                pace.util.X_DIM,
            ],
            (5, 12, 6, 12),
            (3, 4),
            0.5,
            0,
            (5, 3, 2),
            (5, 3, 2),
            id="72_rank_z_y_tile_x_half_edge_tiles",
        ),
    ],
)
def test_subtile_extent_with_tile_dimensions(
    array_dims,
    tile_extent,
    layout,
    edge_interior_ratio,
    rank,
    tile_expected,
    cubedsphere_expected,
):
    data_array = np.zeros((tile_extent))
    quantity = pace.util.Quantity(data_array, array_dims, "dimensionless", [0, 0, 0, 0])

    tile_partitioner = pace.util.TilePartitioner(layout, edge_interior_ratio)
    cubedsphere_partitioner = pace.util.CubedSpherePartitioner(tile_partitioner)

    tile_result = tile_partitioner.subtile_extent(quantity.metadata, rank)
    assert tile_result == tile_expected
    cubedsphere_result = cubedsphere_partitioner.subtile_extent(quantity.metadata, rank)
    assert cubedsphere_result == cubedsphere_expected
