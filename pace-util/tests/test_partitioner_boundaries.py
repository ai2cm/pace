import pytest

import pace.util
import pace.util.partitioner


# the test examples for the 2x2 cube here were recorded by manually inspecting
# a paper cube with printed ranks


@pytest.fixture
def partitioner_1_by_1():
    grid = pace.util.TilePartitioner((1, 1))
    return pace.util.CubedSpherePartitioner(grid)


@pytest.fixture
def partitioner_2_by_2():
    grid = pace.util.TilePartitioner((2, 2))
    return pace.util.CubedSpherePartitioner(grid)


@pytest.fixture
def partitioner_3_by_3():
    grid = pace.util.TilePartitioner((3, 3))
    return pace.util.CubedSpherePartitioner(grid)


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, 19, 1),  #
        (1, 0, 0),
        (2, 18, 1),
        (3, 2, 0),
        (4, 1, 0),  #
        (5, 4, 0),
        (6, 3, 0),
        (7, 6, 0),
        (8, 3, 1),  #
        (9, 8, 0),
        (10, 2, 1),
        (11, 10, 0),
        (12, 9, 0),  #
        (13, 12, 0),
        (14, 11, 0),
        (15, 14, 0),
        (16, 11, 1),  #
        (17, 16, 0),
        (18, 10, 1),
        (19, 18, 0),
        (20, 17, 0),  #
        (21, 20, 0),
        (22, 19, 0),
        (23, 22, 0),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_left_edge(
    partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations
):
    edge = partitioner_2_by_2.boundary(pace.util.WEST, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [(0, 4, 1), (1, 0, 0), (2, 0, 1), (3, 2, 0), (4, 2, 1), (5, 4, 0)],
)
@pytest.mark.cpu_only
def test_1_by_1_left_edge(
    partitioner_1_by_1, from_rank, to_rank, n_clockwise_rotations
):
    edge = partitioner_1_by_1.boundary(pace.util.WEST, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "rank, layout, n_clockwise_rotations, new_rank",
    [
        (12, (4, 4), 1, 15),
        (14, (4, 4), 1, 7),
        (0, (1, 1), 0, 0),
        (0, (1, 1), 1, 0),
        (2, (2, 2), 1, 3),
    ],
)
@pytest.mark.cpu_only
def test_rotate_subtile_rank(rank, layout, n_clockwise_rotations, new_rank):
    result = pace.util.partitioner.rotate_subtile_rank(
        rank, layout, n_clockwise_rotations
    )
    assert result == new_rank


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, 2, 0),  #
        (1, 3, 0),
        (2, 10, 3),
        (3, 8, 3),
        (4, 6, 0),  #
        (5, 7, 0),
        (6, 8, 0),
        (7, 9, 0),
        (8, 10, 0),  #
        (9, 11, 0),
        (10, 18, 3),
        (11, 16, 3),
        (12, 14, 0),  #
        (13, 15, 0),
        (14, 16, 0),
        (15, 17, 0),
        (16, 18, 0),  #
        (17, 19, 0),
        (18, 2, 3),
        (19, 0, 3),
        (20, 22, 0),  #
        (21, 23, 0),
        (22, 0, 0),
        (23, 1, 0),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_top_edge(partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations):
    edge = partitioner_2_by_2.boundary(pace.util.NORTH, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [(0, 2, 3), (1, 2, 0), (2, 4, 3), (3, 4, 0), (4, 0, 3), (5, 0, 0)],
)
@pytest.mark.cpu_only
def test_1_by_1_top_edge(partitioner_1_by_1, from_rank, to_rank, n_clockwise_rotations):
    edge = partitioner_1_by_1.boundary(pace.util.NORTH, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, 22, 0),  #
        (1, 23, 0),
        (2, 0, 0),
        (3, 1, 0),
        (4, 23, 3),  #
        (5, 21, 3),
        (6, 4, 0),
        (7, 5, 0),
        (8, 6, 0),  #
        (9, 7, 0),
        (10, 8, 0),
        (11, 9, 0),
        (12, 7, 3),  #
        (13, 5, 3),
        (14, 12, 0),
        (15, 13, 0),
        (16, 14, 0),  #
        (17, 15, 0),
        (18, 16, 0),
        (19, 17, 0),
        (20, 15, 3),  #
        (21, 13, 3),
        (22, 20, 0),
        (23, 21, 0),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_bottom_edge(
    partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations
):
    edge = partitioner_2_by_2.boundary(pace.util.SOUTH, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [(0, 5, 0), (1, 5, 3), (2, 1, 0), (3, 1, 3), (4, 3, 0), (5, 3, 3)],
)
@pytest.mark.cpu_only
def test_1_by_1_bottom_edge(
    partitioner_1_by_1, from_rank, to_rank, n_clockwise_rotations
):
    edge = partitioner_1_by_1.boundary(pace.util.SOUTH, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, 1, 0),  #
        (1, 4, 0),
        (2, 3, 0),
        (3, 6, 0),
        (4, 5, 0),  #
        (5, 13, 1),
        (6, 7, 0),
        (7, 12, 1),
        (8, 9, 0),  #
        (9, 12, 0),
        (10, 11, 0),
        (11, 14, 0),
        (12, 13, 0),  #
        (13, 21, 1),
        (14, 15, 0),
        (15, 20, 1),
        (16, 17, 0),  #
        (17, 20, 0),
        (18, 19, 0),
        (19, 22, 0),
        (20, 21, 0),  #
        (21, 5, 1),
        (22, 23, 0),
        (23, 4, 1),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_right_edge(
    partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations
):
    edge = partitioner_2_by_2.boundary(pace.util.EAST, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [(0, 1, 0), (1, 3, 1), (2, 3, 0), (3, 5, 1), (4, 5, 0), (5, 1, 1)],
)
@pytest.mark.cpu_only
def test_1_by_1_right_edge(
    partitioner_1_by_1, from_rank, to_rank, n_clockwise_rotations
):
    edge = partitioner_1_by_1.boundary(pace.util.EAST, from_rank)
    assert edge.from_rank == from_rank
    assert edge.to_rank == to_rank
    assert edge.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize("from_rank", [0, 1, 2, 3, 4, 5])
@pytest.mark.cpu_only
def test_1_by_1_top_left_corner(partitioner_1_by_1, from_rank):
    corner = partitioner_1_by_1.boundary(pace.util.NORTHWEST, from_rank)
    assert corner is None


@pytest.mark.parametrize("from_rank", [0, 1, 2, 3, 4, 5])
@pytest.mark.cpu_only
def test_1_by_1_top_right_corner(partitioner_1_by_1, from_rank):
    corner = partitioner_1_by_1.boundary(pace.util.NORTHEAST, from_rank)
    assert corner is None


@pytest.mark.parametrize("from_rank", [0, 1, 2, 3, 4, 5])
@pytest.mark.cpu_only
def test_1_by_1_bottom_left_corner(partitioner_1_by_1, from_rank):
    corner = partitioner_1_by_1.boundary(pace.util.SOUTHWEST, from_rank)
    assert corner is None


@pytest.mark.parametrize("from_rank", [0, 1, 2, 3, 4, 5])
@pytest.mark.cpu_only
def test_1_by_1_bottom_right_corner(partitioner_1_by_1, from_rank):
    corner = partitioner_1_by_1.boundary(pace.util.SOUTHEAST, from_rank)
    assert corner is None


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, 18, 1),  #
        (1, 2, 0),
        (2, None, None),
        (3, 10, 3),
        (4, 3, 0),  #
        (5, 6, 0),
        (6, None, None),
        (7, 8, 0),
        (8, 2, 1),  #
        (9, 10, 0),
        (10, None, None),
        (11, 18, 3),
        (12, 11, 0),  #
        (13, 14, 0),
        (14, None, None),
        (15, 16, 0),
        (16, 10, 1),  #
        (17, 18, 0),
        (18, None, None),
        (19, 2, 3),
        (20, 19, 0),  #
        (21, 22, 0),
        (22, None, None),
        (23, 0, 0),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_top_left_corner(
    partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations
):
    corner = partitioner_2_by_2.boundary(pace.util.NORTHWEST, from_rank)
    if to_rank is None:
        assert corner is None
    else:
        assert corner.from_rank == from_rank
        assert corner.to_rank == to_rank
        assert corner.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "layout, boundary_type, from_rank, to_rank",
    (
        ((1, 1), pace.util.WEST, 0, 0),
        ((1, 1), pace.util.EAST, 0, 0),
        ((1, 1), pace.util.NORTH, 0, 0),
        ((1, 1), pace.util.SOUTH, 0, 0),
        ((2, 2), pace.util.WEST, 0, 1),
        ((2, 2), pace.util.EAST, 0, 1),
        ((2, 2), pace.util.NORTH, 0, 2),
        ((2, 2), pace.util.SOUTH, 0, 2),
        ((2, 2), pace.util.WEST, 3, 2),
        ((2, 2), pace.util.EAST, 3, 2),
        ((2, 2), pace.util.NORTH, 3, 1),
        ((2, 2), pace.util.SOUTH, 3, 1),
    ),
)
@pytest.mark.cpu_only
def test_tile_boundary(layout, boundary_type, from_rank, to_rank):
    tile = pace.util.TilePartitioner(layout)
    boundary = tile.boundary(boundary_type, from_rank)
    assert boundary.from_rank == from_rank
    assert boundary.to_rank == to_rank
    assert boundary.n_clockwise_rotations == 0


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, 3, 0),  #
        (1, 6, 0),
        (2, 8, 3),
        (3, None, None),
        (4, 7, 0),  #
        (5, 12, 1),
        (6, 9, 0),
        (7, None, None),
        (8, 11, 0),  #
        (9, 14, 0),
        (10, 16, 3),
        (11, None, None),
        (12, 15, 0),  #
        (13, 20, 1),
        (14, 17, 0),
        (15, None, None),
        (16, 19, 0),  #
        (17, 22, 0),
        (18, 0, 3),
        (19, None, None),
        (20, 23, 0),  #
        (21, 4, 1),
        (22, 1, 0),
        (23, None, None),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_top_right_corner(
    partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations
):
    corner = partitioner_2_by_2.boundary(pace.util.NORTHEAST, from_rank)
    if to_rank is None:
        assert corner is None
    else:
        assert corner.from_rank == from_rank
        assert corner.to_rank == to_rank
        assert corner.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, None, None),  #
        (1, 22, 0),
        (2, 19, 1),
        (3, 0, 0),
        (4, None, None),  #
        (5, 23, 3),
        (6, 1, 0),
        (7, 4, 0),
        (8, None, None),  #
        (9, 6, 0),
        (10, 3, 1),
        (11, 8, 0),
        (12, None, None),  #
        (13, 7, 3),
        (14, 9, 0),
        (15, 12, 0),
        (16, None, None),  #
        (17, 14, 0),
        (18, 11, 1),
        (19, 16, 0),
        (20, None, None),  #
        (21, 15, 3),
        (22, 17, 0),
        (23, 20, 0),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_bottom_left_corner(
    partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations
):
    corner = partitioner_2_by_2.boundary(pace.util.SOUTHWEST, from_rank)
    if to_rank is None:
        assert corner is None
    else:
        assert corner.from_rank == from_rank
        assert corner.to_rank == to_rank
        assert corner.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize(
    "from_rank, to_rank, n_clockwise_rotations",
    [
        (0, 23, 0),  #
        (1, None, None),
        (2, 1, 0),
        (3, 4, 0),
        (4, 21, 3),  #
        (5, None, None),
        (6, 5, 0),
        (7, 13, 1),
        (8, 7, 0),  #
        (9, None, None),
        (10, 9, 0),
        (11, 12, 0),
        (12, 5, 3),  #
        (13, None, None),
        (14, 13, 0),
        (15, 21, 1),
        (16, 15, 0),  #
        (17, None, None),
        (18, 17, 0),
        (19, 20, 0),
        (20, 13, 3),  #
        (21, None, None),
        (22, 21, 0),
        (23, 5, 1),
    ],
)
@pytest.mark.cpu_only
def test_2_by_2_bottom_right_corner(
    partitioner_2_by_2, from_rank, to_rank, n_clockwise_rotations
):
    corner = partitioner_2_by_2.boundary(pace.util.SOUTHEAST, from_rank)
    if to_rank is None:
        assert corner is None
    else:
        assert corner.from_rank == from_rank
        assert corner.to_rank == to_rank
        assert corner.n_clockwise_rotations == n_clockwise_rotations


def test_boundary_returns_correct_boundary_type():
    tile = pace.util.TilePartitioner((3, 3))
    partitioner = pace.util.CubedSpherePartitioner(tile)
    for boundary_type in pace.util.BOUNDARY_TYPES:
        boundary = partitioner.boundary(boundary_type, rank=4)  # center face
        assert boundary.boundary_type == boundary_type


# rank 42 is tile 4 (5), subrank 6, so top-left corner
# left is tile 2 top-left corner 1 rotations, up is tile 0 top-left corner 3 rotations
# rank 0 is tile 0 subrank 0
# left is tile 4 top-right corner, 1 rotation
# bottom is tile 5 top-left corner, 0 rotations
@pytest.mark.parametrize(
    "boundary_type, from_rank, to_rank, n_clockwise_rotations",
    [
        (pace.util.WEST, 0, 4 * 9 + 8, 1),
        (pace.util.SOUTH, 0, 5 * 9 + 6, 0),
        (pace.util.WEST, 42, 2 * 9 + 6, 1),
        (pace.util.NORTH, 42, 6, 3),
    ],
)
@pytest.mark.cpu_only
def test_3_by_3_difficult_cases(
    partitioner_3_by_3, boundary_type, from_rank, to_rank, n_clockwise_rotations
):
    corner = partitioner_3_by_3.boundary(boundary_type, from_rank)
    if to_rank is None:
        assert corner is None
    else:
        assert corner.from_rank == from_rank
        assert corner.to_rank == to_rank
        assert corner.n_clockwise_rotations == n_clockwise_rotations


@pytest.mark.parametrize("layout", [(1, 1), (2, 2), (4, 4)])
@pytest.mark.cpu_only
def test_edge_boundaries_pair(layout, subtests):
    order = [pace.util.WEST, pace.util.NORTH, pace.util.EAST, pace.util.SOUTH]
    tile = pace.util.TilePartitioner(layout)
    partitioner = pace.util.CubedSpherePartitioner(tile)
    for rank in range(partitioner.total_ranks):
        for boundary_type in pace.util.EDGE_BOUNDARY_TYPES:
            with subtests.test(rank=rank, boundary_type=boundary_type):
                out_boundary = partitioner.boundary(boundary_type, rank)
                in_boundary = partitioner.boundary(
                    rotate(
                        boundary_type, 2 - out_boundary.n_clockwise_rotations, order
                    ),
                    out_boundary.to_rank,
                )
                assert out_boundary.to_rank == in_boundary.from_rank
                assert in_boundary.to_rank == out_boundary.from_rank
                assert (
                    in_boundary.n_clockwise_rotations % 4
                    == -out_boundary.n_clockwise_rotations % 4
                )


@pytest.mark.parametrize("layout", [(1, 1), (2, 2), (4, 4)])
@pytest.mark.cpu_only
def test_corner_boundaries_pair(layout, subtests):
    order = [
        pace.util.NORTHWEST,
        pace.util.NORTHEAST,
        pace.util.SOUTHEAST,
        pace.util.SOUTHWEST,
    ]
    tile = pace.util.TilePartitioner(layout)
    partitioner = pace.util.CubedSpherePartitioner(tile)
    for rank in range(partitioner.total_ranks):
        for boundary_type in pace.util.CORNER_BOUNDARY_TYPES:
            with subtests.test(rank=rank, boundary_type=boundary_type):
                out_boundary = partitioner.boundary(boundary_type, rank)
                if out_boundary is not None:
                    in_boundary = partitioner.boundary(
                        rotate(
                            boundary_type, 2 - out_boundary.n_clockwise_rotations, order
                        ),
                        out_boundary.to_rank,
                    )
                    assert out_boundary.to_rank == in_boundary.from_rank
                    assert in_boundary.to_rank == out_boundary.from_rank
                    assert (
                        in_boundary.n_clockwise_rotations % 4
                        == -out_boundary.n_clockwise_rotations % 4
                    )


def rotate(boundary_type, n_clockwise_rotations, order):
    target_index = (order.index(boundary_type) + n_clockwise_rotations) % len(order)
    return order[target_index]
