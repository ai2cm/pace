from typing import Sequence

import numpy as np
import pytest
from gt4py import gtscript

import pace.dsl.stencil
from pace.dsl.typing import Index3D
from pace.util import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
)


@pytest.mark.parametrize("domain, n_halo", [pytest.param((4, 4, 4), 3, id="3_halo")])
@pytest.mark.parametrize("south_edge", [True, False])
@pytest.mark.parametrize("north_edge", [True, False])
@pytest.mark.parametrize("west_edge", [True, False])
@pytest.mark.parametrize("east_edge", [True, False])
@pytest.mark.parametrize(
    "origin_offset, domain_offset, i_start, i_end, j_start, j_end",
    [
        pytest.param(
            (0, 0),
            (0, 0),
            gtscript.I[0],
            gtscript.I[-1],
            gtscript.J[0],
            gtscript.J[-1],
            id="compute_domain",
        ),
        pytest.param(
            (-1, -1),
            (2, 2),
            gtscript.I[0] + 1,
            gtscript.I[-1] - 1,
            gtscript.J[0] + 1,
            gtscript.J[-1] - 1,
            id="compute_domain_plus_one_halo",
        ),
        pytest.param(
            (-1, 0),
            (2, 0),
            gtscript.I[0] + 1,
            gtscript.I[-1] - 1,
            gtscript.J[0],
            gtscript.J[-1],
            id="compute_domain_plus_one_x_halo",
        ),
    ],
)
def test_axis_offsets(
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    origin_offset: Index3D,
    domain_offset: Index3D,
    i_start,
    i_end,
    j_start,
    j_end,
):
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    origin = (n_halo, n_halo, 0)
    call_origin = tuple(
        compute + offset for (compute, offset) in zip(origin, origin_offset)
    )
    call_domain = tuple(
        compute + offset for (compute, offset) in zip(domain, domain_offset)
    )
    axis_offsets = grid.axis_offsets(call_origin, call_domain)
    if west_edge:
        assert axis_offsets["i_start"] == i_start
    else:
        assert axis_offsets["i_start"] == gtscript.I[0] - np.iinfo(np.int16).max
    if east_edge:
        assert axis_offsets["i_end"] == i_end
    else:
        assert axis_offsets["i_end"] == gtscript.I[-1] + np.iinfo(np.int16).max
    if south_edge:
        assert axis_offsets["j_start"] == j_start
    else:
        assert axis_offsets["j_start"] == gtscript.J[0] - np.iinfo(np.int16).max
    if north_edge:
        assert axis_offsets["j_end"] == j_end
    else:
        assert axis_offsets["j_end"] == gtscript.J[-1] + np.iinfo(np.int16).max


@pytest.mark.parametrize(
    "domain, n_halo, add, origin_full",
    [
        pytest.param((4, 4, 4), 3, (0, 0, 0), (0, 0, 0), id="3_halo"),
        pytest.param((4, 4, 4), 3, (1, 0, 0), (1, 0, 0), id="3_halo_add_i"),
        pytest.param((4, 4, 4), 3, (-1, 0, 0), (-1, 0, 0), id="3_halo_add_i_negative"),
        pytest.param((4, 4, 4), 3, (0, 1, 0), (0, 1, 0), id="3_halo_add_j"),
        pytest.param((4, 4, 4), 3, (0, 0, 1), (0, 0, 1), id="3_halo_add_k"),
        pytest.param((4, 4, 4), 3, (5, 3, 1), (5, 3, 1), id="3_halo_add_ijk"),
        pytest.param(
            (4, 4, 4),
            3,
            (-5, -3, -1),
            (-5, -3, -1),
            id="3_halo_add_ijk_negative",
        ),
    ],
)
@pytest.mark.parametrize(
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
def test_origin_full(
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    origin_full: Index3D,
):
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    result = grid.origin_full(add=add)
    assert result == origin_full


@pytest.mark.parametrize(
    "domain, n_halo, add, origin_compute",
    [
        pytest.param((4, 4, 4), 3, (0, 0, 0), (3, 3, 0), id="3_halo"),
        pytest.param((4, 4, 4), 3, (1, 0, 0), (4, 3, 0), id="3_halo_add_i"),
        pytest.param((4, 4, 4), 3, (-1, 0, 0), (2, 3, 0), id="3_halo_add_i_negative"),
        pytest.param((4, 4, 4), 3, (0, 1, 0), (3, 4, 0), id="3_halo_add_j"),
        pytest.param((4, 4, 4), 3, (0, 0, 1), (3, 3, 1), id="3_halo_add_k"),
        pytest.param((4, 4, 4), 3, (5, 3, 1), (8, 6, 1), id="3_halo_add_ijk"),
        pytest.param(
            (4, 4, 4),
            3,
            (-5, -3, -1),
            (-2, 0, -1),
            id="3_halo_add_ijk_negative",
        ),
    ],
)
@pytest.mark.parametrize(
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
def test_origin_compute(
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    origin_compute: Index3D,
):
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    result = grid.origin_compute(add=add)
    assert result == origin_compute


@pytest.mark.parametrize(
    "domain, n_halo, add, domain_full",
    [
        pytest.param((3, 4, 5), 3, (0, 0, 0), (9, 10, 5), id="3_halo"),
        pytest.param((2, 2, 2), 3, (0, 0, 0), (8, 8, 2), id="3_halo_smaller_domain"),
        pytest.param((2, 3, 4), 0, (0, 0, 0), (2, 3, 4), id="no_halo"),
    ],
)
@pytest.mark.parametrize(
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
def test_domain_full(
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    domain_full: Index3D,
):
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    result = grid.domain_full(add=add)
    assert result == domain_full


@pytest.mark.parametrize(
    "domain, n_halo, add, domain_compute",
    [
        pytest.param((3, 4, 5), 3, (0, 0, 0), (3, 4, 5), id="3_halo"),
        pytest.param((3, 4, 6), 1, (0, 0, 0), (3, 4, 6), id="1_halo_2_buffer"),
        pytest.param((2, 2, 2), 3, (0, 0, 0), (2, 2, 2), id="3_halo_smaller_domain"),
        pytest.param((2, 3, 4), 0, (0, 0, 0), (2, 3, 4), id="no_halo"),
    ],
)
@pytest.mark.parametrize(
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
def test_domain_compute(
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    domain_compute: Index3D,
):
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    result = grid.domain_compute(add=add)
    assert result == domain_compute


@pytest.mark.parametrize(
    "n_halo, domain, dims, halos, origin_expected, domain_expected",
    [
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (3, 3, 0),
            (4, 4, 7),
            id="compute_domain",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            tuple(),
            (3, 3, 0),
            (4, 4, 7),
            id="compute_domain_no_halo_arg",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [Z_DIM, Y_DIM, X_DIM],
            (0, 0, 0),
            (0, 3, 3),
            (7, 4, 4),
            id="reverse_compute_domain",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Z_DIM],
            (0, 0),
            (3, 0),
            (4, 7),
            id="xz_compute_domain",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (1, 0, 0),
            (2, 3, 0),
            (6, 4, 7),
            id="x_halo",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 1, 0),
            (3, 2, 0),
            (4, 6, 7),
            id="y_halo",
        ),
        # z_halo is an unrealistic case, but the API supports it
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 0, 1),
            (3, 3, -1),
            (4, 4, 9),
            id="z_halo",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (2, 2),
            (1, 1, 0),
            (8, 8, 7),
            id="xy_2_halo",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM],
            (2, 2),
            (1, 1),
            (8, 8),
            id="xy_2_halo_no_zdim",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (3, 3, 0),
            (5, 4, 7),
            id="x_interface",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            (0, 0, 0),
            (3, 3, 0),
            (4, 5, 7),
            id="y_interface",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            (0, 0, 0),
            (3, 3, 0),
            (4, 4, 8),
            id="z_interface",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            (0, 3),
            (3, 0, 0),
            (5, 10, 7),
            id="x_interface_y_halo",
        ),
        pytest.param(
            1,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (1, 1, 0),
            (4, 4, 7),
            id="compute_domain_smaller_origin",
        ),
        pytest.param(
            3,
            (2, 3, 6),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (3, 3, 0),
            (2, 3, 6),
            id="compute_domain_smaller_domain",
        ),
    ],
)
@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
def test_get_origin_domain(
    n_halo: int,
    domain: Index3D,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    dims: Sequence[str],
    halos: Sequence[int],
    origin_expected: Sequence[int],
    domain_expected: Sequence[int],
):
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    origin_out, domain_out = grid.get_origin_domain(dims, halos)
    assert origin_out == origin_expected
    assert domain_out == domain_expected


@pytest.mark.parametrize(
    "n_halo, domain, dims, halos, origin_expected, domain_expected",
    [
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (3, 3, 0),
            (4, 4, 7),
            id="compute_domain",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            tuple(),
            (3, 3, 0),
            (4, 4, 7),
            id="compute_domain_no_halo_arg",
        ),
    ],
)
@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
def test_get_origin_domain_restricted_vertical(
    n_halo: int,
    domain: Index3D,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    dims: Sequence[str],
    halos: Sequence[int],
    origin_expected: Sequence[int],
    domain_expected: Sequence[int],
):
    k_start = 2
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    grid = grid.restrict_vertical(k_start=k_start)
    origin_out, domain_out = grid.get_origin_domain(dims, halos)
    assert origin_out[2] == origin_expected[2] + k_start
    assert domain_out[2] == domain_expected[2] - k_start
    assert origin_out[:2] == origin_expected[:2]
    assert domain_out[:2] == domain_expected[:2]


@pytest.mark.parametrize(
    "n_halo, domain, dims, halos, shape_expected",
    [
        pytest.param(
            3,
            (5, 6, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (8, 9, 7),
            id="compute",
        ),
        pytest.param(
            3,
            (5, 6, 7),
            [X_DIM, Y_DIM, Z_DIM],
            tuple(),
            (8, 9, 7),
            id="compute_empty_halo",
        ),
        pytest.param(
            3,
            (5, 6, 7),
            [Z_DIM, Y_DIM, X_DIM],
            (0, 0, 0),
            (7, 9, 8),
            id="compute_reverse",
        ),
        pytest.param(
            0,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (4, 4, 7),
            id="no_halos_anywhere",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_INTERFACE_DIM, Y_DIM, Z_DIM],
            (0, 0, 0),
            (8, 7, 7),
            id="x_interface",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_INTERFACE_DIM, Z_DIM],
            (0, 0, 0),
            (7, 8, 7),
            id="y_interface",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            (0, 0, 0),
            (7, 7, 8),
            id="z_interface",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (3, 3),
            (10, 10, 7),
            id="halos_required",
        ),
        pytest.param(
            3,
            (4, 4, 7),
            [X_DIM, Y_DIM, Z_DIM],
            (0, 3),
            (7, 10, 7),
            id="y_halos_required",
        ),
    ],
)
@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
def test_get_shape(
    n_halo: int,
    domain: Index3D,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    dims: Sequence[str],
    halos: Sequence[int],
    shape_expected: Sequence[int],
):
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    shape_out = grid.get_shape(dims, halos)
    assert shape_out == shape_expected


@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
@pytest.mark.parametrize("n_halo", [0, 2])
def test_restrict_vertical_defaults(
    n_halo, south_edge, north_edge, west_edge, east_edge
):
    domain = (3, 4, 10)
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    restricted = grid.restrict_vertical()
    assert restricted.origin[2] == 0
    assert restricted.domain[2] == 10


@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
@pytest.mark.parametrize("n_halo", [0, 2])
def test_restrict_vertical_default_domain(
    n_halo, south_edge, north_edge, west_edge, east_edge
):
    domain = (3, 4, 10)
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    restricted = grid.restrict_vertical(k_start=2)
    assert restricted.origin[2] == 2
    assert restricted.domain[2] == 8


@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
@pytest.mark.parametrize("n_halo", [0, 2])
def test_restrict_vertical_max_shape(
    n_halo, south_edge, north_edge, west_edge, east_edge
):
    domain = (3, 4, 10)
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    restricted = grid.restrict_vertical(k_start=2)
    # max_shape should still include lowest points
    assert restricted.max_shape == grid.max_shape


@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
@pytest.mark.parametrize("n_halo", [0, 2])
@pytest.mark.parametrize("k_start, nk", [(0, 10), (2, 8)])
def test_restrict_vertical(
    n_halo, south_edge, north_edge, west_edge, east_edge, k_start, nk
):
    domain = (3, 4, 10)
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    restricted = grid.restrict_vertical(k_start=k_start, nk=nk)
    assert restricted.origin[2] == k_start
    assert restricted.domain[2] == nk


@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
@pytest.mark.parametrize("n_halo", [0, 2])
@pytest.mark.parametrize("k_start, nk", [(0, 10), (2, 8)])
@pytest.mark.parametrize("second_k_start, second_nk", [(0, 8), (2, 4)])
def test_restrict_vertical_twice(
    n_halo,
    south_edge,
    north_edge,
    west_edge,
    east_edge,
    k_start,
    nk,
    second_k_start,
    second_nk,
):
    domain = (3, 4, 10)
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    restricted = grid.restrict_vertical(k_start=k_start, nk=nk)
    second_restricted = restricted.restrict_vertical(
        k_start=second_k_start, nk=second_nk
    )
    assert second_restricted.origin[2] == k_start + second_k_start
    assert second_restricted.domain[2] == second_nk


@pytest.mark.parametrize(
    # edges shouldn't matter for this test, but let's make sure behaviors
    # are all the same
    "south_edge, north_edge, west_edge, east_edge",
    [
        pytest.param(True, True, True, True, id="all_edges"),
        pytest.param(False, False, False, False, id="no_edges"),
        pytest.param(True, False, False, True, id="southeast_corner"),
    ],
)
@pytest.mark.parametrize("n_halo", [0, 2])
@pytest.mark.parametrize("k_start, nk", [(-2, 10), (2, 10)])
def test_restrict_vertical_raises(
    n_halo, south_edge, north_edge, west_edge, east_edge, k_start, nk
):
    domain = (3, 4, 10)
    grid = pace.dsl.stencil.GridIndexing(
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    with pytest.raises(ValueError):
        grid.restrict_vertical(k_start=k_start, nk=nk)
