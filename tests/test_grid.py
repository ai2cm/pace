import numpy as np
import pytest
from gt4py import gtscript

import fv3core.utils.grid
from fv3core.utils.typing import Index3D


@pytest.mark.parametrize(
    "origin, domain, n_halo", [pytest.param((3, 3, 3), (4, 4, 4), 3, id="3_halo")]
)
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
    origin: Index3D,
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
    grid = fv3core.utils.grid.GridIndexing(
        origin=origin,
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
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
        assert axis_offsets["i_start"] == gtscript.I[0] - np.iinfo(np.int32).max
    if east_edge:
        assert axis_offsets["i_end"] == i_end
    else:
        assert axis_offsets["i_end"] == gtscript.I[-1] + np.iinfo(np.int32).max
    if south_edge:
        assert axis_offsets["j_start"] == j_start
    else:
        assert axis_offsets["j_start"] == gtscript.J[0] - np.iinfo(np.int32).max
    if north_edge:
        assert axis_offsets["j_end"] == j_end
    else:
        assert axis_offsets["j_end"] == gtscript.J[-1] + np.iinfo(np.int32).max


@pytest.mark.parametrize(
    "origin, domain, n_halo, add, origin_full",
    [
        pytest.param((3, 3, 3), (4, 4, 4), 3, (0, 0, 0), (0, 0, 3), id="3_halo"),
        pytest.param((3, 3, 3), (4, 4, 4), 3, (1, 0, 0), (1, 0, 3), id="3_halo_add_i"),
        pytest.param(
            (3, 3, 3), (4, 4, 4), 3, (-1, 0, 0), (-1, 0, 3), id="3_halo_add_i_negative"
        ),
        pytest.param((3, 3, 3), (4, 4, 4), 3, (0, 1, 0), (0, 1, 3), id="3_halo_add_j"),
        pytest.param((3, 3, 3), (4, 4, 4), 3, (0, 0, 1), (0, 0, 4), id="3_halo_add_k"),
        pytest.param(
            (3, 3, 3), (4, 4, 4), 3, (5, 3, 1), (5, 3, 4), id="3_halo_add_ijk"
        ),
        pytest.param(
            (3, 3, 3),
            (4, 4, 4),
            3,
            (-5, -3, -1),
            (-5, -3, 2),
            id="3_halo_add_ijk_negative",
        ),
        pytest.param(
            (4, 5, 6), (4, 4, 4), 3, (0, 0, 0), (1, 2, 6), id="buffer_at_start_of_array"
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
    origin: Index3D,
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    origin_full: Index3D,
):
    grid = fv3core.utils.grid.GridIndexing(
        origin=origin,
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
    "origin, domain, n_halo, add, origin_compute",
    [
        pytest.param((3, 3, 3), (4, 4, 4), 3, (0, 0, 0), (3, 3, 3), id="3_halo"),
        pytest.param((3, 3, 3), (4, 4, 4), 3, (1, 0, 0), (4, 3, 3), id="3_halo_add_i"),
        pytest.param(
            (3, 3, 3), (4, 4, 4), 3, (-1, 0, 0), (2, 3, 3), id="3_halo_add_i_negative"
        ),
        pytest.param((3, 3, 3), (4, 4, 4), 3, (0, 1, 0), (3, 4, 3), id="3_halo_add_j"),
        pytest.param((3, 3, 3), (4, 4, 4), 3, (0, 0, 1), (3, 3, 4), id="3_halo_add_k"),
        pytest.param(
            (3, 3, 3), (4, 4, 4), 3, (5, 3, 1), (8, 6, 4), id="3_halo_add_ijk"
        ),
        pytest.param(
            (3, 3, 3),
            (4, 4, 4),
            3,
            (-5, -3, -1),
            (-2, 0, 2),
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
    origin: Index3D,
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    origin_compute: Index3D,
):
    grid = fv3core.utils.grid.GridIndexing(
        origin=origin,
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
    "origin, domain, n_halo, add, domain_full",
    [
        pytest.param((3, 3, 3), (3, 4, 5), 3, (0, 0, 0), (9, 10, 5), id="3_halo"),
        pytest.param(
            (3, 3, 3), (3, 4, 6), 1, (0, 0, 0), (5, 6, 6), id="1_halo_2_buffer"
        ),
        pytest.param(
            (1, 2, 3), (3, 4, 7), 3, (0, 0, 0), (9, 10, 7), id="123_origin_3_halo"
        ),
        pytest.param(
            (3, 3, 3), (2, 2, 2), 3, (0, 0, 0), (8, 8, 2), id="3_halo_smaller_domain"
        ),
        pytest.param((0, 0, 0), (2, 3, 4), 0, (0, 0, 0), (2, 3, 4), id="no_halo"),
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
    origin: Index3D,
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    domain_full: Index3D,
):
    grid = fv3core.utils.grid.GridIndexing(
        origin=origin,
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
    "origin, domain, n_halo, add, domain_compute",
    [
        pytest.param((3, 3, 3), (3, 4, 5), 3, (0, 0, 0), (3, 4, 5), id="3_halo"),
        pytest.param(
            (3, 3, 3), (3, 4, 6), 1, (0, 0, 0), (3, 4, 6), id="1_halo_2_buffer"
        ),
        pytest.param((1, 2, 3), (3, 4, 7), 3, (0, 0, 0), (3, 4, 7), id="123_halo"),
        pytest.param(
            (3, 3, 3), (2, 2, 2), 3, (0, 0, 0), (2, 2, 2), id="3_halo_smaller_domain"
        ),
        pytest.param((0, 0, 0), (2, 3, 4), 0, (0, 0, 0), (2, 3, 4), id="no_halo"),
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
    origin: Index3D,
    domain: Index3D,
    n_halo: int,
    south_edge: bool,
    north_edge: bool,
    west_edge: bool,
    east_edge: bool,
    add: Index3D,
    domain_compute: Index3D,
):
    grid = fv3core.utils.grid.GridIndexing(
        origin=origin,
        domain=domain,
        n_halo=n_halo,
        south_edge=south_edge,
        north_edge=north_edge,
        west_edge=west_edge,
        east_edge=east_edge,
    )
    result = grid.domain_compute(add=add)
    assert result == domain_compute
