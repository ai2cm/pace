import numpy as np
import pytest

import pace.util


@pytest.fixture
def units():
    return "m"


@pytest.fixture
def dims(request):
    return [pace.util.X_DIM, pace.util.Y_DIM]


@pytest.fixture
def shape(request):
    return request.param


@pytest.fixture
def origin(request):
    return request.param


@pytest.fixture
def extent(request):
    return request.param


@pytest.fixture
def layout(request):
    return request.param


@pytest.fixture
def quantity(shape, dims, units, origin, extent, numpy):
    return pace.util.Quantity(
        numpy.zeros(shape), dims=dims, units=units, origin=origin, extent=extent
    )


@pytest.fixture
def tile_partitioner(layout):
    return pace.util.TilePartitioner(layout)


@pytest.fixture
def rank(request):
    return request.param


@pytest.mark.parametrize(
    "shape, origin, extent, n_halo",
    [
        ((6, 6), (2, 2), (2, 2), 2),
        ((6, 6), (2, 2), (2, 2), 1),
        ((3, 3), (1, 1), (1, 1), 1),
        ((6, 6), (3, 2), (2, 2), 1),
    ],
    indirect=["shape", "origin", "extent"],
)
@pytest.mark.parametrize("rank, layout", [(0, (1, 1))])
@pytest.mark.parametrize("direction", ["x", "y"])
def test_fill_scalar_corners_copies_from_halo(
    quantity, direction, tile_partitioner, rank, n_halo
):
    quantity.data[:] = 0
    # put nans in corners
    quantity.view.southwest[-n_halo:0, -n_halo:0] = quantity.np.nan
    quantity.view.southeast[0:n_halo, -n_halo:0] = quantity.np.nan
    quantity.view.northwest[-n_halo:0, 0:n_halo] = quantity.np.nan
    quantity.view.northeast[0:n_halo, 0:n_halo] = quantity.np.nan
    quantity.view[:] = 2
    pace.util.fill_scalar_corners(
        quantity=quantity,
        direction=direction,
        tile_partitioner=tile_partitioner,
        rank=rank,
        n_halo=n_halo,
    )
    assert quantity.np.sum(quantity.np.isnan(quantity.data)) == 0
    assert quantity.np.all(quantity.view[:] == 2)  # should be unchanged
    quantity.np.testing.assert_array_equal(
        quantity.view.southwest[-n_halo:0, -n_halo:0], 0
    )
    quantity.np.testing.assert_array_equal(
        quantity.view.southeast[0:n_halo, -n_halo:0], 0
    )
    quantity.np.testing.assert_array_equal(
        quantity.view.northwest[-n_halo:0, 0:n_halo], 0
    )
    quantity.np.testing.assert_array_equal(
        quantity.view.northeast[0:n_halo, 0:n_halo], 0
    )


@pytest.mark.parametrize(
    "quantity_in, direction, layout, rank, n_halo, reference",
    [
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10, 11],
                        [12, 13, 14, 15, 16, 17],
                        [18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29],
                        [30, 31, 32, 33, 34, 35],
                    ]
                ).T,
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(2, 2),
            ),
            "x",
            (1, 1),
            1,
            2,
            np.array(
                [
                    [18, 12, 2, 3, 17, 23],
                    [19, 13, 8, 9, 16, 22],
                    [12, 13, 14, 15, 16, 17],
                    [18, 19, 20, 21, 22, 23],
                    [13, 19, 26, 27, 22, 16],
                    [12, 18, 32, 33, 23, 17],
                ]
            ).T,
            id="all_corners_x",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 0, 3, 4, 5],
                        [6, 7, 8, 1, 9, 10, 11],
                        [12, 13, 14, 2, 15, 16, 17],
                        [0, 1, 2, 3, 4, 5, 6],
                        [18, 19, 20, 4, 21, 22, 23],
                        [24, 25, 26, 5, 27, 28, 29],
                        [30, 31, 32, 6, 33, 34, 35],
                    ]
                ).T,
                dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
                units="m",
                origin=(2, 2),
                extent=(3, 3),
            ),
            "x",
            (1, 1),
            1,
            2,
            np.array(
                [
                    [18, 0, 2, 0, 3, 6, 23],
                    [19, 1, 8, 1, 9, 5, 22],
                    [12, 13, 14, 2, 15, 16, 17],
                    [0, 1, 2, 3, 4, 5, 6],
                    [18, 19, 20, 4, 21, 22, 23],
                    [13, 1, 26, 5, 27, 5, 16],
                    [12, 0, 32, 6, 33, 6, 17],
                ]
            ).T,
            id="all_corners_x_interfaces",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10, 11],
                        [12, 13, 14, 15, 16, 17],
                        [0, 1, 2, 4, 5, 6],
                        [18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29],
                        [30, 31, 32, 33, 34, 35],
                    ]
                ).T,
                dims=[pace.util.X_DIM, pace.util.Y_INTERFACE_DIM],
                units="m",
                origin=(2, 2),
                extent=(2, 3),
            ),
            "x",
            (1, 1),
            1,
            2,
            np.array(
                [
                    [18, 0, 2, 3, 6, 23],
                    [19, 1, 8, 9, 5, 22],
                    [12, 13, 14, 15, 16, 17],
                    [0, 1, 2, 4, 5, 6],
                    [18, 19, 20, 21, 22, 23],
                    [13, 1, 26, 27, 5, 16],
                    [12, 0, 32, 33, 6, 17],
                ]
            ).T,
            id="all_corners_x_one_iface_dim",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 0, 3, 4, 5],
                        [6, 7, 8, 1, 9, 10, 11],
                        [12, 13, 14, 2, 15, 16, 17],
                        [0, 1, 2, 3, 4, 5, 6],
                        [18, 19, 20, 4, 21, 22, 23],
                        [24, 25, 26, 5, 27, 28, 29],
                        [30, 31, 32, 6, 33, 34, 35],
                    ]
                ).T,
                dims=[pace.util.X_INTERFACE_DIM, pace.util.Y_INTERFACE_DIM],
                units="m",
                origin=(2, 2),
                extent=(3, 3),
            ),
            "y",
            (1, 1),
            1,
            2,
            np.array(
                [
                    [3, 9, 2, 0, 3, 8, 2],
                    [0, 1, 8, 1, 9, 1, 0],
                    [12, 13, 14, 2, 15, 16, 17],
                    [0, 1, 2, 3, 4, 5, 6],
                    [18, 19, 20, 4, 21, 22, 23],
                    [6, 5, 26, 5, 27, 5, 6],
                    [33, 27, 32, 6, 33, 26, 32],
                ]
            ).T,
            id="all_corners_y_interfaces",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10, 11],
                        [12, 13, 14, 15, 16, 17],
                        [18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29],
                        [30, 31, 32, 33, 34, 35],
                    ]
                ).T,
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(2, 2),
            ),
            "x",
            (2, 2),
            2,
            2,
            np.array(
                [
                    [0, 1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16, 17],
                    [18, 19, 20, 21, 22, 23],
                    [13, 19, 26, 27, 28, 29],
                    [12, 18, 32, 33, 34, 35],
                ]
            ).T,
            id="one_corner_x",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10, 11],
                        [12, 13, 14, 15, 16, 17],
                        [18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29],
                        [30, 31, 32, 33, 34, 35],
                    ]
                ).T,
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(2, 2),
            ),
            "y",
            (1, 1),
            1,
            2,
            np.array(
                [
                    [3, 9, 2, 3, 8, 2],
                    [2, 8, 8, 9, 9, 3],
                    [12, 13, 14, 15, 16, 17],
                    [18, 19, 20, 21, 22, 23],
                    [32, 26, 26, 27, 27, 33],
                    [33, 27, 32, 33, 26, 32],
                ]
            ).T,
            id="all_corners_y",
        ),
    ],
    indirect=["layout"],
)
@pytest.mark.cpu_only
def test_fill_corners(
    quantity_in, direction, tile_partitioner, rank, n_halo, reference
):
    pace.util.fill_scalar_corners(
        quantity=quantity_in,
        direction=direction,
        tile_partitioner=tile_partitioner,
        rank=rank,
        n_halo=n_halo,
    )
    quantity_in.np.testing.assert_array_equal(quantity_in.data, reference)
