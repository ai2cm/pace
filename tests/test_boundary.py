import fv3util
import numpy as np


def boundary_data(quantity, boundary_type, n_points, interior=True):
    boundary_slice = fv3util.boundary._get_boundary_slice(
        quantity.dims, quantity.origin, quantity.extent, boundary_type, n_points, interior
    )
    return quantity.data[tuple(boundary_slice)]


def test_boundary_data_1_by_1_array_1_halo():
    quantity = fv3util.Quantity(
        np.random.randn(3, 3),
        dims=[fv3util.Y_DIM, fv3util.X_DIM],
        units='m',
        origin=(1, 1),
        extent=(1, 1)
    )
    for side in ('left', 'right', 'top', 'bottom'):
        assert boundary_data(quantity, side, n_points=1, interior=True) == quantity.data[1, 1]

    assert boundary_data(quantity, 'top', n_points=1, interior=False) == quantity.data[2, 1]
    assert boundary_data(quantity, 'bottom', n_points=1, interior=False) == quantity.data[0, 1]
    assert boundary_data(quantity, 'left', n_points=1, interior=False) == quantity.data[1, 0]
    assert boundary_data(quantity, 'right', n_points=1, interior=False) == quantity.data[1, 2]


def test_boundary_data_2_by_2_array_2_halo():
    quantity = fv3util.Quantity(
        np.random.randn(6, 6),
        dims=[fv3util.Y_DIM, fv3util.X_DIM],
        units='m',
        origin=(2, 2),
        extent=(2, 2)
    )
    for side in ('left', 'right', 'top', 'bottom'):
        np.testing.assert_array_equal(
            boundary_data(quantity, side, n_points=2, interior=True),
            quantity.data[2:4, 2:4]
        )

    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'top', n_points=1, interior=True),
        quantity.data[3:4, 2:4]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'top', n_points=1, interior=False),
        quantity.data[4:5, 2:4]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'top', n_points=2, interior=False),
        quantity.data[4:6, 2:4]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'bottom', n_points=1, interior=True),
        quantity.data[2:3, 2:4]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'bottom', n_points=1, interior=False),
        quantity.data[1:2, 2:4]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'bottom', n_points=2, interior=False),
        quantity.data[0:2, 2:4]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'left', n_points=2, interior=False),
        quantity.data[2:4, 0:2]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'left', n_points=1, interior=True),
        quantity.data[2:4, 2:3]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'left', n_points=1, interior=False),
        quantity.data[2:4, 1:2]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'right', n_points=1, interior=False),
        quantity.data[2:4, 4:5]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'right', n_points=2, interior=False),
        quantity.data[2:4, 4:6]
    )
    quantity.np.testing.assert_array_equal(
        boundary_data(quantity, 'right', n_points=1, interior=True),
        quantity.data[2:4, 3:4]
    )
