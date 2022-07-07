import copy
import dataclasses

import numpy as np

import pace.util


def test_deepcopy_copy_is_editable_by_view():
    nx, ny, nz = 12, 12, 15
    quantity = pace.util.Quantity(
        np.zeros([nx, ny, nz]),
        origin=(0, 0, 0),
        extent=(nx, ny, nz),
        dims=["x", "y", "z"],
        units="",
    )
    quantity_copy = copy.deepcopy(quantity)
    # assertion below is only valid if we're overwriting the entire data through view
    assert np.product(quantity_copy.view[:].shape) == np.product(
        quantity_copy.data.shape
    )
    quantity_copy.view[:] = 1.0
    np.testing.assert_array_equal(quantity.data, 0.0)
    np.testing.assert_array_equal(quantity_copy.data, 1.0)


def test_deepcopy_copy_is_editable_by_data():
    nx, ny, nz = 12, 12, 15
    quantity = pace.util.Quantity(
        np.zeros([nx, ny, nz]),
        origin=(0, 0, 0),
        extent=(nx, ny, nz),
        dims=["x", "y", "z"],
        units="",
    )
    quantity_copy = copy.deepcopy(quantity)
    quantity_copy.data[:] = 1.0
    np.testing.assert_array_equal(quantity.data, 0.0)
    np.testing.assert_array_equal(quantity_copy.data, 1.0)


def test_deepcopy_of_dataclass_is_editable_by_data():
    nx, ny, nz = 12, 12, 15
    quantity = pace.util.Quantity(
        np.zeros([nx, ny, nz]),
        origin=(0, 0, 0),
        extent=(nx, ny, nz),
        dims=["x", "y", "z"],
        units="",
    )
    quantity_copy = copy.deepcopy(quantity)
    quantity_copy.data[:] = 1.0

    @dataclasses.dataclass
    class MyClass:
        quantity: pace.util.Quantity

    instance = MyClass(quantity)
    instance_copy = copy.deepcopy(instance)
    instance_copy.quantity.data[:] = 1.0
    np.testing.assert_array_equal(instance.quantity.data, 0.0)
    np.testing.assert_array_equal(instance_copy.quantity.data, 1.0)
