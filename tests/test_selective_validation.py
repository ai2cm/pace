from unittest.mock import MagicMock

import numpy as np
import pytest

from fv3core.testing.validation import get_selective_class


class DummyClass:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        pass


def check_selective_region_and_values(instance, name, array, domain):
    validated_gridcells = np.product(domain)
    total_gridcells = np.product(array.shape)
    assert np.sum(np.isnan(array)) == total_gridcells - validated_gridcells
    validation_subset = instance.subset_output(name, array)
    assert validation_subset.shape == domain
    assert (
        np.sum(np.isnan(validation_subset)) == 0
    ), "validation slice does not match the values set to nan"


def ensure_no_selective_validation(instance, name, array):
    assert np.sum(np.isnan(array)) == 0
    assert instance.subset_output(name, array).shape == array.shape


@pytest.mark.parametrize(
    "shape1, origin1, domain1, shape2, origin2, domain2",
    [
        pytest.param(
            (3, 3, 3),
            (0, 0, 0),
            (3, 3, 3),
            (3, 3),
            (0, 0),
            (3, 3),
            id="whole_array_valid",
        ),
        pytest.param(
            (3, 3, 3),
            (0, 0, 0),
            (0, 0, 0),
            (3, 3),
            (0, 0),
            (0, 0),
            id="whole_array_invalid",
        ),
        pytest.param(
            (4, 4, 4), (1, 2, 0), (2, 1, 3), (4, 4), (1, 0), (2, 3), id="some_valid"
        ),
    ],
)
@pytest.mark.parametrize(
    "all_arg_names, selective_arg_names_shape1, selective_arg_names_shape2",
    [
        pytest.param(["a"], [], [], id="no_selective_validation"),
        pytest.param(["a"], ["a"], [], id="one_out_of_one_arg"),
        pytest.param(["a", "b", "c"], ["a"], [], id="one_out_of_three_args"),
        pytest.param(
            ["a", "b", "c"], ["a", "b"], [], id="two_out_of_three_args_same_shape"
        ),
        pytest.param(
            ["a", "b", "c"], ["a"], ["b"], id="two_out_of_three_args_different_shapes"
        ),
    ],
)
def test_selective_validation(
    all_arg_names,
    selective_arg_names_shape1,
    selective_arg_names_shape2,
    shape1,
    origin1,
    domain1,
    shape2,
    origin2,
    domain2,
):
    """
    Tests behavior of the returned class from get_selective_class. Tests that it:
        - sets the correct number of gridcells to NaN
        - returns a validation subset that is the right shape, and has no nans
        - subsets only the arguments it was told to, and leaves others untouched
            with no nans
        - uses the correct domain and origin for each variable if they differ
    """
    origin_domain_func1 = MagicMock(return_value=(origin1, domain1))
    origin_domain_func2 = MagicMock(return_value=(origin2, domain2))
    name_to_function = {}

    kwargs = {name: np.zeros(shape1) for name in all_arg_names}
    for name in selective_arg_names_shape1:
        name_to_function[name] = origin_domain_func1
        kwargs[name] = np.zeros(shape1)
    for name in selective_arg_names_shape2:
        name_to_function[name] = origin_domain_func2
        kwargs[name] = np.zeros(shape2)
    Wrapped = get_selective_class(DummyClass, name_to_function)
    instance = Wrapped()
    instance(**kwargs)
    for name, array in kwargs.items():
        if name in selective_arg_names_shape1:
            check_selective_region_and_values(instance, name, array, domain1)
        elif name in selective_arg_names_shape2:
            check_selective_region_and_values(instance, name, array, domain2)
        else:
            ensure_no_selective_validation(instance, name, array)
    assert origin_domain_func1.call_count == len(selective_arg_names_shape1)
    assert origin_domain_func2.call_count == len(selective_arg_names_shape2)
