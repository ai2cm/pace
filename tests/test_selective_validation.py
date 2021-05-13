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


@pytest.mark.parametrize(
    "shape, origin, domain",
    [
        pytest.param((3, 3, 3), (0, 0, 0), (3, 3, 3), id="whole_array_valid"),
        pytest.param((3, 3, 3), (0, 0, 0), (0, 0, 0), id="whole_array_invalid"),
        pytest.param((4, 4, 4), (1, 2, 0), (2, 1, 3), id="some_valid"),
    ],
)
@pytest.mark.parametrize(
    "all_arg_names, selective_arg_names",
    [
        pytest.param(["a"], [], id="no_selective_validation"),
        pytest.param(["a"], ["a"], id="one_out_of_one_arg"),
        pytest.param(["a", "b", "c"], ["b"], id="one_out_of_three_args"),
        pytest.param(["a", "b", "c"], ["a", "b"], id="two_out_of_three_args"),
    ],
)
def test_wrapped_class(all_arg_names, selective_arg_names, shape, origin, domain):
    """
    Tests behavior of the returned class from get_selective_class. Tests that it:
        - sets the correct number of gridcells to NaN
        - returns a validation subset that is the right shape, and has no nans
        - subsets only the arguments it was told to, and leaves others untouched
            with no nans
    """
    origin_domain_func = MagicMock(return_value=(origin, domain))
    Wrapped = get_selective_class(DummyClass, selective_arg_names, origin_domain_func)
    kwargs = {name: np.zeros(shape) for name in all_arg_names}
    instance = Wrapped()
    instance(**kwargs)
    for name, array in kwargs.items():
        if name in selective_arg_names:
            validated_gridcells = np.product(domain)
            total_gridcells = np.product(array.shape)
            assert np.sum(np.isnan(array)) == total_gridcells - validated_gridcells
            validation_subset = instance.subset_output(name, array)
            assert validation_subset.shape == domain
            assert (
                np.sum(np.isnan(validation_subset)) == 0
            ), "validation slice does not match the values set to nan"
        else:
            assert np.sum(np.isnan(array)) == 0
            assert instance.subset_output(name, array).shape == array.shape
