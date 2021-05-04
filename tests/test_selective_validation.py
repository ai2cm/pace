import numpy as np
import pytest

from fv3core.utils.validation import SelectiveValidation


@pytest.mark.parametrize(
    "shape, origin, domain",
    [
        pytest.param((3, 3, 3), (0, 0, 0), (3, 3, 3), id="whole_array_valid"),
        pytest.param((3, 3, 3), (0, 0, 0), (0, 0, 0), id="whole_array_invalid"),
        pytest.param((4, 4, 4), (1, 2, 0), (2, 1, 3), id="some_valid"),
    ],
)
def test_setting_nans(shape, origin, domain):
    original_mode = SelectiveValidation.TEST_MODE
    try:
        array = np.zeros(shape)
        selector = SelectiveValidation(origin, domain)
        SelectiveValidation.TEST_MODE = True
        selector.set_nans_if_test_mode(array)
        assert np.sum(np.isnan(array)) == (np.product(array.shape) - np.product(domain))
        assert (
            np.sum(np.isnan(array[selector.validation_slice])) == 0
        ), "validation_slice does not match the values set to nan"
    finally:
        SelectiveValidation.TEST_MODE = original_mode
