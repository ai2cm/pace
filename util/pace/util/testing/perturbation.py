from typing import Mapping

import numpy as np


def perturb(input: Mapping[str, np.ndarray]):
    """
    Adds roundoff-level noise to the input array in-place through multiplication.

    Will only make changes to float64 or float32 arrays.
    """
    roundoff = 1e-16
    for data in input.values():
        if isinstance(data, np.ndarray) and data.dtype in (np.float64, np.float32):
            not_fill_value = data < 1e30
            # multiply data by roundoff-level error
            data[not_fill_value] *= 1.0 + np.random.uniform(
                low=-roundoff, high=roundoff, size=data[not_fill_value].shape
            )
