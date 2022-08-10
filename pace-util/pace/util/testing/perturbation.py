import numpy as np


def perturb(input):
    roundoff = 1e-16
    for data in input.values():
        if isinstance(data, np.ndarray) and data.dtype in (np.float64, np.float32):
            not_fill_value = data < 1e30
            # multiply data by roundoff-level error
            data[not_fill_value] *= 1.0 + np.random.uniform(
                low=-roundoff, high=roundoff, size=data[not_fill_value].shape
            )
