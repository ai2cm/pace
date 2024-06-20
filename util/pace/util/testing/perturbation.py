from typing import Mapping

import numpy as np


def perturb(input: Mapping[str, np.ndarray]):
    """
    Adds roundoff-level noise to the input array in-place through multiplication.

    Will only make changes to float64 or float32 arrays.
    """
    roundoff = 1e-16
    for name, data in input.items():
        if isinstance(data, np.ndarray) and data.dtype in (np.float64, np.float32):
            valid_points = data < 1e30
            if name in [
                "q_con",
                "qcld",
                "qice",
                "qliquid",
                "qrain",
                "qsnow",
                "qgraupel",
            ]:
                valid_points = np.where((data < 1e30) & (data > 0))
            # multiply data by roundoff-level error
            data[valid_points] *= 1.0 + np.random.uniform(
                low=-roundoff, high=roundoff, size=data[valid_points].shape
            )
