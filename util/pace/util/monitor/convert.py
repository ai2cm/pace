import numpy as np

from .._optional_imports import cupy
from ..quantity import Quantity


def convert_to_numpy(quantity: Quantity) -> np.ndarray:
    try:
        output = np.asarray(quantity.view[:])
    except ValueError as err:
        if err.args[0] == "object __array__ method not producing an array":
            output = cupy.asnumpy(quantity.view[:])
        else:
            raise err
    except TypeError as err:
        if err.args[0].startswith(
            "Implicit conversion to a NumPy array is not allowed."
        ):
            output = cupy.asnumpy(quantity.view[:])
        else:
            raise err
    return output
