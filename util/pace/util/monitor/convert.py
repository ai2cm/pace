import numpy as np

from .._optional_imports import cupy


def to_numpy(array, dtype=None) -> np.ndarray:
    """
    Input array can be a numpy array or a cupy array. Returns numpy array.
    """
    try:
        output = np.asarray(array, dtype=dtype)
    except ValueError as err:
        if err.args[0] == "object __array__ method not producing an array":
            output = cupy.asnumpy(array, dtype=dtype)
        else:
            raise err
    except TypeError as err:
        if err.args[0].startswith(
            "Implicit conversion to a NumPy array is not allowed."
        ):
            output = cupy.asnumpy(array, dtype=dtype)
        else:
            raise err
    return output
