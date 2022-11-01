import numpy as np

from .._optional_imports import cupy


def to_numpy(array) -> np.ndarray:
    """
    Input array can be a numpy array or a cupy array. Returns numpy array.
    """
    try:
        output = np.asarray(array)
    except ValueError as err:
        if err.args[0] == "object __array__ method not producing an array":
            output = cupy.asnumpy(array)
        else:
            raise err
    except TypeError as err:
        if err.args[0].startswith(
            "Implicit conversion to a NumPy array is not allowed."
        ):
            output = cupy.asnumpy(array)
        else:
            raise err
    return output
