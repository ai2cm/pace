from typing import Union

import numpy as np


def compare_arr(computed_data, ref_data):
    """
    Smooth error near zero values.
    Inputs are arrays.
    """
    if ref_data.dtype in (np.float64, np.int64, np.float32, np.int32):
        denom = np.abs(ref_data) + np.abs(computed_data)
        compare = 2.0 * np.abs(computed_data - ref_data) / denom
        compare[denom == 0] = 0.0
        return compare
    elif ref_data.dtype in (np.bool,):
        return np.logical_xor(computed_data, ref_data)
    else:
        raise TypeError(f"recieved data with unexpected dtype {ref_data.dtype}")


def compare_scalar(computed_data: np.float64, ref_data: np.float64) -> np.float64:
    """Smooth error near zero values. Scalar versions."""
    err_as_array = compare_arr(np.atleast_1d(computed_data), np.atleast_1d(ref_data))
    return err_as_array[0]


def success_array(
    computed_data: np.ndarray,
    ref_data: np.ndarray,
    eps: float,
    ignore_near_zero_errors: Union[dict, bool],
    near_zero: float,
):
    success = np.logical_or(
        np.logical_and(np.isnan(computed_data), np.isnan(ref_data)),
        compare_arr(computed_data, ref_data) < eps,
    )
    if isinstance(ignore_near_zero_errors, dict):
        if ignore_near_zero_errors.keys():
            near_zero = ignore_near_zero_errors["near_zero"]
            success = np.logical_or(
                success,
                np.logical_and(
                    np.abs(computed_data) < near_zero,
                    np.abs(ref_data) < near_zero,
                ),
            )
    elif ignore_near_zero_errors:
        success = np.logical_or(
            success,
            np.logical_and(
                np.abs(computed_data) < near_zero, np.abs(ref_data) < near_zero
            ),
        )
    return success


def success(computed_data, ref_data, eps, ignore_near_zero_errors, near_zero=0.0):
    return np.all(
        success_array(computed_data, ref_data, eps, ignore_near_zero_errors, near_zero)
    )
