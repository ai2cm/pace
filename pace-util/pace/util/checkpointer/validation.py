import collections
import contextlib
import os.path
from typing import List, Mapping, MutableMapping, Tuple

import numpy as np
import xarray as xr

from .base import Checkpointer
from .thresholds import (
    ArrayLike,
    SavepointName,
    Threshold,
    VariableName,
    cast_to_ndarray,
)


def _clip_pace_array_to_target(
    array: np.ndarray, target_shape: Tuple[int, ...]
) -> np.ndarray:
    """
    Clip an array from pace to align it to a target shape from target serialized data.

    Assumes the target shape has the same number of halo points
    at the start and end of each axis.

    Assumes the input array has the same number of halo points at the start and end of
    each axis, but with an additional buffer point at the end of each axis if the
    data is defined on cell centers.

    Args:
        array: array to clip
        target_shape: shape of target array
    """
    array = _remove_buffer_if_needed(array, target_shape)
    return _remove_symmetric_halos(array, target_shape)


def _remove_buffer_if_needed(array: np.ndarray, target_shape: Tuple[int, ...]):
    selection = []
    for array_len, target_len in zip(array.shape, target_shape):
        if (array_len - target_len) % 2 == 1:
            # clip the buffer point
            selection.append(slice(0, -1))
        else:
            selection.append(slice(None, None))
    return array[tuple(selection)]


def _remove_symmetric_halos(array: np.ndarray, target_shape: Tuple[int, ...]):
    selection = []
    for array_len, target_len in zip(array.shape, target_shape):
        n_halo_clip = (array_len - target_len) // 2
        if n_halo_clip == 0:
            selection.append(slice(None, None))
        else:
            selection.append(slice(n_halo_clip, -n_halo_clip))
    return array[tuple(selection)]


class ValidationCheckpointer(Checkpointer):
    """
    Checkpointer which can be used to validate the output of a test.
    """

    def __init__(
        self,
        savepoint_data_path: str,
        thresholds: Mapping[SavepointName, List[Mapping[VariableName, Threshold]]],
        rank: int,
    ):
        self._savepoint_data_path = savepoint_data_path
        self._thresholds = thresholds
        self._rank = rank
        self._n_calls: MutableMapping[SavepointName, int] = collections.defaultdict(int)

    @contextlib.contextmanager
    def trial(self):
        """
        Context manager for a trial.

        When entered, resets reference data comparison back to the start of the data.

        A new context manager should entered before the code being tested is called,
        and exited at the end of code execution.
        """
        self._n_calls = collections.defaultdict(int)
        yield

    def __call__(self, savepoint_name: str, **kwargs: ArrayLike) -> None:
        """
        Checks the arrays passed as keyword arguments against thresholds specified.

        Args:
            savepoint_name: name of the savepoint
            **kwargs: array data for variables in that savepoint

        Raises:
            AssertionError: if the thresholds on any variable are not met
        """

        nc_file = os.path.join(self._savepoint_data_path, savepoint_name + ".nc")
        ds = xr.open_dataset(nc_file)

        n_calls = self._n_calls[savepoint_name]
        var_thresholds = self._thresholds[savepoint_name][n_calls]
        for varname, array in kwargs.items():
            if varname not in ds:
                raise ValueError(f"argument {varname} not in netCDF file {nc_file}")

            expected = ds[varname][n_calls, self._rank]
            output = _clip_pace_array_to_target(cast_to_ndarray(array), expected.shape)

            np.testing.assert_allclose(
                output, expected, rtol=var_thresholds[varname].relative
            )
            np.testing.assert_allclose(
                output, expected, atol=var_thresholds[varname].absolute
            )
        self._n_calls[savepoint_name] += 1
