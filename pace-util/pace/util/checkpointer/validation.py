import os.path
from typing import List, Mapping

import numpy as np
import xarray as xr

from .base import Checkpointer
from .thresholds import SavepointName, Threshold, VariableName, cast_to_ndarray


def _clip_fortran_array(array: xr.DataArray) -> np.ndarray:
    """Convert the Fortran array size to that in Pace."""
    # Still unclear how this works.
    return array


class ValidationCheckpointer(Checkpointer):
    """
    Checkpointer which can be used to validate the output of a test.
    """

    def __init__(
        self,
        savepoint_data_path: str,
        thresholds: Mapping[SavepointName, List[Mapping[VariableName, Threshold]]],
        nhalo: int = 3,
    ):
        self._savepoint_data_path = savepoint_data_path
        self._thresholds = thresholds
        self._nhalo = nhalo

    def __call__(self, savepoint_name: str, **kwargs) -> None:
        """Checks the arrays passed as keyword arguments against thresholds specified.

        Raises:
            AssertionError: If the thresholds on any variable are not met.

        """

        nc_file = os.path.join(self._savepoint_data_path, savepoint_name + ".nc")
        ds = xr.open_dataset(nc_file)

        threshold = self.thresholds[savepoint_name]
        for name, array in kwargs.items():
            if name not in ds:
                raise ValueError(f"argument {name} not in netCDF file {nc_file}")

            output = cast_to_ndarray(array)
            expected = _clip_fortran_array(ds[name])

            rel_values = threshold[name].relative * np.abs(expected)
            rtol_failures = np.abs(output - expected) > rel_values
            if any(rtol_failures):
                raise AssertionError(
                    f"{name} is not within relative tolerance {threshold[name].relative} at {sum(rtol_failures)} indices"
                )

            atol = threshold[name].absolute
            if atol:
                atol_failures = np.abs(output - expected) > atol
                if any(atol_failures):
                    raise AssertionError(
                        f"{name} is not within absolute tolerance {atol} at {sum(atol_failures)} indices"
                    )
