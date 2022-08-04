import os.path
from typing import List, Mapping

import numpy as np
import xarray as xr

from .base import Checkpointer
from .thresholds import SavepointName, Threshold, VariableName, cast_to_ndarray


class ValidationCheckpointer(Checkpointer):
    """
    Checkpointer which can be used to validate the output of a test.
    """

    def __init__(
        self, savepoint_data_path: str, thresholds: Mapping[SavepointName, List[Mapping[VariableName, Threshold]]], nhalo: int = 3
    ):
        self._savepoint_data_path = savepoint_data_path
        self._thresholds = thresholds
        self._nhalo = nhalo

    def __call__(self, savepoint_name: str, **kwargs) -> None:
        """Checks that all keyword argument arrays match the data at the savepoint name
        to the thresholds specified.

        Raises:
            AssertionError: If the thresholds on any variable are not met.

        """

        nc_file = os.path.join(self._savepoint_data_path, savepoint_name + ".nc")
        ds = xr.Dataset(nc_file)

        threshold = self.thresholds[savepoint_name]
        for name, array in kwargs.items():
            if name not in ds:
                raise ValueError(f"argument {name} not in netCDF file {nc_file}")

            cast_to_ndarray(array)

            if any(np.abs(output - expected) > threshold[name].relative * np.abs(expected)):
                raise AssertionError("")
            #rtol =
            atol = threshold[name].absolute

            # numpy.testing.assert_allclose uses atol=0 by default, which means it isn't specified.
            if atol is None:
                atol = 0
            np.testing.assert_allclose(array, serialized, rtol=rtol, atol=atol)
