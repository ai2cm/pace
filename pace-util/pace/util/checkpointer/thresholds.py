import collections
import contextlib
import dataclasses
from typing import List, Mapping, Optional, Union

import gt4py.storage
import numpy as np

from ..quantity import Quantity
from .base import Checkpointer


SavepointName = str
VariableName = str


class InsufficientTrialsError(Exception):
    pass


@dataclasses.dataclass
class Threshold:
    relative: float
    absolute: Optional[float] = None


def cast_to_ndarray(
    array: Union[Quantity, gt4py.storage.Storage, np.ndarray]
) -> np.ndarray:
    if isinstance(array, Quantity):
        array = array.data
    if isinstance(array.data, np.ndarray):
        return array.data
    else:
        return array


class ThresholdCalibrationCheckpointer(Checkpointer):
    def __init__(self, factor: float = 1.0):
        """
        Args:
            factor: set thresholds equal to this factor of the maximum error
                seen across trials
        """
        self._minimums: Mapping[
            SavepointName, List[Mapping[VariableName, np.ndarray]]
        ] = collections.defaultdict(list)
        self._maximums: Mapping[
            SavepointName, List[Mapping[VariableName, np.ndarray]]
        ] = collections.defaultdict(list)
        self._factor = factor
        self._sums: Mapping[
            SavepointName, List[Mapping[VariableName, np.ndarray]]
        ] = collections.defaultdict(list)
        self._n_trials = 0
        self._n_calls: Mapping[SavepointName, int] = collections.defaultdict(int)

    def __call__(self, savepoint_name, **kwargs):
        n_calls = self._n_calls[savepoint_name]
        if len(self._minimums[savepoint_name]) < n_calls + 1:
            self._minimums[savepoint_name].append(
                collections.defaultdict(lambda: np.inf)
            )
            self._maximums[savepoint_name].append(
                collections.defaultdict(lambda: -np.inf)
            )
        for varname, array in kwargs.items():
            array: np.ndarray = cast_to_ndarray(array)
            self._minimums[savepoint_name][n_calls][varname] = np.minimum(
                self._minimums[savepoint_name][n_calls][varname], array
            )
            self._maximums[savepoint_name][n_calls][varname] = np.maximum(
                self._maximums[savepoint_name][n_calls][varname], array
            )
            self._sums[savepoint_name][n_calls][varname] += array

        self._n_calls[savepoint_name] += 1

    @contextlib.contextmanager
    def trial(self):
        """
        Context manager for a trial.

        A new context manager should entered each time the code being
        calibrated is called, and exited at the end of code execution.
        If each of these calls is done with slightly perturbed inputs,
        this calibrator will be able to estimate an error tolerance for
        each savepoint call.
        """
        for name in self._n_calls:
            self._n_calls[name] = 0
        yield
        self._n_trials += 1

    @property
    def thresholds(
        self,
    ) -> Mapping[SavepointName, List[Mapping[VariableName, Threshold]]]:
        if self._n_trials < 2:
            raise InsufficientTrialsError(
                "at least 2 trials required to generate thresholds"
            )
        return_value = {}
        for savepoint_name in self._minimums:
            return_value[savepoint_name] = []
            for trial in range(self._n_trials):
                return_value[savepoint_name].append({})
                for varname, minimum in self._minimums[savepoint_name][trial].items():
                    maximum = self._maximums[savepoint_name][trial][varname]
                    mean = self._sums[savepoint_name][trial][varname] / self._n_trials
                    return_value[savepoint_name][trial][varname] = Threshold(
                        relative=self._factor * (maximum - minimum) / mean,
                        absolute=self._factor * (maximum - minimum),
                    )
