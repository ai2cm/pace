import collections
import contextlib
import dataclasses
from typing import Dict, List, Mapping, Union

import gt4py.storage
import numpy as np

from ..quantity import Quantity
from .base import Checkpointer


SavepointName = str
VariableName = str
ArrayLike = Union[Quantity, gt4py.storage.Storage, np.ndarray]


class InsufficientTrialsError(Exception):
    pass


@dataclasses.dataclass
class Threshold:
    relative: float
    absolute: float

    def merge(self, other: "Threshold") -> "Threshold":
        """
        Provide a threshold which is always satisfied
        if both input thresholds are satisfied.

        This is generally a less strict threshold than either input.
        """
        return Threshold(
            relative=max(self.relative, other.relative),
            absolute=max(self.absolute, other.absolute),
        )


@dataclasses.dataclass
class SavepointThresholds:
    savepoints: Dict[SavepointName, List[Dict[VariableName, Threshold]]]


def cast_to_ndarray(array: ArrayLike) -> np.ndarray:
    if isinstance(array, Quantity):
        array = array.data
    if isinstance(array.data, np.ndarray):
        return array.data
    else:
        return array


class ThresholdCalibrationCheckpointer(Checkpointer):
    """
    Calibrates thresholds to be used by a ValidationCheckpointer.

    Does this by recording the minimum and maximum values seen across trials,
    and using them to derive the maximum relative and absolute error one could
    have across any pair of trials, then multiplying this by a user-provided factor.
    """

    def __init__(self, factor: float = 1.0):
        """
        Args:
            factor: set thresholds equal to this factor of the maximum error
                seen across trials
        """
        # we keep dictionaries (over savepoint name) of lists (over call count)
        # of dictionaries (over variable name) of numpy arrays
        self._minimums: Mapping[
            SavepointName, List[Mapping[VariableName, np.ndarray]]
        ] = collections.defaultdict(list)
        self._maximums: Mapping[
            SavepointName, List[Mapping[VariableName, np.ndarray]]
        ] = collections.defaultdict(list)
        self._factor = factor
        self._abs_sums: Mapping[
            SavepointName, List[Mapping[VariableName, np.ndarray]]
        ] = collections.defaultdict(list)
        self._n_trials = 0
        self._n_calls: Mapping[SavepointName, int] = collections.defaultdict(int)

    def __call__(self, savepoint_name, **kwargs):
        """
        Record values for a savepoint.

        Args:
            savepoint_name: name of the savepoint
            **kwargs: data for the savepoint
        """
        i_call = self._n_calls[savepoint_name]
        if len(self._minimums[savepoint_name]) < i_call + 1:
            self._minimums[savepoint_name].append(
                collections.defaultdict(lambda: np.inf)
            )
            self._maximums[savepoint_name].append(
                collections.defaultdict(lambda: -np.inf)
            )
            self._abs_sums[savepoint_name].append(collections.defaultdict(lambda: 0.0))
        for varname, array in kwargs.items():
            array: np.ndarray = cast_to_ndarray(array)
            self._minimums[savepoint_name][i_call][varname] = np.minimum(
                self._minimums[savepoint_name][i_call][varname], array
            )
            self._maximums[savepoint_name][i_call][varname] = np.maximum(
                self._maximums[savepoint_name][i_call][varname], array
            )
            self._abs_sums[savepoint_name][i_call][varname] += np.abs(array)

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
    ) -> SavepointThresholds:
        if self._n_trials < 2:
            raise InsufficientTrialsError(
                "at least 2 trials required to generate thresholds"
            )
        savepoints: Dict[SavepointName, List[Dict[VariableName, Threshold]]] = {}
        for savepoint_name in self._minimums:
            savepoints[savepoint_name] = []
            for i_call in range(self._n_calls[savepoint_name]):
                savepoints[savepoint_name].append({})
                for varname, minimum in self._minimums[savepoint_name][i_call].items():
                    maximum = self._maximums[savepoint_name][i_call][varname]
                    mean_abs = (
                        self._abs_sums[savepoint_name][i_call][varname] / self._n_trials
                    )
                    if np.all(mean_abs == 0.0):
                        relative = 0.0
                    else:
                        relative = self._factor * np.nanmax(
                            (maximum - minimum) / mean_abs
                        )
                    savepoints[savepoint_name][i_call][varname] = Threshold(
                        relative=float(relative),
                        absolute=float(self._factor * np.max(maximum - minimum)),
                    )
        return SavepointThresholds(savepoints=savepoints)
