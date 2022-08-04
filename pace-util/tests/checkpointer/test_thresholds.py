import numpy as np
import pytest

from pace.util.checkpointer import (
    InsufficientTrialsError,
    Threshold,
    ThresholdCalibrationCheckpointer,
)


def test_thresholds_no_trials():
    checkpointer = ThresholdCalibrationCheckpointer()
    with pytest.raises(InsufficientTrialsError):
        checkpointer.thresholds


def test_thresholds_one_empty_trial():
    checkpointer = ThresholdCalibrationCheckpointer()
    with checkpointer.trial():
        pass
    with pytest.raises(InsufficientTrialsError):
        checkpointer.thresholds


def test_thresholds_two_empty_trials():
    checkpointer = ThresholdCalibrationCheckpointer()
    for _ in range(2):
        with checkpointer.trial():
            pass
    assert checkpointer.thresholds == {}


def test_thresholds_one_data_trial():
    checkpointer = ThresholdCalibrationCheckpointer()
    with checkpointer.trial():
        data = np.asarray([0.0, 0.0, 0.0])
        checkpointer("savepoint_name", data=data)
    with pytest.raises(InsufficientTrialsError):
        checkpointer.thresholds


def test_thresholds_two_data_trials_zero_threshold():
    checkpointer = ThresholdCalibrationCheckpointer()
    with checkpointer.trial():
        data = np.asarray([0.0, 0.0, 0.0])
        checkpointer("savepoint_name", data=data)
    with checkpointer.trial():
        data = np.asarray([0.0, 0.0, 0.0])
        checkpointer("savepoint_name", data=data)
    checkpointer.thresholds == {"data": Threshold(relative=0.0)}


@pytest.mark.parametrize(
    "factor, val1, val2, rel_threshold",
    [
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 2.0),
        (1.5, 0.0, 1.0, 3.0),
        (1.0, 4.0, 5.0, 1.0 / 4.5),
        (1.0, -1.0, 1.0, 2.0),
    ],
)
def test_thresholds_two_data_trials_nonzero_threshold(
    factor, val1, val2, rel_threshold
):
    checkpointer = ThresholdCalibrationCheckpointer(factor=factor)
    with checkpointer.trial():
        data = np.asarray([val1, 0.0, 0.0])
        checkpointer("savepoint_name", data=data)
    with checkpointer.trial():
        data = np.asarray([val2, 0.0, 0.0])
        checkpointer("savepoint_name", data=data)
    checkpointer.thresholds == {"data": Threshold(relative=rel_threshold)}
