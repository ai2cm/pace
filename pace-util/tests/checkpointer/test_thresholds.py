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
    assert checkpointer.thresholds.savepoints == {}


def test_thresholds_one_data_trial():
    checkpointer = ThresholdCalibrationCheckpointer()
    with checkpointer.trial():
        data = np.asarray([0.0, 0.0, 0.0])
        checkpointer("savepoint_name", data=data)
    with pytest.raises(InsufficientTrialsError):
        checkpointer.thresholds


@pytest.mark.parametrize(
    "factor, values, rel_threshold, abs_threshold",
    [
        pytest.param(1.0, [0.0, 0.0], 0.0, 0.0, id="zero_threshold"),
        pytest.param(1.0, [0.0, 1.0], 2.0, 1.0, id="nonzero_threshold"),
        pytest.param(1.5, [0.0, 1.0], 3.0, 1.5, id="non_identity_factor"),
        pytest.param(1.0, [4.0, 6.0], 0.4, 2.0, id="larger_mean"),
        pytest.param(1.0, [-1.0, 1.0], 2.0, 2.0, id="varying_sign"),
        pytest.param(1.0, [-5.0, 5.0, 10.0, 0.0], 3.0, 15.0, id="more_values"),
    ],
)
def test_thresholds_sufficient_trials(factor, values, rel_threshold, abs_threshold):
    checkpointer = ThresholdCalibrationCheckpointer(factor=factor)
    for val in values:
        with checkpointer.trial():
            data = np.asarray([val, 0.0, 0.0])
            checkpointer("savepoint_name", data=data)
    assert checkpointer.thresholds.savepoints == {
        "savepoint_name": [
            {"data": Threshold(relative=rel_threshold, absolute=abs_threshold)}
        ]
    }


def test_thresholds_more_variables():
    checkpointer = ThresholdCalibrationCheckpointer(factor=1.0)
    with checkpointer.trial():
        data1 = np.asarray([0.0, 0.0, 0.0])
        data2 = np.asarray([0.0, 0.0, 0.0])
        checkpointer("savepoint_name", data1=data1, data2=data2)
    with checkpointer.trial():
        data1 = np.asarray([0.0, 0.0, 0.0])
        data2 = np.asarray([1.0, 0.0, 0.0])
        checkpointer("savepoint_name", data1=data1, data2=data2)
    assert checkpointer.thresholds.savepoints == {
        "savepoint_name": [
            {
                "data1": Threshold(relative=0.0, absolute=0.0),
                "data2": Threshold(relative=2.0, absolute=1.0),
            }
        ]
    }


def test_thresholds_two_calls():
    checkpointer = ThresholdCalibrationCheckpointer(factor=1.0)
    with checkpointer.trial():
        data1 = np.asarray([0.0, 0.0, 0.0])
        data2 = np.asarray([0.0, 0.0, 0.0])
        checkpointer("savepoint_name", data=data1)
        checkpointer("savepoint_name", data=data2)
    with checkpointer.trial():
        data1 = np.asarray([0.0, 0.0, 0.0])
        data2 = np.asarray([1.0, 0.0, 0.0])
        checkpointer("savepoint_name", data=data1)
        checkpointer("savepoint_name", data=data2)
    assert checkpointer.thresholds.savepoints == {
        "savepoint_name": [
            {
                "data": Threshold(relative=0.0, absolute=0.0),
            },
            {"data": Threshold(relative=2.0, absolute=1.0)},
        ]
    }
