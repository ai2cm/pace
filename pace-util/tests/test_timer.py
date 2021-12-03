import time

import pytest

from pace.util import NullTimer, Timer


@pytest.fixture
def timer():
    return Timer()


@pytest.fixture
def null_timer():
    return NullTimer()


def test_start_stop(timer):
    timer.start("label")
    timer.stop("label")
    times = timer.times
    assert "label" in times
    assert len(times) == 1
    assert timer.hits["label"] == 1
    assert len(timer.hits) == 1


def test_null_timer_cannot_be_enabled(null_timer):
    with pytest.raises(NotImplementedError):
        null_timer.enable()


def test_null_timer_is_disabled(null_timer):
    assert not null_timer.enabled


def test_clock(timer):
    with timer.clock("label"):
        # small arbitrary computation task to time
        time.sleep(0.1)
    times = timer.times
    assert "label" in times
    assert len(times) == 1
    assert abs(times["label"] - 0.1) < 1e-2
    assert timer.hits["label"] == 1
    assert len(timer.hits) == 1


def test_start_twice(timer):
    """cannot call start twice consecutively with no stop"""
    timer.start("label")
    with pytest.raises(ValueError) as err:
        timer.start("label")
    assert "clock already started for 'label'" in str(err.value)


def test_clock_in_clock(timer):
    """should not be able to create a given clock inside itself"""
    with timer.clock("label"):
        with pytest.raises(ValueError) as err:
            with timer.clock("label"):
                pass
    assert "clock already started for 'label'" in str(err.value)


def test_consecutive_start_stops(timer):
    """total time increases with consecutive clock blocks"""
    timer.start("label")
    time.sleep(0.01)
    timer.stop("label")
    previous_time = timer.times["label"]
    for i in range(5):
        timer.start("label")
        time.sleep(0.01)
        timer.stop("label")
        assert timer.times["label"] >= previous_time + 0.01
        previous_time = timer.times["label"]
    assert timer.hits["label"] == 6


def test_consecutive_clocks(timer):
    """total time increases with consecutive clock blocks"""
    with timer.clock("label"):
        time.sleep(0.01)
    previous_time = timer.times["label"]
    for i in range(5):
        with timer.clock("label"):
            time.sleep(0.01)
        assert timer.times["label"] >= previous_time + 0.01
        previous_time = timer.times["label"]
    assert timer.hits["label"] == 6


@pytest.mark.parametrize(
    "ops, result",
    [
        ([], True),
        (["enable"], True),
        (["disable"], False),
        (["disable", "enable"], True),
        (["disable", "disable"], False),
    ],
)
def test_enable_disable(timer, ops, result):
    for op in ops:
        getattr(timer, op)()
    assert timer.enabled == result


def test_disabled_timer_does_not_add_key(timer):
    timer.disable()
    with timer.clock("label1"):
        time.sleep(0.01)
    assert len(timer.times) == 0
    with timer.clock("label2"):
        time.sleep(0.01)
    assert len(timer.times) == 0
    assert len(timer.hits) == 0


def test_disabled_timer_does_not_add_time(timer):
    with timer.clock("label"):
        time.sleep(0.01)
    initial_time = timer.times["label"]
    timer.disable()
    with timer.clock("label"):
        time.sleep(0.01)
    assert timer.times["label"] == initial_time
    assert timer.hits["label"] == 1


@pytest.fixture(params=["clean", "one_label", "two_labels"])
def used_timer(request, timer):
    if request.param == "clean":
        return timer
    elif request.param == "one_label":
        with timer.clock("label1"):
            time.sleep(0.01)
        return timer
    elif request.param == "two_labels":
        with timer.clock("label1"):
            time.sleep(0.01)
        with timer.clock("label2"):
            time.sleep(0.01)
        return timer


def test_timer_reset(used_timer):
    used_timer.reset()
    assert len(used_timer.times) == 0
    assert len(used_timer.hits) == 0
