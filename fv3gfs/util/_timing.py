import contextlib
from typing import Mapping
from timeit import default_timer as time
import warnings


class Timer:
    """Class to accumulate timings for named operations."""

    def __init__(self):
        self._clock_starts = {}
        self._accumulated_time = {}
        self._enabled = True

    def start(self, name):
        """Start timing a given named operation."""
        if self._enabled:
            if name in self._clock_starts:
                raise ValueError(f"clock already started for '{name}'")
            else:
                self._clock_starts[name] = time()

    def stop(self, name):
        """Stop timing a given named operation, and add the time elapsed to
        accumulated timing.
        """
        if self._enabled:
            if name not in self._accumulated_time:
                self._accumulated_time[name] = time() - self._clock_starts.pop(name)
            else:
                self._accumulated_time[name] += time() - self._clock_starts.pop(name)

    @contextlib.contextmanager
    def clock(self, name):
        """Context manager to produce timings of operations.

        Args:
            name: the name of the operation being timed

        Example:
            The context manager times operations that happen within its context. The
            following would time a time.sleep operation::

                >>> import time
                >>> from fv3gfs.util import Timer
                >>> timer = Timer()
                >>> with timer.clock("sleep"):
                ...     time.sleep(1)
                ...
                >>> timer.times
                {'sleep': 1.0032463260000029}
        """
        self.start(name)
        yield
        self.stop(name)

    @property
    def times(self) -> Mapping[str, float]:
        """accumulated timings for each operation name"""
        if len(self._clock_starts) > 0:
            warnings.warn(
                "Retrieved times while clocks are still going, "
                "incomplete times are not included: "
                f"{list(self._clock_starts.keys())}",
                RuntimeWarning,
            )
        return self._accumulated_time.copy()

    def reset(self):
        """Remove all accumulated timings."""
        self._accumulated_time.clear()

    def enable(self):
        """Enable the Timer."""
        self._enabled = True

    def disable(self):
        """Disable the Timer."""
        if len(self._clock_starts) > 0:
            raise RuntimeError(
                "Cannot disable timer while clocks are still going: "
                f"{list(self._clock_starts.keys())}"
            )
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Indicates whether the timer is currently enabled."""
        return self._enabled


class NullTimer(Timer):
    """A Timer class which does not actually accumulate timings.

    Meant to be used in place of an optional timer.
    """

    def start(self, name):
        pass

    def stop(self, name):
        pass
