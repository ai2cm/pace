import cProfile


class Profiler:
    def __init__(self):
        self._enabled = True
        self.profiler = cProfile.Profile()
        self.profiler.disable()

    def enable(self):
        self.profiler.enable()

    def dump_stats(self, filename: str):
        self.profiler.disable()
        self._enabled = False
        self.profiler.dump_stats(filename)

    @property
    def enabled(self) -> bool:
        """Indicates whether the profiler is currently enabled."""
        return self._enabled


class NullProfiler:
    """A profiler class which does not actually profile anything.

    Meant to be used in place of an optional profiler.
    """

    def __init__(self):
        self.profiler = None
        self._enabled = False

    def enable(self):
        pass

    def dump_stats(self, filename: str):
        pass

    @property
    def enabled(self) -> bool:
        """Indicates whether the profiler is enabled."""
        return False
