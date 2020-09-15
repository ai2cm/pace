from typing import Iterable, Callable
from ..quantity import Quantity
from .sizer import SubtileGridSizer

try:
    import gt4py
except ImportError:
    gt4py = None


def _wrap_storage_call(function, backend):
    def wrapped(shape, dtype=float, **kwargs):
        return function(backend, [0] * len(shape), shape, dtype, **kwargs)

    wrapped.__name__ = function.__name__
    return wrapped


class StorageNumpy:
    def __init__(self, backend: str):
        """Initialize an object which behaves like the numpy module, but uses
        gt4py storage objects for zeros, ones, and empty.

        Args:
            backend: gt4py backend
        """
        self.empty = _wrap_storage_call(gt4py.storage.empty, backend)
        self.zeros = _wrap_storage_call(gt4py.storage.zeros, backend)
        self.ones = _wrap_storage_call(gt4py.storage.ones, backend)


class QuantityFactory:
    def __init__(self, sizer: SubtileGridSizer, numpy):
        self._sizer = sizer
        self._numpy = numpy

    @classmethod
    def from_backend(cls, sizer: SubtileGridSizer, backend: str):
        """Initialize a QuantityFactory to use a specific gt4py backend.

        Args:
            sizer: object which determines array sizes
            backend: gt4py backend
        """
        numpy = StorageNumpy(backend)
        return cls(sizer, numpy)

    def empty(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.empty, dims, units, dtype)

    def zeros(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.zeros, dims, units, dtype)

    def ones(self, dims: Iterable[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.ones, dims, units, dtype)

    def _allocate(
        self, allocator: Callable, dims: Iterable[str], units: str, dtype: type = float
    ):
        origin = self._sizer.get_origin(dims)
        extent = self._sizer.get_extent(dims)
        shape = self._sizer.get_shape(dims)
        try:
            data = allocator(shape, dtype=dtype, default_origin=origin)
        except TypeError:
            data = allocator(shape, dtype=dtype)
        return Quantity(data, dims=dims, units=units, origin=origin, extent=extent,)
