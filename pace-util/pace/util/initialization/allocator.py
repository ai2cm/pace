from typing import Callable, Iterable, Sequence, Tuple

from ..quantity import Quantity
from .sizer import GridSizer


try:
    import gt4py
except ImportError:
    gt4py = None


def _wrap_storage_call(function, backend):
    def wrapped(shape, dtype=float, **kwargs):
        kwargs["managed_memory"] = True
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
    def __init__(self, sizer: GridSizer, numpy):
        self.sizer: GridSizer = sizer
        self._numpy = numpy

    def set_extra_dim_lengths(self, **kwargs):
        """
        Set the length of extra (non-x/y/z) dimensions.
        """
        self.sizer.extra_dim_lengths.update(kwargs)

    @classmethod
    def from_backend(cls, sizer: GridSizer, backend: str):
        """Initialize a QuantityFactory to use a specific gt4py backend.

        Args:
            sizer: object which determines array sizes
            backend: gt4py backend
        """
        numpy = StorageNumpy(backend)
        return cls(sizer, numpy)

    def empty(self, dims: Sequence[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.empty, dims, units, dtype)

    def zeros(self, dims: Sequence[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.zeros, dims, units, dtype)

    def ones(self, dims: Sequence[str], units: str, dtype: type = float):
        return self._allocate(self._numpy.ones, dims, units, dtype)

    def _allocate(
        self, allocator: Callable, dims: Sequence[str], units: str, dtype: type = float
    ):
        origin = self.get_origin(dims)
        extent = self.get_extent(dims)
        shape = self.get_shape(dims)
        try:
            data = allocator(shape, dtype=dtype, default_origin=origin)
        except TypeError:
            data = allocator(shape, dtype=dtype)
        return Quantity(
            data,
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
        )

    def get_origin(self, dims: Iterable[str]) -> Tuple[int, ...]:
        return self.sizer.get_origin(dims=dims)

    def get_extent(self, dims: Iterable[str]) -> Tuple[int, ...]:
        return self.sizer.get_extent(dims=dims)

    def get_shape(self, dims: Iterable[str]) -> Tuple[int, ...]:
        return self.sizer.get_shape(dims=dims)
