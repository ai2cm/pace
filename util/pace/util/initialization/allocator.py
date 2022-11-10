from typing import Callable, Sequence

import numpy as np

from .._optional_imports import gt4py
from ..constants import SPATIAL_DIMS
from ..quantity import Quantity
from .sizer import GridSizer


def _wrap_storage_call(function, backend):
    def wrapped(shape, dtype=float, **kwargs):
        kwargs["managed_memory"] = True
        kwargs.setdefault("default_origin", [0] * len(shape))
        return function(backend, shape=shape, dtype=dtype, **kwargs)

    wrapped.__name__ = function.__name__
    return wrapped


class StorageNumpy:
    def __init__(self, backend: str):
        """Initialize an object which behaves like the numpy module, but uses
        gt4py storage objects for zeros, ones, and empty.

        Args:
            backend: gt4py backend
        """
        self.backend = backend

    def empty(self, *args, **kwargs) -> np.ndarray:
        return gt4py.storage.empty(*args, backend=self.backend, **kwargs)

    def ones(self, *args, **kwargs) -> np.ndarray:
        return gt4py.storage.ones(*args, backend=self.backend, **kwargs)

    def zeros(self, *args, **kwargs) -> np.ndarray:
        return gt4py.storage.zeros(*args, backend=self.backend, **kwargs)

    def from_array(self, *args, **kwargs) -> np.ndarray:
        return gt4py.storage.from_array(*args, backend=self.backend, **kwargs)


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

    def empty(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = float,
    ):
        return self._allocate(self._numpy.empty, dims, units, dtype)

    def zeros(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = float,
    ):
        return self._allocate(self._numpy.zeros, dims, units, dtype)

    def ones(
        self,
        dims: Sequence[str],
        units: str,
        dtype: type = float,
    ):
        return self._allocate(self._numpy.ones, dims, units, dtype)

    def from_array(
        self,
        data: np.ndarray,
        dims: Sequence[str],
        units: str,
    ):
        """
        Create a Quantity from a numpy array.

        That numpy array must correspond to the correct shape and extent
        for the given dims.
        """
        # TODO: Replace this once aligned_index fix is included.
        quantity_data = self._numpy.from_array(
            data, data.dtype, aligned_index=[0] * len(data.shape)
        )
        return Quantity(data=quantity_data, dims=dims, units=units)

    def _allocate(
        self,
        allocator: Callable,
        dims: Sequence[str],
        units: str,
        dtype: type = float,
    ):
        origin = self.sizer.get_origin(dims)
        extent = self.sizer.get_extent(dims)
        shape = self.sizer.get_shape(dims)
        dimensions = [
            axis
            if any(dim in axis_dims for axis_dims in SPATIAL_DIMS)
            else str(shape[index])
            for index, (dim, axis) in enumerate(
                zip(dims, ("I", "J", "K", *([None] * (len(dims) - 3))))
            )
        ]
        try:
            data = allocator(
                shape, dtype=dtype, aligned_index=origin, dimensions=dimensions
            )
        except TypeError:
            data = allocator(shape, dtype=dtype)
        return Quantity(data, dims=dims, units=units, origin=origin, extent=extent)
