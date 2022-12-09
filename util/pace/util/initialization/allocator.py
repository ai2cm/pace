from typing import Callable, Optional, Sequence

import numpy as np

from .._optional_imports import gt4py
from ..constants import SPATIAL_DIMS, X_DIMS, Y_DIMS, Z_DIMS
from ..quantity import Quantity, QuantityHaloSpec
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
        base = self.empty(dims=dims, units=units, dtype=data.dtype)
        base.data[:] = base.np.asarray(data)
        return base

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
        mask = tuple(
            [
                any(dim in coord_dims for dim in dims)
                for coord_dims in [X_DIMS, Y_DIMS, Z_DIMS]
            ]
        )
        extra_dims = [i for i in dims if i not in SPATIAL_DIMS]
        if len(extra_dims) > 0 or not dims:
            mask = None
        try:
            data = allocator(shape, dtype=dtype, default_origin=origin, mask=mask)
        except TypeError:
            data = allocator(shape, dtype=dtype)
        return Quantity(data, dims=dims, units=units, origin=origin, extent=extent)

    def get_quantity_halo_spec(
        self, dims: Sequence[str], n_halo: Optional[int] = None, dtype: type = float
    ) -> QuantityHaloSpec:
        """Build memory specifications for the halo update.

        Args:
            dims: dimensionality of the data
            n_halo: number of halo points to update, defaults to self.n_halo
            dtype: data type of the data
            backend: gt4py backend to use
        """

        # TEMPORARY: we do a nasty temporary allocation here to read in the hardware
        # memory layout. Further work in GT4PY will allow for deferred allocation
        # which will give access to those information while making sure
        # we don't allocate
        # Refactor is filed in ticket DSL-820

        temp_quantity = self.zeros(dims=dims, units="", dtype=dtype)

        if n_halo is None:
            n_halo = self.sizer.n_halo

        return temp_quantity.halo_spec(n_halo)
