from typing import Callable, Optional, Sequence

import numpy as np

from .._optional_imports import gt4py
from ..constants import SPATIAL_DIMS
from ..quantity import Quantity, QuantityHaloSpec
from .sizer import GridSizer


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

    def _backend(self) -> Optional[str]:
        try:
            return self._numpy.backend
        except AttributeError:
            return None

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
        return Quantity(
            data,
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
            gt4py_backend=self._backend(),
        )

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
