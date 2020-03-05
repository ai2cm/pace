from typing import Tuple, Dict, Iterable
import functools
import collections.abc
import dataclasses
import numpy as np
from . import constants
import xarray as xr
try:
    import cupy
except ImportError:
    cupy = np  # avoids attribute errors while also disabling cupy support
try:
    import gt4py
except ImportError:
    gt4py = None


class FrozenDict(collections.abc.Mapping):

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        return hash(tuple(sorted(self._d.iteritems())))


@dataclasses.dataclass
class QuantityMetadata:
    origin: Tuple[int, ...]
    extent: Tuple[int, ...]
    dims: Tuple[str, ...]
    units: str
    data_type: type
    dtype: type

    @property
    def dim_lengths(self):
        return dict(zip(self.dims, self.extent))

    @property
    def np(self):
        if issubclass(self.data_type, np.ndarray):
            return np
        elif issubclass(self.data_type, cupy.ndarray):
            return cupy
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {self.data_type}"
            )


class Quantity:
    """
    Data container for physical quantities.
    """

    def __init__(
            self,
            data,
            dims: Iterable[str, ...],
            units: str,
            origin: Iterable[int, ...] = None,
            extent: Iterable[int, ...] = None):
        """
        Initialize a Quantity.

        Args:
            data: ndarray-like object containing the underlying data
            dims: dimension names for each axis
            units: units of the quantity
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
        """
        if origin is None:
            origin = (0,) * len(dims)  # default origin at origin of array
        else:
            origin = tuple(origin)
        if extent is None:
            extent = tuple(length - start for length, start in zip(data.shape, origin))
        else:
            extent = tuple(extent)
        self._metadata = QuantityMetadata(
            origin=origin,
            extent=extent,
            dims=tuple(dims),
            units=units,
            data_type=type(data),
            dtype=data.dtype,
        )
        self._attrs = {}
        self._data = data
        self._compute_domain_view = BoundedArrayView(self.data, self.origin, self.extent)

    @classmethod
    def from_data_array(cls, data_array, origin=None, extent=None):
        """
        Initialize a Quantity from an xarray.DataArray.

        Args:
        
        """
        if 'units' not in data_array.attrs:
            raise ValueError('need units attribute to create Quantity from DataArray')
        return cls(
            data_array.values,
            data_array.dims,
            data_array.attrs['units'],
            origin=origin,
            extent=extent
        )

    def __repr__(self):
        return (
            f"Quantity(\n    data=\n{self.data},\n    dims={self.dims},\n"
            f"    units={self.units},\n    origin={self.origin},\n"
            f"    extent={self.extent}\n)"
        )

    def sel(self, **kwargs):
        return self.view[tuple(kwargs.get(dim, slice(None, None)) for dim in self.dims)]

    @classmethod
    def from_storage(cls, gt4py_storage, dims, units):
        raise NotImplementedError()

    @property
    def metadata(self):
        return self._metadata

    @property
    def units(self):
        return self.metadata.units

    @property
    def attrs(self):
        return dict(**self._attrs, units=self._metadata.units)

    @property
    def dims(self):
        return self.metadata.dims
    
    @property
    def values(self):
        return_array = np.asarray(self._data)
        return_array.flags.writeable = False
        return return_array
    
    @property
    def view(self):
        return self._compute_domain_view

    @property
    def data(self):
        return self._data

    @property
    def origin(self):
        return self.metadata.origin
    
    @property
    def extent(self):
        return self.metadata.extent
    
    @property
    def storage(self):
        raise NotImplementedError()

    @property
    def data_array(self):
        return xr.DataArray(
            self.view[:],
            dims=self.dims,
            attrs=self.attrs
        )

    @property
    def np(self):
        return self.metadata.np


class BoundedArrayView:

    def __init__(self, array, origin, extent):
        self._data = array
        self._origin = origin
        self._extent = extent

    @property
    def origin(self):
        return self._origin
    
    @property
    def extent(self):
        return self._extent

    def __getitem__(self, index):
        return self._data[self._get_compute_index(index)]

    def __setitem__(self, index, value):
        self._data[self._get_compute_index(index)] = value

    def _get_compute_index(self, index):
        if not isinstance(index, (tuple, list)):
            index = (index,)
        index = fill_index(index, len(self._data.shape))
        shifted_index = []
        for entry, origin, extent in zip(index, self.origin, self.extent):
            if isinstance(entry, slice):
                shifted_slice = shift_slice(entry, origin)
                shifted_index.append(bound_default_slice(shifted_slice, origin, origin + extent))
            elif entry is None:
                shifted_index.append(entry)
            else:
                shifted_index.append(entry + origin)
        return tuple(shifted_index)


def fill_index(index, length):
    return tuple(index) + (slice(None, None, None),) * (length - len(index))


def shift_slice(slice_in, shift):
    if slice_in.start is None:
        start = None
    else:
        start = slice_in.start + shift
    if slice_in.stop is None:
        stop = None
    else:
        stop = slice_in.stop + shift
    return slice(start, stop, slice_in.step)


def bound_default_slice(slice_in, start=None, stop=None):
    if slice_in.start is not None:
        start = slice_in.start
    if slice_in.stop is not None:
        stop = slice_in.stop
    return slice(start, stop, slice_in.step)
