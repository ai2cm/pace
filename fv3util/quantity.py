from typing import Tuple, Iterable, Dict
from types import ModuleType
import dataclasses
import numpy as np
import xarray as xr
try:
    import cupy
except ImportError:
    cupy = np  # avoids attribute errors while also disabling cupy support
try:
    import gt4py
except ImportError:
    gt4py = None

__all__ = ['Quantity', 'QuantityMetadata']


@dataclasses.dataclass
class QuantityMetadata:
    origin: Tuple[int, ...]
    "the start of the computational domain"
    extent: Tuple[int, ...]
    "the shape of the computational domain"
    dims: Tuple[str, ...]
    "names of each dimension"
    units: str
    "units of the quantity"
    data_type: type
    "ndarray-like type used to store the data"
    dtype: type
    "dtype of the data in the ndarray-like object"

    @property
    def dim_lengths(self) -> Dict[str, int]:
        """mapping of dimension names to their lengths"""
        return dict(zip(self.dims, self.extent))

    @property
    def np(self) -> ModuleType:
        """numpy-like module used to interact with the data"""
        if issubclass(self.data_type, np.ndarray):
            return np
        elif issubclass(self.data_type, cupy.ndarray):
            return cupy
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {self.data_type}"
            )


class BoundedArrayView:

    def __init__(self, array, origin, extent):
        self._data = array
        self._origin = origin
        self._extent = extent

    @property
    def origin(self):
        """the start of the computational domain"""
        return self._origin
    
    @property
    def extent(self):
        """the shape of the computational domain"""
        return self._extent

    def __getitem__(self, index):
        if len(self.origin) == 0:
            if isinstance(index, tuple) and len(index) > 0:
                raise IndexError('more than one index given for a zero-dimension array')
            elif isinstance(index, slice) and index != slice(None, None, None):
                raise IndexError('cannot slice a zero-dimension array')
            else:
                return self._data  # array[()] does not return an ndarray
        else:
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
                shifted_slice = shift_slice(entry, origin, extent)
                shifted_index.append(bound_default_slice(shifted_slice, origin, origin + extent))
            elif entry is None:
                shifted_index.append(entry)
            else:
                shifted_index.append(entry + origin)
        return tuple(shifted_index)


class Quantity:
    """
    Data container for physical quantities.
    """

    def __init__(
            self,
            data,
            dims: Iterable[str],
            units: str,
            origin: Iterable[int] = None,
            extent: Iterable[int] = None):
        """
        Initialize a Quantity.

        Args:
            data: ndarray-like object containing the underlying data
            dims: dimension names for each axis
            units: units of the quantity
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
        """
        if isinstance(data, (int, float, list)):
            data = np.asarray(data)
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
    def from_data_array(
            cls,
            data_array: xr.DataArray,
            origin: Iterable[int] = None,
            extent: Iterable[int] = None):
        """
        Initialize a Quantity from an xarray.DataArray.

        Args:
            data_array
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
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

    def sel(self, **kwargs: (slice, int)) -> np.ndarray:
        """Convenience method to perform indexing on `view` using dimension names
        without knowing dimension order.
        
        Args:
            **kwargs: slice/index to retrieve for a given dimension name

        Returns:
            view_selection: an ndarray-like selection of the given indices
                on `self.view`
        """
        return self.view[tuple(kwargs.get(dim, slice(None, None)) for dim in self.dims)]

    @property
    def metadata(self) -> QuantityMetadata:
        return self._metadata

    @property
    def units(self) -> str:
        """units of the quantity"""
        return self.metadata.units

    @property
    def attrs(self) -> dict:
        return dict(**self._attrs, units=self._metadata.units)

    @property
    def dims(self) -> Tuple[str, ...]:
        """names of each dimension"""
        return self.metadata.dims
    
    @property
    def values(self) -> np.ndarray:
        return_array = np.asarray(self._data)
        return_array.flags.writeable = False
        return return_array
    
    @property
    def view(self) -> BoundedArrayView:
        """a view into the computational domain of the underlying data"""
        return self._compute_domain_view

    @property
    def data(self) -> np.ndarray:
        """the underlying array of data"""
        return self._data

    @property
    def origin(self) -> Tuple[int, ...]:
        """the start of the computational domain"""
        return self.metadata.origin
    
    @property
    def extent(self) -> Tuple[int, ...]:
        """the shape of the computational domain"""
        return self.metadata.extent

    @property
    def data_array(self) -> xr.DataArray:
        return xr.DataArray(
            self.view[:],
            dims=self.dims,
            attrs=self.attrs
        )

    @property
    def np(self) -> ModuleType:
        return self.metadata.np


def fill_index(index, length):
    return tuple(index) + (slice(None, None, None),) * (length - len(index))


def shift_slice(slice_in, shift, extent):
    start = shift_index(slice_in.start, shift, extent)
    stop = shift_index(slice_in.stop, shift, extent)
    return slice(start, stop, slice_in.step)


def shift_index(current_value, shift, extent):
    if current_value is None:
        new_value = None
    else:
        new_value = current_value + shift
        if new_value < 0:
            new_value = extent + new_value
    return new_value


def bound_default_slice(slice_in, start=None, stop=None):
    if slice_in.start is not None:
        start = slice_in.start
    if slice_in.stop is not None:
        stop = slice_in.stop
    return slice(start, stop, slice_in.step)
