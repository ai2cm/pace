from typing import Tuple
import dataclasses
import numpy as np
try:
    import cupy
except ImportError:
    cupy = np  # avoids attribute errors while also disabling cupy support
try:
    import gt4py
except ImportError:
    gt4py = None


@dataclasses.dataclass
class QuantityMetadata:
    dims: Tuple[str, ...]
    units: str
    data_type: type
    dtype: type

    @classmethod
    def from_quantity(cls, quantity):
        return cls(dims=quantity.dims, units=quantity.units, data_type=type(quantity.data), dtype=quantity.data.dtype)

    @property
    def np(self):
        if isinstance(self.data_type, np.ndarray):
            return np
        elif isinstance(self.data_type, cupy.ndarray):
            return cupy
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {type(self._data)}"
            )


class Quantity:

    def __init__(self, data, dims, units, origin=None, extent=None):
        self._dims = tuple(dims)
        self._attrs = {'units': units}
        self._data = data
        if origin is None:
            self._origin = (0,) * len(dims)  # default origin at origin of array
        else:
            self._origin = tuple(origin)
        if extent is None:
            self._extent = tuple(length - start for length, start in zip(data.shape, self._origin))
        else:
            self._extent = tuple(extent)
        self._compute_view = BoundedArrayView(self._data, self._origin, self._extent)

    @classmethod
    def from_xarray(cls, data_array, origin=None, extent=None):
        if 'units' not in data_array.attrs:
            raise ValueError('need units attribute to create Quantity from DataArray')
        return cls(
            data_array.values,
            data_array.dims,
            data_array.attrs['units'],
            origin=origin,
            extent=extent
        )

    @classmethod
    def from_data_array(cls, gt4py_storage, dims, units):
        raise NotImplementedError()

    @property
    def units(self):
        return self._attrs['units']

    @property
    def attrs(self):
        return self._attrs

    @property
    def dims(self):
        return self._dims
    
    @property
    def values(self):
        return_array = np.asarray(self._data)
        return_array.flags.writeable = False
        return return_array

    @property
    def compute_domain(self):
        return_list = []
        for start, length in zip(self._origin, self._extent):
            return_list.append(slice(start, start + length))
        return tuple(return_list)
    
    @property
    def compute_view(self):
        return self._compute_view

    @property
    def data(self):
        return self._data

    @property
    def origin(self):
        return self._origin
    
    @property
    def extent(self):
        return self._extent
    
    @property
    def storage(self):
        raise NotImplementedError()

    @property
    def data_array(self):
        raise NotImplementedError()

    @property
    def np(self):
        if isinstance(self._data, np.ndarray):
            return np
        elif isinstance(self._data, cupy.ndarray):
            return cupy
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {type(self._data)}"
            )


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
        return self._data.__getitem__(self._get_compute_index(index))

    def __setitem__(self, index, value):
        return self._data.__setitem__(self._get_compute_index(index), value)

    def _get_compute_index(self, index):
        if not isinstance(index, (tuple, list)):
            index = (index,)
        index = fill_index(index, len(self._data.shape))
        shifted_index = []
        for entry, origin, extent in zip(index, self.origin, self.extent):
            if isinstance(entry, slice):
                shifted_slice = shift_slice(entry, -origin)
                shifted_index.append(bound_default_slice(shifted_slice, 0, extent))
        return tuple(shifted_index)


def fill_index(index, length):
    return tuple(index) + (slice(None, None, None),) * (length - len(index))


def shift_slice(slice_in, shift):
    if slice_in.start is None:
        start = None
    else:
        start = slice_in.start - shift
    if slice_in.stop is None:
        stop = None
    else:
        stop = slice_in.stop - shift
    return slice(start, stop, slice_in.step)


def bound_default_slice(slice_in, start=None, stop=None):
    if slice_in.start is not None:
        start = slice_in.start
    if slice_in.stop is not None:
        stop = slice_in.stop
    return slice(start, stop, slice_in.step)
