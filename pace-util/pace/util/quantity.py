import dataclasses
import warnings
from typing import Dict, Iterable, Sequence, Tuple, Union, cast

import numpy as np

from . import _xarray, constants
from ._boundary_utils import bound_default_slice, shift_boundary_slice_tuple
from .types import NumpyModule


try:
    import cupy
except ImportError:
    cupy = np  # avoids attribute errors while also disabling cupy support
try:
    import gt4py
except ImportError:
    gt4py = None

__all__ = ["Quantity", "QuantityMetadata"]


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
    gt4py_backend: Union[str, None] = None
    "backend to use for gt4py storages"

    @property
    def dim_lengths(self) -> Dict[str, int]:
        """mapping of dimension names to their lengths"""
        return dict(zip(self.dims, self.extent))

    @property
    def np(self) -> NumpyModule:
        """numpy-like module used to interact with the data"""
        if issubclass(self.data_type, cupy.ndarray):
            return cupy
        elif issubclass(self.data_type, np.ndarray):
            return np
        else:
            raise TypeError(
                f"quantity underlying data is of unexpected type {self.data_type}"
            )


class BoundaryArrayView:
    def __init__(self, data, boundary_type, dims, origin, extent):
        self._data = data
        self._boundary_type = boundary_type
        self._dims = dims
        self._origin = origin
        self._extent = extent

    def __getitem__(self, index):
        if len(self._origin) == 0:
            if isinstance(index, tuple) and len(index) > 0:
                raise IndexError("more than one index given for a zero-dimension array")
            elif isinstance(index, slice) and index != slice(None, None, None):
                raise IndexError("cannot slice a zero-dimension array")
            else:
                return self._data  # array[()] does not return an ndarray
        else:
            return self._data[self._get_array_index(index)]

    def __setitem__(self, index, value):
        self._data[self._get_array_index(index)] = value

    def _get_array_index(self, index):
        if isinstance(index, list):
            index = tuple(index)
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > len(self._dims):
            raise IndexError(
                f"{len(index)} is too many indices for a "
                f"{len(self._dims)}-dimensional quantity"
            )
        if len(index) < len(self._dims):
            index = index + (slice(None, None),) * (len(self._dims) - len(index))
        return shift_boundary_slice_tuple(
            self._dims, self._origin, self._extent, self._boundary_type, index
        )

    def sel(self, **kwargs: Union[slice, int]) -> np.ndarray:
        """Convenience method to perform indexing using dimension names
        without knowing dimension order.

        Args:
            **kwargs: slice/index to retrieve for a given dimension name

        Returns:
            view_selection: an ndarray-like selection of the given indices
                on `self.view`
        """
        return self[tuple(kwargs.get(dim, slice(None, None)) for dim in self._dims)]


class BoundedArrayView:
    """
    A container of objects which provide indexing relative to corners and edges
    of the computational domain for convenience.

    Default start and end indices for all dimensions are modified to be the
    start and end of the compute domain. When using edge and corner attributes, it is
    recommended to explicitly write start and end offsets to avoid confusion.

    Indexing on the object itself (view[:]) is offset by the origin, and default
    start and end indices are modified to be the start and end of the compute domain.

    For corner attributes e.g. `northwest`, modified indexing is done for the two
    axes according to the edges which make up the corner. In other words, indexing
    is offset relative to the intersection of the two edges which make the corner.

    For `interior`, start indices of the horizontal dimensions are relative to the
    origin, and end indices are relative to the origin + extent. For example,
    view.interior[0:0, 0:0, :] would retrieve the entire compute domain for an x/y/z
    array, while view.interior[-1:1, -1:1, :] would also include one halo point.
    """

    def __init__(
        self, array, dims: Sequence[str], origin: Sequence[int], extent: Sequence[int]
    ):
        self._data = array
        self._dims = tuple(dims)
        self._origin = tuple(origin)
        self._extent = tuple(extent)
        self._northwest = BoundaryArrayView(
            array, constants.NORTHWEST, dims, origin, extent
        )
        self._northeast = BoundaryArrayView(
            array, constants.NORTHEAST, dims, origin, extent
        )
        self._southwest = BoundaryArrayView(
            array, constants.SOUTHWEST, dims, origin, extent
        )
        self._southeast = BoundaryArrayView(
            array, constants.SOUTHEAST, dims, origin, extent
        )
        self._interior = BoundaryArrayView(
            array, constants.INTERIOR, dims, origin, extent
        )

    @property
    def origin(self) -> Tuple[int, ...]:
        """the start of the computational domain"""
        return self._origin

    @property
    def extent(self) -> Tuple[int, ...]:
        """the shape of the computational domain"""
        return self._extent

    def __getitem__(self, index):
        if len(self.origin) == 0:
            if isinstance(index, tuple) and len(index) > 0:
                raise IndexError("more than one index given for a zero-dimension array")
            elif isinstance(index, slice) and index != slice(None, None, None):
                raise IndexError("cannot slice a zero-dimension array")
            else:
                return self._data  # array[()] does not return an ndarray
        else:
            return self._data[self._get_compute_index(index)]

    def __setitem__(self, index, value):
        self._data[self._get_compute_index(index)] = value

    def _get_compute_index(self, index):
        if not isinstance(index, (tuple, list)):
            index = (index,)
        if len(index) > len(self._dims):
            raise IndexError(
                f"{len(index)} is too many indices for a "
                f"{len(self._dims)}-dimensional quantity"
            )
        index = fill_index(index, len(self._data.shape))
        shifted_index = []
        for entry, origin, extent in zip(index, self.origin, self.extent):
            if isinstance(entry, slice):
                shifted_slice = shift_slice(entry, origin, extent)
                shifted_index.append(
                    bound_default_slice(shifted_slice, origin, origin + extent)
                )
            elif entry is None:
                shifted_index.append(entry)
            else:
                shifted_index.append(entry + origin)
        return tuple(shifted_index)

    @property
    def northwest(self) -> BoundaryArrayView:
        return self._northwest

    @property
    def northeast(self) -> BoundaryArrayView:
        return self._northeast

    @property
    def southwest(self) -> BoundaryArrayView:
        return self._southwest

    @property
    def southeast(self) -> BoundaryArrayView:
        return self._southeast

    @property
    def interior(self) -> BoundaryArrayView:
        return self._interior


def ensure_int_tuple(arg, arg_name):
    return_list = []
    for item in arg:
        try:
            return_list.append(int(item))
        except ValueError:
            raise TypeError(
                f"tuple arg {arg_name}={arg} contains item {item} of "
                f"unexpected type {type(item)}"
            )
    return tuple(return_list)


def _validate_quantity_property_lengths(shape, dims, origin, extent):
    n_dims = len(shape)
    for var, desc in (
        (dims, "dimension names"),
        (origin, "origins"),
        (extent, "extents"),
    ):
        if len(var) != n_dims:
            raise ValueError(
                f"received {len(var)} {desc} for {n_dims} dimensions: {var}"
            )


class Quantity:
    """
    Data container for physical quantities.
    """

    def __init__(
        self,
        data,
        dims: Sequence[str],
        units: str,
        origin: Sequence[int] = None,
        extent: Sequence[int] = None,
        gt4py_backend: Union[str, None] = None,
    ):
        """
        Initialize a Quantity.

        Args:
            data: ndarray-like object containing the underlying data
            dims: dimension names for each axis
            units: units of the quantity
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
            gt4py_backend: backend to use for gt4py storages, if not given this will
                be derived from a Storage if given as the data argument, otherwise the
                storage attribute is disabled and will raise an exception. Will raise
                a TypeError if this is given with a gt4py storage type as data
        """
        if origin is None:
            origin = (0,) * len(dims)  # default origin at origin of array
        else:
            origin = tuple(origin)

        if extent is None:
            extent = tuple(length - start for length, start in zip(data.shape, origin))
        else:
            extent = tuple(extent)

        if isinstance(data, (int, float, list)):
            data = np.asarray(data)
        elif gt4py is not None and isinstance(data, gt4py.storage.storage.Storage):
            if gt4py_backend is not None:
                raise TypeError(
                    "cannot select gt4py backend with keyword argument "
                    "when providing storage as data"
                )
            else:
                gt4py_backend = data.backend
            if isinstance(data, gt4py.storage.storage.GPUStorage):
                self._storage = data
                self._data = data.gpu_view
            elif isinstance(data, gt4py.storage.storage.CPUStorage):
                self._storage = data
                self._data = data.data
            else:
                raise TypeError(
                    "only storages supported are CPUStorage and GPUStorage, "
                    f"got {type(data)}"
                )
        elif gt4py_backend is not None:
            extra_dims = [i for i in dims if i not in constants.SPATIAL_DIMS]
            if len(extra_dims) > 0 or not dims:
                mask = None
            else:
                mask = tuple(
                    [
                        any(dim in coord_dims for dim in dims)
                        for coord_dims in [
                            constants.X_DIMS,
                            constants.Y_DIMS,
                            constants.Z_DIMS,
                        ]
                    ]
                )
            self._storage, self._data = self._initialize_storage(
                data, origin=origin, gt4py_backend=gt4py_backend, mask=mask
            )
        else:
            self._data = data
            self._storage = None

        _validate_quantity_property_lengths(data.shape, dims, origin, extent)
        self._metadata = QuantityMetadata(
            origin=ensure_int_tuple(origin, "origin"),
            extent=ensure_int_tuple(extent, "extent"),
            dims=tuple(dims),
            units=units,
            data_type=type(self._data),
            dtype=data.dtype,
            gt4py_backend=gt4py_backend,
        )
        self._attrs = {}  # type: ignore[var-annotated]
        self._compute_domain_view = BoundedArrayView(
            self.data, self.dims, self.origin, self.extent
        )

    @classmethod
    def from_data_array(
        cls,
        data_array: _xarray.DataArray,
        origin: Sequence[int] = None,
        extent: Sequence[int] = None,
        gt4py_backend: Union[str, None] = None,
    ) -> "Quantity":
        """
        Initialize a Quantity from an xarray.DataArray.

        Args:
            data_array
            origin: first point in data within the computational domain
            extent: number of points along each axis within the computational domain
            gt4py_backend: backend to use for gt4py storages, if not given this will
                be derived from a Storage if given as the data argument, otherwise the
                storage attribute is disabled and will raise an exception
        """
        if "units" not in data_array.attrs:
            raise ValueError("need units attribute to create Quantity from DataArray")
        return cls(
            data_array.values,
            cast(Tuple[str], data_array.dims),
            data_array.attrs["units"],
            origin=origin,
            extent=extent,
            gt4py_backend=gt4py_backend,
        )

    def __repr__(self):
        return (
            f"Quantity(\n    data=\n{self.data},\n    dims={self.dims},\n"
            f"    units={self.units},\n    origin={self.origin},\n"
            f"    extent={self.extent}\n)"
        )

    def sel(self, **kwargs: Union[slice, int]) -> np.ndarray:
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
    def storage(self):
        """A gt4py storage representing the data in this Quantity.

        Will raise TypeError if the gt4py backend was not specified when initializing
        this object, either by providing a Storage for data or explicitly specifying
        a backend.
        """
        if self._storage is None:
            raise TypeError(
                "gt4py backend was not specified when initializing this object"
            )
        return self._storage

    def _initialize_storage(self, data, origin, gt4py_backend: str, mask: Tuple):
        storage = gt4py.storage.storage.empty(
            gt4py_backend,
            default_origin=origin,
            shape=data.shape,
            dtype=data.dtype,
            mask=mask,
            managed_memory=True,  # required to get GPUStorage with only gpu data copy
        )
        storage[...] = data
        # storage must initialize new memory. when GDP-3 is merged, we can instead
        # initialize storage from self._data
        # when GDP-3 is merged, we can instead use the data in self._data to
        # initialize the storage, instead of making a copy.
        if isinstance(storage, gt4py.storage.storage.CPUStorage):
            data = storage.data
        elif isinstance(storage, gt4py.storage.storage.GPUStorage):
            data = storage.gpu_view
        else:
            raise NotImplementedError(
                f"received unexpected storage type {type(storage)} "
                f"for gt4py_backend {gt4py_backend}, did gt4py get updated?"
            )
        return storage, data

    @property
    def metadata(self) -> QuantityMetadata:
        return self._metadata

    @property
    def units(self) -> str:
        """units of the quantity"""
        return self.metadata.units

    @property
    def gt4py_backend(self) -> Union[str, None]:
        return self.metadata.gt4py_backend

    @property
    def attrs(self) -> dict:
        return dict(**self._attrs, units=self._metadata.units)

    @property
    def dims(self) -> Tuple[str, ...]:
        """names of each dimension"""
        return self.metadata.dims

    @property
    def values(self) -> np.ndarray:
        warnings.warn(
            "values exists only for backwards-compatibility with "
            "DataArray and will be removed, use .view[:] instead",
            DeprecationWarning,
        )
        return_array = np.asarray(self.view[:])
        return_array.flags.writeable = False
        return return_array

    @property
    def view(self) -> BoundedArrayView:
        """a view into the computational domain of the underlying data"""
        return self._compute_domain_view

    @property
    def data(self) -> Union[np.ndarray, cupy.ndarray]:
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
    def data_array(self) -> _xarray.DataArray:
        return _xarray.DataArray(self.view[:], dims=self.dims, attrs=self.attrs)

    @property
    def np(self) -> NumpyModule:
        return self.metadata.np

    def transpose(self, target_dims: Sequence[Union[str, Iterable[str]]]) -> "Quantity":
        """Change the dimension order of this Quantity.

        If you know you are working with cell-centered variables, you can do:

        >>> from pace.util import X_DIM, Y_DIM, Z_DIM
        >>> transposed_quantity = quantity.transpose([X_DIM, Y_DIM, Z_DIM])

        To support re-ordering without checking whether quantities are on
        cell centers or interfaces, the API supports giving a list of dimension names
        for dimensions. For example, to re-order to X-Y-Z dimensions regardless of the
        grid the variable is on, one could do:

        >>> from pace.util import X_DIMS, Y_DIMS, Z_DIMS
        >>> transposed_quantity = quantity.transpose([X_DIMS, Y_DIMS, Z_DIMS])

        Args:
            target_dims: a list of output dimensions. Instead of a single dimension
                name, an iterable of dimensions can be used instead for any entries.
                For example, you may want to use pace.util.X_DIMS to place an
                x-dimension without knowing whether it is on cell centers or interfaces.

        Returns:
            transposed: Quantity with the requested output dimension order

        Raises:
            ValueError: if any of the target dimensions do not exist on this Quantity,
                or if this Quantity contains multiple values from an iterable entry
        """
        target_dims = _collapse_dims(target_dims, self.dims)
        transpose_order = [self.dims.index(dim) for dim in target_dims]
        transposed = Quantity(
            self.np.transpose(self.data, transpose_order),  # type: ignore[attr-defined]
            dims=transpose_sequence(self.dims, transpose_order),
            units=self.units,
            origin=transpose_sequence(self.origin, transpose_order),
            extent=transpose_sequence(self.extent, transpose_order),
            gt4py_backend=self.gt4py_backend,
        )
        transposed._attrs = self._attrs
        return transposed


def transpose_sequence(sequence, order):
    return sequence.__class__(sequence[i] for i in order)


def _collapse_dims(target_dims, dims):
    return_list = []
    for target in target_dims:
        if isinstance(target, str):
            if target in dims:
                return_list.append(target)
            else:
                raise ValueError(
                    f"requested dimension {target} is not defined in "
                    f"quantity dimensions {dims}"
                )
        elif isinstance(target, Iterable):
            matches = [d for d in target if d in dims]
            if len(matches) > 1:
                raise ValueError(
                    f"multiple matches for {target} found in quantity dimensions {dims}"
                )
            elif len(matches) == 0:
                raise ValueError(
                    f"no matches for {target} found in quantity dimensions {dims}"
                )
            else:
                return_list.append(matches[0])
    return return_list


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
