import dataclasses
import warnings
from typing import Any, Dict, Iterable, Sequence, Tuple, Union, cast

import numpy as np

from . import _xarray, constants
from ._boundary_utils import bound_default_slice, shift_boundary_slice_tuple
from ._optional_imports import cupy, dace, gt4py
from .types import NumpyModule


if cupy is None:
    import numpy as cupy

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


@dataclasses.dataclass
class QuantityHaloSpec:
    """Describe the memory to be exchanged, including size of the halo."""

    n_points: int
    strides: Tuple[int]
    itemsize: int
    shape: Tuple[int]
    origin: Tuple[int, ...]
    extent: Tuple[int, ...]
    dims: Tuple[str, ...]
    numpy_module: NumpyModule
    dtype: Any


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
            # If converting basic data, use a numpy ndarray.
            data = np.asarray(data)

        if not isinstance(data, (np.ndarray, cupy.ndarray)):
            raise TypeError(
                f"Only supports numpy.ndarray and cupy.ndarray, got {type(data)}"
            )

        if gt4py_backend is not None:
            gt4py_backend_cls = gt4py.cartesian.backend.from_name(gt4py_backend)
            assert gt4py_backend_cls is not None
            is_optimal_layout = gt4py_backend_cls.storage_info["is_optimal_layout"]

            dimensions: Tuple[Union[str, int], ...] = tuple(
                [
                    axis
                    if any(dim in axis_dims for axis_dims in constants.SPATIAL_DIMS)
                    else str(data.shape[index])
                    for index, (dim, axis) in enumerate(
                        zip(dims, ("I", "J", "K", *([None] * (len(dims) - 3))))
                    )
                ]
            )

            self._data = (
                data
                if is_optimal_layout(data, dimensions)
                else self._initialize_data(
                    data,
                    origin=origin,
                    gt4py_backend=gt4py_backend,
                    dimensions=dimensions,
                )
            )
        else:
            if data is None:
                raise TypeError("requires 'data' to be passed")
            # We have no info about the gt4py_backend, so just assign it.
            self._data = data

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

    def halo_spec(self, n_halo: int) -> QuantityHaloSpec:
        return QuantityHaloSpec(
            n_halo,
            self.data.strides,
            self.data.itemsize,
            self.data.shape,
            self.metadata.origin,
            self.metadata.extent,
            self.metadata.dims,
            self.np,
            self.metadata.dtype,
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

    def _initialize_data(self, data, origin, gt4py_backend: str, dimensions: Tuple):
        """Allocates an ndarray with optimal memory layout, and copies the data over."""
        storage = gt4py.storage.from_array(
            data,
            data.dtype,
            backend=gt4py_backend,
            aligned_index=origin,
            dimensions=dimensions,
        )
        return storage

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

    @property
    def __array_interface__(self):
        return self.data.__array_interface__

    @property
    def __cuda_array_interface__(self):
        return self.data.__cuda_array_interface__

    @property
    def shape(self):
        return self.data.shape

    def __descriptor__(self) -> Any:
        """The descriptor is a property that dace uses.
        This relies on `dace` capacity to read out data from the buffer protocol.
        If the internal data given doesn't follow the protocol it will most likely
        fail.
        """
        if dace:
            return dace.data.create_datadescriptor(self.data)
        else:
            raise ImportError(
                "Attempt to use DaCe orchestrated backend but "
                "DaCe module is not available."
            )

    def transpose(self, target_dims: Sequence[Union[str, Iterable[str]]]) -> "Quantity":
        """Change the dimension order of this Quantity.

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

        Examples:
            Let's say we have a cell-centered variable:

            >>> import pace.util
            >>> import numpy as np
            >>> quantity = pace.util.Quantity(
            ...     data=np.zeros([2, 3, 4]),
            ...     dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            ...             units="m",
            ... )

            If you know you are working with cell-centered variables, you can do:

            >>> from pace.util import X_DIM, Y_DIM, Z_DIM
            >>> transposed_quantity = quantity.transpose([X_DIM, Y_DIM, Z_DIM])

            To support re-ordering without checking whether quantities are on
            cell centers or interfaces, the API supports giving a list of dimension
            names for dimensions. For example, to re-order to X-Y-Z dimensions
            regardless of the grid the variable is on, one could do:

            >>> from pace.util import X_DIMS, Y_DIMS, Z_DIMS
            >>> transposed_quantity = quantity.transpose([X_DIMS, Y_DIMS, Z_DIMS])
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
