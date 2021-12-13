import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID, uuid1

import numpy as np

from .buffer import Buffer
from .cuda_kernels import (
    pack_scalar_f64_kernel,
    pack_vector_f64_kernel,
    unpack_scalar_f64_kernel,
    unpack_vector_f64_kernel,
)
from .quantity import Quantity
from .rotate import rotate_scalar_data, rotate_vector_data
from .types import NumpyModule
from .utils import device_synchronize


try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass
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


# ------------------------------------------------------------------------
# Simple pool of streams to lower the driver pressure
# Use _pop/_push_stream to manipulate the pool

STREAM_POOL: List["cp.cuda.Stream"] = []


def _pop_stream() -> "cp.cuda.Stream":
    if len(STREAM_POOL) == 0:
        return cp.cuda.Stream(non_blocking=True)
    return STREAM_POOL.pop()


def _push_stream(stream: "cp.cuda.Stream"):
    STREAM_POOL.append(stream)


# ------------------------------------------------------------------------
# Indices array

# Keyed cached - key is a str at the moment to go around the fact that
# a slice is not hashable. getting a string from
# Tuple(slices, rotation, shape, strides, itemsize) e.g. # noqa
# str(Tuple[Any, int, Tuple[int], Tuple[int], int]) # noqa
INDICES_CACHE: Dict[str, "cp.ndarray"] = {}


def _build_flatten_indices(
    key,
    shape,
    slices: Tuple[slice],
    dims,
    strides,
    itemsize: int,
    rotate: bool,
    rotation: int,
) -> "cp.ndarray":
    """Build an array of indexing from a slice & memory description to
    build an indexation into the "flatten" memory.

    Go from a memory layout (strides, itemsize, shape) and slices into it to a
    single array of indices. We leverage numpy iterator and calculate from
    the multi_index using memory layout the index into the original memory buffer.
    """

    # Have to go down to numpy to leverage indices calculation
    arr_indices = np.empty(shape, dtype=np.int32, order="C")[slices]

    # Get offset from first index
    offset_dims = []
    for s in slices:
        offset_dims.append(s.start)
    offset_to_slice = sum(np.array(offset_dims) * strides) // itemsize

    # Flatten the index into an indices array
    with np.nditer(
        arr_indices,
        flags=["multi_index"],
        op_flags=["writeonly"],
        order="K",
    ) as it:
        for array_value in it:
            offset = sum(np.array(it.multi_index) * strides) // itemsize
            array_value[...] = offset_to_slice + offset

    if rotate:
        # sending data across the boundary will rotate the data
        # n_clockwise_rotations times, due to the difference in axis orientation.
        # Thus we rotate that number of times counterclockwise before sending,
        # to get the right final orientation. We apply those rotations to the
        # indices here to prepare for a straightforward copy in cu kernel
        arr_indices = rotate_scalar_data(arr_indices, dims, cp, -rotation)
    return cp.asarray(arr_indices.flatten(order="C"))


# ------------------------------------------------------------------------
# HaloDataTransformer helpers


def _slices_size(slices: Tuple[slice, ...]) -> int:
    """Compute linear size from slices."""
    length = 1
    for s in slices:
        assert s.step is None
        length *= abs(s.start - s.stop)
    return length


@dataclass
class HaloExchangeSpec:
    """Memory description of the data exchanged.

    The data stored here target a single exchange, with an optional
    rotation to give prior to pack. Slices are tupled following the
    convention of one slice per dimension

    Args:
        specification: memory layout of the data
        pack_slices: indexing to pack, one slice per dimension
        pack_clockwise_rotation:  number of 90-degree rotations to perform
            before packing
        unpack_slices: indexing to unpack, one slice per dimension
    """

    specification: QuantityHaloSpec
    pack_slices: Tuple[slice, ...]
    pack_clockwise_rotation: int
    unpack_slices: Tuple[slice, ...]

    def __post_init__(self):
        self._id = uuid1()
        self.pack_buffer_size = _slices_size(self.pack_slices)
        self._unpack_buffer_size = _slices_size(self.unpack_slices)


class _HaloDataTransformerType(Enum):
    """Dimensionality of the data in the packed buffer."""

    UNKNOWN = 0
    SCALAR = 1
    VECTOR = 2


# ------------------------------------------------------------------------
# HaloDataTransformer classes


class HaloDataTransformer(abc.ABC):
    """Transform data to exchange in a format optimized for network communication.

    Current strategy: pack/unpack multiple nD array into/from a single buffer.
    Offers a pack and an unpack buffer to use for communicating data.

    The class is responsible for packing & unpacking, not communication.
    Order of operations:
    - get HaloDataTransformer via get() with N transformation
      with the proper halo specifications.
      At the end of get() a _compile() will be triggered, reading
      the internal buffers.
    - call async_pack(quantities) to start packing the quantities in the
      internal buffer.
    - synchronize() to make sure all operations are finished or use get_pack_buffer()
      when ready to communicate which will internally call synchronize.
    [... user should communicate the buffers...]
    - call async_unpack(quantities) to start unpacking
    - call synchronize() to finish all the unpacking operations and make sure
      the quantities passed in async_unpack have been updated.

    The class will hold onto the buffers up until deletion, where they will be
    returned to an internal buffer pool.
    """

    _pack_buffer: Optional[Buffer]
    _unpack_buffer: Optional[Buffer]

    _infos_x: Tuple[HaloExchangeSpec, ...]
    _infos_y: Tuple[HaloExchangeSpec, ...]

    def __init__(
        self,
        np_module: NumpyModule,
        exchange_descriptors_x: Sequence[HaloExchangeSpec],
        exchange_descriptors_y: Optional[Sequence[HaloExchangeSpec]] = None,
    ) -> None:
        """
        Args:
            np_module: numpy-like module for allocation
            exchange_descriptors_x: list of memory information describing an exchange.
                Used for scalar data and the x-component of vectors.
            exchange_descriptors_y: list of memory information describing an exchange.
                Optional, used for the y-component of vectors only. If `none` the
                data will packed as a scalar.
        """
        self._type = (
            _HaloDataTransformerType.SCALAR
            if exchange_descriptors_y is None
            else _HaloDataTransformerType.VECTOR
        )
        if exchange_descriptors_y is not None and len(exchange_descriptors_y) != len(
            exchange_descriptors_x
        ):
            raise RuntimeError(
                "Vector halo exchange must have same exchange data for X and Y"
            )
        self._np_module = np_module
        self._infos_x = tuple(exchange_descriptors_x)
        self._infos_y = (
            tuple(exchange_descriptors_y)
            if exchange_descriptors_y is not None
            else tuple()
        )
        self._pack_buffer = None
        self._unpack_buffer = None
        self._compile()

    def finalize(self):
        """Del routine, making sure all buffers were inserted back into cache."""
        # Synchronize all work
        self.synchronize()

        # Push the buffers back in the cache
        Buffer.push_to_cache(self._pack_buffer)
        self._pack_buffer = None
        Buffer.push_to_cache(self._unpack_buffer)
        self._unpack_buffer = None

    @staticmethod
    def get(
        np_module: NumpyModule,
        exchange_descriptors_x: Sequence[HaloExchangeSpec],
        exchange_descriptors_y: Optional[Sequence[HaloExchangeSpec]] = None,
    ) -> "HaloDataTransformer":
        """Construct a module from a numpy-like module.

        Args:
            np_module: numpy-like module to determin child transformer type.
            exchange_descriptors_x: list of memory information describing an exchange.
                Used for scalar data and the x-component of vectors.
            exchange_descriptors_y: list of memory information describing an exchange.
                Optional, used for the y-component of vectors only. If `none` the data
                will packed as a scalar.

        Returns:
            an initialized packed buffer.
        """
        if np_module is np:
            return HaloDataTransformerCPU(
                np,
                exchange_descriptors_x,
                exchange_descriptors_y=exchange_descriptors_y,
            )
        elif np_module is cp:
            return HaloDataTransformerGPU(
                cp,
                exchange_descriptors_x,
                exchange_descriptors_y=exchange_descriptors_y,
            )

        raise NotImplementedError(
            f"Quantity module {np_module} has no HaloDataTransformer implemented"
        )

    def get_unpack_buffer(self) -> Buffer:
        """Retrieve unpack buffer.

        Synchronizes operations.
        """
        if self._unpack_buffer is None:
            raise RuntimeError("Recv buffer can't be retrieved before allocate()")
        self.synchronize()
        return self._unpack_buffer

    def get_pack_buffer(self) -> Buffer:
        """Retrieve pack buffer.

        Synchronizes operations.
        """
        if self._pack_buffer is None:
            raise RuntimeError("Send buffer can't be retrieved before allocate()")
        self.synchronize()
        return self._pack_buffer

    def _compile(self):
        """Allocate contiguous memory buffers from description queued."""

        # Compute required size
        buffer_size = 0
        dtype = None
        for edge_x in self._infos_x:
            buffer_size += edge_x.pack_buffer_size
            dtype = edge_x.specification.dtype
        if self._type is _HaloDataTransformerType.VECTOR:
            for edge_y in self._infos_y:
                buffer_size += edge_y.pack_buffer_size

        # Retrieve two properly sized buffers
        self._pack_buffer = Buffer.pop_from_cache(
            self._np_module.zeros, (buffer_size), dtype
        )
        self._unpack_buffer = Buffer.pop_from_cache(
            self._np_module.zeros, (buffer_size), dtype
        )

    def ready(self) -> bool:
        """Check if the buffers are ready for communication."""
        return self._pack_buffer is not None and self._unpack_buffer is not None

    @abc.abstractmethod
    def async_pack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Pack all given quantities into a single send Buffer.

        Does not guarantee the buffer returned by `get_unpack_buffer` has
        received data, doing so requires calling `synchronize`.
        Reaching for the buffer via get_pack_buffer() will call synchronize().

        Args:
            quantities_x: scalar or vector x-component quantities to pack,
                if one is vector they must all be vector

            quantities_y: if quantities are vector, the y-component
                quantities.
        """
        pass

    @abc.abstractmethod
    def async_unpack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Unpack the buffer into destination quantities.

        Does not guarantee the buffer returned by `get_unpack_buffer` has
        received data, doing so requires calling `synchronize`.
        Reaching for the buffer via get_unpack_buffer() will call synchronize().

        Args:
            quantities_x: scalar or vector x-component quantities to be unpacked into,
                if one is vector they must all be vector
            quantities_y: if quantities are vector, the y-component
                quantities.
        """
        pass

    @abc.abstractmethod
    def synchronize(self):
        """Synchronize all operations.

        Guarantees all memory is now safe to access.
        """
        pass


class HaloDataTransformerCPU(HaloDataTransformer):
    """Pack/unpack data in a single buffer using numpy flattening & slicing.

    Default behavior, could be done with any numpy-like library.
    """

    def synchronize(self):
        if self._pack_buffer is not None:
            self._pack_buffer.finalize_memory_transfer()
        if self._unpack_buffer is not None:
            self._unpack_buffer.finalize_memory_transfer()

    def async_pack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._pack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._pack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} pack")

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened

    def _pack_scalar(self, quantities: List[Quantity]):
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)}"
                    f" is different that edges count {len(self._infos_x)}"
                )
            # TODO Per quantity check

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity, info_x in zip(quantities, self._infos_x):
            data_size = _slices_size(info_x.pack_slices)
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation.\
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            source_view = rotate_scalar_data(
                quantity.data[info_x.pack_slices],
                quantity.dims,
                quantity.np,
                -info_x.pack_clockwise_rotation,
            )
            self._pack_buffer.assign_from(
                source_view.flatten(),
                buffer_slice=np.index_exp[offset : offset + data_size],
            )
            offset += data_size

    def _pack_vector(self, quantities_x: List[Quantity], quantities_y: List[Quantity]):
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)})"
                    " is different that specifications count "
                    f"(x: {len(self._infos_x)}, y: {len(self._infos_y)}"
                )
            # TODO Per quantity check

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        assert len(quantities_y) == len(quantities_x)
        assert len(self._infos_x) == len(self._infos_y)
        offset = 0
        for (
            quantity_x,
            quantity_y,
            info_x,
            info_y,
        ) in zip(quantities_x, quantities_y, self._infos_x, self._infos_y):
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            x_view, y_view = rotate_vector_data(
                quantity_x.data[info_x.pack_slices],
                quantity_y.data[info_y.pack_slices],
                -info_x.pack_clockwise_rotation,
                quantity_x.dims,
                quantity_x.np,
            )

            # Pack X/Y data slices in the buffer
            self._pack_buffer.assign_from(
                x_view.flatten(),
                buffer_slice=np.index_exp[offset : offset + x_view.size],
            )
            offset += x_view.size
            self._pack_buffer.assign_from(
                y_view.flatten(),
                buffer_slice=np.index_exp[offset : offset + y_view.size],
            )
            offset += y_view.size

    def async_unpack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._unpack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._unpack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} unpack")

        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened

    def _unpack_scalar(self, quantities: List[Quantity]):
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)}"
                    f" is different that specifications count {len(self._infos_x)}"
                )
            # TODO Per quantity check

        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity, info_x in zip(quantities, self._infos_x):
            quantity_view = quantity.data[info_x.unpack_slices]
            data_size = _slices_size(info_x.unpack_slices)
            self._unpack_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + data_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += data_size

    def _unpack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)})"
                    " is different that specifications count "
                    f"(x: {len(self._infos_x)}, y: {len(self._infos_y)})"
                )
            # TODO Per quantity check

        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity_x, quantity_y, info_x, info_y in zip(
            quantities_x, quantities_y, self._infos_x, self._infos_y
        ):
            quantity_view = quantity_x.data[info_x.unpack_slices]
            data_size = _slices_size(info_x.unpack_slices)
            self._unpack_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + data_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += data_size
            quantity_view = quantity_y.data[info_y.unpack_slices]
            data_size = _slices_size(info_y.unpack_slices)
            self._unpack_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + data_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += data_size


class HaloDataTransformerGPU(HaloDataTransformer):
    """Pack/unpack data in a single buffer using CUDA Kernels.

    In order to efficiently pack/unpack on the GPU to a single GPU buffer
    we use streamed (e.g. async) kernels per quantity per edge to send. The
    kernels are store in `cuda_kernels.py`, they both follow the same simple pattern
    by reading the indices to the device memory of the data to pack/unpack.
    `_flatten_indices` is the routine that take the layout of the memory and
    the slice and compute an array of index into the original memory.
    """

    # Temporary "safe" code path
    #  _CODE_PATH_DEVICE_WIDE_SYNC: turns off streaming and issue a single
    #   device wide synchronization call instead
    _CODE_PATH_DEVICE_WIDE_SYNC = False

    @dataclass
    class _CuKernelArgs:
        """All arguments required for the CUDA kernels."""

        stream: "cp.cuda.Stream"
        x_send_indices: "cp.ndarray"
        x_recv_indices: "cp.ndarray"
        y_send_indices: Optional["cp.ndarray"]
        y_recv_indices: Optional["cp.ndarray"]

    def __init__(
        self,
        np_module: NumpyModule,
        exchange_descriptors_x: Sequence[HaloExchangeSpec],
        exchange_descriptors_y: Optional[Sequence[HaloExchangeSpec]] = None,
    ) -> None:
        self._cu_kernel_args: Dict[UUID, HaloDataTransformerGPU._CuKernelArgs] = {}
        super().__init__(
            np_module,
            exchange_descriptors_x,
            exchange_descriptors_y=exchange_descriptors_y,
        )

    def _flatten_indices(
        self,
        exchange_data: HaloExchangeSpec,
        slices: Tuple[slice],
        rotate: bool,
    ) -> "cp.ndarray":
        """Extract a flat array of indices from the memory layout and the slice.

        Also take care of rotating the indices to account for axis orientation.
        """
        key = str(
            (
                slices,
                exchange_data.pack_clockwise_rotation,
                exchange_data.specification.shape,
                exchange_data.specification.strides,
                exchange_data.specification.itemsize,
            )
        )

        # We use a lazy caching mechanism here because in our use case
        # (halo exchange) there is a limited set of index patterns but a
        # large number of exchanges.
        if key not in INDICES_CACHE.keys():
            INDICES_CACHE[key] = _build_flatten_indices(
                key,
                exchange_data.specification.shape,
                slices,
                exchange_data.specification.dims,
                exchange_data.specification.strides,
                exchange_data.specification.itemsize,
                rotate,
                exchange_data.pack_clockwise_rotation,
            )

        # We don't return a copy since the indices are read-only in the algorithm
        return INDICES_CACHE[key]

    def _compile(self):
        # Super to get buffer allocation
        super()._compile()
        # Allocate the streams & build the indices arrays
        if self._type == _HaloDataTransformerType.SCALAR:
            for info_x in self._infos_x:
                self._cu_kernel_args[info_x._id] = HaloDataTransformerGPU._CuKernelArgs(
                    stream=_pop_stream(),
                    x_send_indices=self._flatten_indices(
                        info_x, info_x.pack_slices, True
                    ),
                    x_recv_indices=self._flatten_indices(
                        info_x, info_x.unpack_slices, False
                    ),
                    y_send_indices=None,
                    y_recv_indices=None,
                )
        else:
            assert self._type == _HaloDataTransformerType.VECTOR
            for info_x, info_y in zip(self._infos_x, self._infos_y):
                self._cu_kernel_args[info_x._id] = HaloDataTransformerGPU._CuKernelArgs(
                    stream=_pop_stream(),
                    x_send_indices=self._flatten_indices(
                        info_x, info_x.pack_slices, True
                    ),
                    x_recv_indices=self._flatten_indices(
                        info_x, info_x.unpack_slices, False
                    ),
                    y_send_indices=self._flatten_indices(
                        info_y, info_y.pack_slices, True
                    ),
                    y_recv_indices=self._flatten_indices(
                        info_y, info_y.unpack_slices, False
                    ),
                )

    def synchronize(self):
        if self._CODE_PATH_DEVICE_WIDE_SYNC:
            self._safe_synchronize()
        else:
            self._streamed_synchronize()

    def _streamed_synchronize(self):
        for cu_kernel in self._cu_kernel_args.values():
            cu_kernel.stream.synchronize()

    def _safe_synchronize(self):
        device_synchronize()

    def _get_stream(self, stream) -> "cp.cuda.stream":
        if self._CODE_PATH_DEVICE_WIDE_SYNC:
            return cp.cuda.Stream.null
        else:
            return stream

    def async_pack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Pack the quantities into a single buffer via streamed cuda kernels

        Writes into self._pack_buffer using self._x_infos and self._y_infos
        to read the offsets and sizes per quantity.

        Args:
            quantities_x: list of quantities to pack. Must fit the specifications given
                at init time.
            quantities_y: Same as above but optional, used only for vector transfer.
        """

        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._opt_pack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._opt_pack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} pack")

    def _opt_pack_scalar(self, quantities: List[Quantity]):
        """Specialized packing for scalar. See async_pack docs for usage."""
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)}"
                    f" is different that specifications count {len(self._infos_x)}"
                )
            # TODO Per quantity check

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for info_x, quantity in zip(self._infos_x, quantities):
            cu_kernel_args = self._cu_kernel_args[info_x._id]

            # Use private stream
            with self._get_stream(cu_kernel_args.stream):
                if quantity.metadata.dtype != np.float64:
                    raise RuntimeError(f"Kernel requires f64 given {np.float64}")

                # Launch kernel
                blocks = 128
                grid_x = (info_x.pack_buffer_size // blocks) + 1
                if pack_scalar_f64_kernel is None:
                    RuntimeError("CUDA nvrtc failed")
                else:
                    pack_scalar_f64_kernel(
                        (grid_x,),
                        (blocks,),
                        (
                            quantity.data[:],  # source_array
                            cu_kernel_args.x_send_indices,  # indices
                            info_x.pack_buffer_size,  # nIndex
                            offset,
                            self._pack_buffer.array,
                        ),
                    )

                # Next transformer offset into send buffer
                offset += info_x.pack_buffer_size

    def _opt_pack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        """Specialized packing for vectors. See async_pack docs for usage."""
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)}"
                    " is different that specifications count "
                    f"(x: {len(self._infos_x)}, y: {len(self._infos_y)}"
                )
            # TODO Per quantity check
        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        assert len(self._infos_x) == len(self._infos_y)
        assert len(quantities_x) == len(quantities_y)
        offset = 0
        for (
            quantity_x,
            quantity_y,
            info_x,
            info_y,
        ) in zip(quantities_x, quantities_y, self._infos_x, self._infos_y):
            cu_kernel_args = self._cu_kernel_args[info_x._id]

            # Use private stream
            with self._get_stream(cu_kernel_args.stream):

                # Buffer sizes
                transformer_size = info_x.pack_buffer_size + info_y.pack_buffer_size

                if quantity_x.metadata.dtype != np.float64:
                    raise RuntimeError(f"Kernel requires f64 given {np.float64}")

                # Launch kernel
                blocks = 128
                grid_x = (transformer_size // blocks) + 1
                if pack_vector_f64_kernel is None:
                    RuntimeError("CUDA nvrtc failed")
                else:
                    pack_vector_f64_kernel(
                        (grid_x,),
                        (blocks,),
                        (
                            quantity_x.data[:],  # source_array_x
                            quantity_y.data[:],  # source_array_y
                            cu_kernel_args.x_send_indices,  # indices_x
                            cu_kernel_args.y_send_indices,  # indices_y
                            info_x.pack_buffer_size,  # nIndex_x
                            info_y.pack_buffer_size,  # nIndex_y
                            offset,
                            (-info_x.pack_clockwise_rotation) % 4,  # rotation
                            self._pack_buffer.array,
                        ),
                    )

                # Next transformer offset into send buffer
                offset += transformer_size

    def async_unpack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Unpack the quantities from a single buffer via streamed cuda kernels

        Reads from self._unpack_buffer using self._x_infos and self._y_infos
        to read the offsets and sizes per quantity.

        Args:
            quantities_x: list of quantities to unpack. Must fit
                the specifications given at init time.
            quantities_y: Same as above but optional, used only for vector transfer.
        """
        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._opt_unpack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._opt_unpack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} unpack")

    def _opt_unpack_scalar(self, quantities: List[Quantity]):
        """Specialized unpacking for scalars. See async_unpack docs for usage."""
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)})"
                    f" is different that specifications count ({len(self._infos_x)})"
                )
            # TODO Per quantity check
        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity, info_x in zip(quantities, self._infos_x):
            cu_kernel_args = self._cu_kernel_args[info_x._id]

            # Use private stream
            with self._get_stream(cu_kernel_args.stream):

                # Launch kernel
                blocks = 128
                grid_x = (info_x._unpack_buffer_size // blocks) + 1
                if unpack_scalar_f64_kernel is None:
                    RuntimeError("CUDA nvrtc failed")
                else:
                    unpack_scalar_f64_kernel(
                        (grid_x,),
                        (blocks,),
                        (
                            self._unpack_buffer.array,  # source_buffer
                            cu_kernel_args.x_recv_indices,  # indices
                            info_x._unpack_buffer_size,  # nIndex
                            offset,
                            quantity.data[:],  # destination_array
                        ),
                    )

                # Next transformer offset into recv buffer
                offset += info_x._unpack_buffer_size

    def _opt_unpack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        """Specialized unpacking for vectors. See async_unpack docs for usage."""
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)}"
                    " is different that specifications count "
                    f"(x: {len(self._infos_x)}, y: {len(self._infos_y)}"
                )
            # TODO Per quantity check
        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        assert len(self._infos_x) == len(self._infos_y)
        assert len(quantities_x) == len(quantities_y)
        offset = 0
        for (
            quantity_x,
            quantity_y,
            info_x,
            info_y,
        ) in zip(quantities_x, quantities_y, self._infos_x, self._infos_y):
            # We only have writte a f64 kernel
            if quantity_x.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            cu_kernel_args = self._cu_kernel_args[info_x._id]

            # Use private stream
            with self._get_stream(cu_kernel_args.stream):

                # Buffer sizes
                edge_size = info_x._unpack_buffer_size + info_y._unpack_buffer_size

                # Launch kernel
                blocks = 128
                grid_x = (edge_size // blocks) + 1
                if unpack_vector_f64_kernel is None:
                    RuntimeError("CUDA nvrtc failed")
                else:
                    unpack_vector_f64_kernel(
                        (grid_x,),
                        (blocks,),
                        (
                            self._unpack_buffer.array,
                            cu_kernel_args.x_recv_indices,  # indices_x
                            cu_kernel_args.y_recv_indices,  # indices_y
                            info_x._unpack_buffer_size,  # nIndex_x
                            info_y._unpack_buffer_size,  # nIndex_y
                            offset,
                            quantity_x.data[:],  # destination_array_x
                            quantity_y.data[:],  # destination_array_y
                        ),
                    )

                # Next transformer offset into send buffer
                offset += edge_size

    def finalize(self):
        super().finalize()
        # Push the streams back in the pool
        for cu_info in self._cu_kernel_args.values():
            _push_stream(cu_info.stream)
