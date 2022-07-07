from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from . import constants
from ._timing import NullTimer, Timer
from .boundary import Boundary
from .buffer import Buffer
from .halo_data_transformer import (
    HaloDataTransformer,
    HaloExchangeSpec,
    QuantityHaloSpec,
)
from .quantity import Quantity
from .rotate import rotate_scalar_data
from .types import AsyncRequest, NumpyModule
from .utils import device_synchronize


if TYPE_CHECKING:
    from .communicator import Communicator

_HaloSendTuple = Tuple[AsyncRequest, Buffer]
_HaloRecvTuple = Tuple[AsyncRequest, Buffer, np.ndarray]
_HaloRequestSendList = List[_HaloSendTuple]
_HaloRequestRecvList = List[_HaloRecvTuple]


class HaloUpdater:
    """Exchange halo information between ranks.

    The class is responsible for the entire exchange and uses the __init__
    to precompute the maximum of information to have minimum overhead at runtime.
    Therefore it should be cached for early and re-used at runtime.

    - from_scalar_specifications/from_vector_specifications are used to
      create a HaloUpdater from a list of memory specifications
    - update and start/wait trigger the halo exchange
    - the class creates a "pattern" of exchange that can fit
      any memory given to do/start
    - temporary references to the Quanitites are held between start and wait
    """

    def __init__(
        self,
        comm: "Communicator",
        tag: int,
        transformers: Dict[int, HaloDataTransformer],
        timer: Timer,
    ):
        """Build the updater.

        Args:
            comm: communicator responsible for send/recv commands.
            tag: network tag to be used for communication
            transformers: mapping from destination rank to transformers used to
                pack/unpack before and after communication
            timer: timing operations
        """
        self._comm = comm
        self._tag = tag
        self._transformers = transformers
        self._timer = timer
        self._recv_requests: List[AsyncRequest] = []
        self._send_requests: List[AsyncRequest] = []
        self._inflight_x_quantities: Optional[Tuple[Quantity, ...]] = None
        self._inflight_y_quantities: Optional[Tuple[Quantity, ...]] = None
        self._finalize_on_wait = False

    def force_finalize_on_wait(self):
        """HaloDataTransformer are finalized after a wait call

        This is a temporary fix. See DSL-816 which will remove self._finalize_on_wait.
        """
        self._finalize_on_wait = True

    def __del__(self):
        """Clean up all buffers on garbage collection"""
        if (
            self._inflight_x_quantities is not None
            or self._inflight_y_quantities is not None
        ):
            raise RuntimeError(
                "An halo exchange wasn't completed and a wait() call was expected"
            )
        if not self._finalize_on_wait:
            for transformer in self._transformers.values():
                transformer.finalize()

    @classmethod
    def from_scalar_specifications(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        specifications: Iterable[QuantityHaloSpec],
        boundaries: Iterable[Boundary],
        tag: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        """
        Create/retrieve as many packed buffer as needed and
        queue the slices to exchange.

        Args:
            comm: communicator to post network messages
            numpy_like_module: module implementing numpy API
            specifications: data specifications to exchange, including
                number of halo points
            boundaries: informations on the exchange boundaries.
            tag: network tag (to differentiate messaging) for this node.
            optional_timer: timing of operations.

        Returns:
            HaloUpdater ready to exchange data.
        """

        timer = optional_timer if optional_timer is not None else NullTimer()

        # Sort the specification per target rank
        exchange_specs_dict = defaultdict(list)
        for boundary in boundaries:
            for specification in specifications:
                exchange_specs_dict[boundary.to_rank].append(
                    HaloExchangeSpec(
                        specification,
                        boundary.send_slice(specification),
                        boundary.n_clockwise_rotations,
                        boundary.recv_slice(specification),
                    ),
                )

        # Create the data transformers to support pack/unpack
        # One transformer per target rank
        transformers: Dict[int, HaloDataTransformer] = {}
        for rank, exchange_specs in exchange_specs_dict.items():
            transformers[rank] = HaloDataTransformer.get(
                numpy_like_module, exchange_specs
            )

        return cls(comm, tag, transformers, timer)

    @classmethod
    def from_vector_specifications(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        specifications_x: Iterable[QuantityHaloSpec],
        specifications_y: Iterable[QuantityHaloSpec],
        boundaries: Iterable[Boundary],
        tag: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        """
        Create/retrieve as many packed buffer as needed and queue
        the slices to exchange.

        Args:
            comm: communicator to post network messages
            numpy_like_module: module implementing numpy API
            specifications_x: specifications to exchange along the x axis.
                Length must match y specifications.
            specifications_y: specifications to exchange along the y axis.
                Length must match x specifications.
            boundaries: informations on the exchange boundaries.
            tag: network tag (to differentiate messaging) for this node.
            optional_timer: timing of operations.

        Returns:
            HaloUpdater ready to exchange data.
        """
        timer = optional_timer if optional_timer is not None else NullTimer()

        exchange_descriptors_x = defaultdict(list)
        exchange_descriptors_y = defaultdict(list)
        for boundary in boundaries:
            for specification_x, specification_y in zip(
                specifications_x, specifications_y
            ):
                exchange_descriptors_x[boundary.to_rank].append(
                    HaloExchangeSpec(
                        specification_x,
                        boundary.send_slice(specification_x),
                        boundary.n_clockwise_rotations,
                        boundary.recv_slice(specification_x),
                    )
                )
                exchange_descriptors_y[boundary.to_rank].append(
                    HaloExchangeSpec(
                        specification_y,
                        boundary.send_slice(specification_y),
                        boundary.n_clockwise_rotations,
                        boundary.recv_slice(specification_y),
                    )
                )

        transformers = {}
        for (rank_x, exchange_descriptor_x), (_rank_y, exchange_descriptor_y) in zip(
            exchange_descriptors_x.items(), exchange_descriptors_y.items()
        ):
            transformers[rank_x] = HaloDataTransformer.get(
                numpy_like_module,
                exchange_descriptor_x,
                exchange_descriptors_y=exchange_descriptor_y,
            )

        return cls(comm, tag, transformers, timer)

    def update(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Exhange the data and blocks until finished."""
        self.start(quantities_x, quantities_y)
        self.wait()

    def start(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Start data exchange."""
        self._comm._device_synchronize()

        if (
            self._inflight_x_quantities is not None
            or self._inflight_y_quantities is not None
        ):
            raise RuntimeError(
                "Previous exchange hasn't been properly finished."
                "E.g. previous start() call didn't have a wait() call."
            )

        # Post recv MPI order
        with self._timer.clock("Irecv"):
            self._recv_requests = []
            for to_rank, transformer in self._transformers.items():
                self._recv_requests.append(
                    self._comm.comm.Irecv(
                        transformer.get_unpack_buffer().array,
                        source=to_rank,
                        tag=self._tag,
                    )
                )

        # Pack quantities halo points data into buffers
        with self._timer.clock("pack"):
            for transformer in self._transformers.values():
                transformer.async_pack(quantities_x, quantities_y)

        self._inflight_x_quantities = tuple(quantities_x)
        self._inflight_y_quantities = (
            tuple(quantities_y) if quantities_y is not None else None
        )

        # Post send MPI order
        with self._timer.clock("Isend"):
            self._send_requests = []
            for to_rank, transformer in self._transformers.items():
                self._send_requests.append(
                    self._comm.comm.Isend(
                        transformer.get_pack_buffer().array,
                        dest=to_rank,
                        tag=self._tag,
                    )
                )

    def wait(self):
        """Finalize data exchange."""
        if __debug__ and self._inflight_x_quantities is None:
            raise RuntimeError('Halo update "wait" call before "start"')
        # Wait message to be exchange
        with self._timer.clock("wait"):
            for send_req in self._send_requests:
                send_req.wait()
            for recv_req in self._recv_requests:
                recv_req.wait()

        # Unpack buffers (updated by MPI with neighbouring halos)
        # to proper quantities
        with self._timer.clock("unpack"):
            for buffer in self._transformers.values():
                buffer.async_unpack(
                    self._inflight_x_quantities, self._inflight_y_quantities
                )
            if self._finalize_on_wait:
                for transformer in self._transformers.values():
                    transformer.finalize()
            else:
                for transformer in self._transformers.values():
                    transformer.synchronize()

        self._inflight_x_quantities = None
        self._inflight_y_quantities = None


class HaloUpdateRequest:
    """Asynchronous request object for halo updates."""

    def __init__(
        self,
        send_data: _HaloRequestSendList,
        recv_data: _HaloRequestRecvList,
        timer: Optional[Timer] = None,
    ):
        """Build a halo request.
        Args:
            send_data: a tuple of the MPI request and the buffer sent
            recv_data: a tuple of the MPI request, the temporary buffer and
                the destination buffer
            timer: optional, time the wait & unpack of a halo exchange
        """
        self._send_data = send_data
        self._recv_data = recv_data
        self._timer: Timer = timer if timer is not None else NullTimer()

    def wait(self):
        """Wait & unpack data into destination buffers
        Clean up by inserting back all buffers back in cache
        for potential reuse
        """
        for request, transfer_buffer in self._send_data:
            with self._timer.clock("wait"):
                request.wait()
            with self._timer.clock("unpack"):
                Buffer.push_to_cache(transfer_buffer)
        for request, transfer_buffer, destination_array in self._recv_data:
            with self._timer.clock("wait"):
                request.wait()
            with self._timer.clock("unpack"):
                transfer_buffer.assign_to(destination_array)
                Buffer.push_to_cache(transfer_buffer)


def on_c_grid(x_quantity, y_quantity):
    if (
        constants.X_DIM not in x_quantity.dims
        or constants.Y_INTERFACE_DIM not in x_quantity.dims
    ):
        return False
    if (
        constants.Y_DIM not in y_quantity.dims
        or constants.X_INTERFACE_DIM not in y_quantity.dims
    ):
        return False
    else:
        return True


class VectorInterfaceHaloUpdater:
    def __init__(
        self,
        comm,
        boundaries: Mapping[int, Boundary],
        force_cpu: bool = False,
        timer: Optional[Timer] = None,
    ):
        """Initialize a CubedSphereCommunicator.

        Args:
            comm: mpi4py.Comm object
            partitioner: cubed sphere partitioner
            force_cpu: Force all communication to go through central memory. Optional.
            timer: Time communication operations. Optional.
        """
        self.timer: Timer = timer if timer is not None else NullTimer()
        self._last_halo_tag = 0
        self._force_cpu = force_cpu
        self.comm = comm
        self.boundaries = boundaries

    def _get_halo_tag(self) -> int:
        self._last_halo_tag += 1
        return self._last_halo_tag

    def start_synchronize_vector_interfaces(
        self, x_quantity: Quantity, y_quantity: Quantity
    ) -> HaloUpdateRequest:
        """
        Synchronize shared points at the edges of a vector interface variable.

        Sends the values on the south and west edges to overwrite the values on adjacent
        subtiles. Vector must be defined on the Arakawa C grid.

        For interface variables, the edges of the tile are computed on both ranks
        bordering that edge. This routine copies values across those shared edges
        so that both ranks have the same value for that edge. It also handles any
        rotation of vector quantities needed to move data across the edge.

        Args:
            x_quantity: the x-component quantity to be synchronized
            y_quantity: the y-component quantity to be synchronized

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if not on_c_grid(x_quantity, y_quantity):
            raise ValueError("vector must be defined on Arakawa C-grid")
        device_synchronize()
        tag = self._get_halo_tag()
        send_requests = self._Isend_vector_shared_boundary(
            x_quantity, y_quantity, tag=tag
        )
        recv_requests = self._Irecv_vector_shared_boundary(
            x_quantity, y_quantity, tag=tag
        )
        return HaloUpdateRequest(send_requests, recv_requests, self.timer)

    def _Isend_vector_shared_boundary(
        self, x_quantity, y_quantity, tag=0
    ) -> _HaloRequestSendList:
        south_boundary = self.boundaries[constants.SOUTH]
        west_boundary = self.boundaries[constants.WEST]
        south_data = x_quantity.view.southwest.sel(
            **{
                constants.Y_INTERFACE_DIM: 0,
                constants.X_DIM: slice(
                    0, x_quantity.extent[x_quantity.dims.index(constants.X_DIM)]
                ),
            }
        )
        south_data = rotate_scalar_data(
            south_data,
            [constants.X_DIM],
            x_quantity.np,
            -south_boundary.n_clockwise_rotations,
        )
        if south_boundary.n_clockwise_rotations in (3, 2):
            south_data = -south_data
        west_data = y_quantity.view.southwest.sel(
            **{
                constants.X_INTERFACE_DIM: 0,
                constants.Y_DIM: slice(
                    0, y_quantity.extent[y_quantity.dims.index(constants.Y_DIM)]
                ),
            }
        )
        west_data = rotate_scalar_data(
            west_data,
            [constants.Y_DIM],
            y_quantity.np,
            -west_boundary.n_clockwise_rotations,
        )
        if west_boundary.n_clockwise_rotations in (1, 2):
            west_data = -west_data
        send_requests = [
            self._Isend(
                self._maybe_force_cpu(x_quantity.np),
                south_data,
                dest=south_boundary.to_rank,
                tag=tag,
            ),
            self._Isend(
                self._maybe_force_cpu(y_quantity.np),
                west_data,
                dest=west_boundary.to_rank,
                tag=tag,
            ),
        ]
        return send_requests

    def _maybe_force_cpu(self, module: NumpyModule) -> NumpyModule:
        """
        Get a numpy-like module depending on configuration and
        Quantity original allocator.
        """
        if self._force_cpu:
            return np
        return module

    def _Irecv_vector_shared_boundary(
        self, x_quantity, y_quantity, tag=0
    ) -> _HaloRequestRecvList:
        north_rank = self.boundaries[constants.NORTH].to_rank
        east_rank = self.boundaries[constants.EAST].to_rank
        north_data = x_quantity.view.northwest.sel(
            **{
                constants.Y_INTERFACE_DIM: -1,
                constants.X_DIM: slice(
                    0, x_quantity.extent[x_quantity.dims.index(constants.X_DIM)]
                ),
            }
        )
        east_data = y_quantity.view.southeast.sel(
            **{
                constants.X_INTERFACE_DIM: -1,
                constants.Y_DIM: slice(
                    0, y_quantity.extent[y_quantity.dims.index(constants.Y_DIM)]
                ),
            }
        )
        recv_requests = [
            self._Irecv(
                self._maybe_force_cpu(x_quantity.np),
                north_data,
                source=north_rank,
                tag=tag,
            ),
            self._Irecv(
                self._maybe_force_cpu(y_quantity.np),
                east_data,
                source=east_rank,
                tag=tag,
            ),
        ]
        return recv_requests

    def _Isend(self, numpy_module, in_array, **kwargs) -> _HaloSendTuple:
        # copy the resulting view in a contiguous array for transfer
        with self.timer.clock("pack"):
            buffer = Buffer.pop_from_cache(
                numpy_module.zeros, in_array.shape, in_array.dtype
            )
            buffer.assign_from(in_array)
            buffer.finalize_memory_transfer()
        with self.timer.clock("Isend"):
            request = self.comm.Isend(buffer.array, **kwargs)
        return (request, buffer)

    def _Irecv(self, numpy_module, out_array, **kwargs) -> _HaloRecvTuple:
        # Prepare a contiguous buffer to receive data
        with self.timer.clock("Irecv"):
            buffer = Buffer.pop_from_cache(
                numpy_module.zeros, out_array.shape, out_array.dtype
            )
            recv_request = self.comm.Irecv(buffer.array, **kwargs)
        return (recv_request, buffer, out_array)
