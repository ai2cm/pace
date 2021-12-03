from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from ._timing import NullTimer, Timer
from .boundary import Boundary
from .halo_data_transformer import (
    HaloDataTransformer,
    HaloExchangeSpec,
    QuantityHaloSpec,
)
from .quantity import Quantity
from .types import AsyncRequest, NumpyModule


if TYPE_CHECKING:
    from .communicator import Communicator


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
