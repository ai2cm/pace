import logging
from typing import List, Mapping, Optional, Sequence, Tuple, Union, cast

import numpy as np

from . import constants
from ._timing import NullTimer, Timer
from .boundary import Boundary
from .buffer import array_buffer, recv_buffer, send_buffer
from .halo_data_transformer import QuantityHaloSpec
from .halo_updater import HaloUpdater, HaloUpdateRequest, VectorInterfaceHaloUpdater
from .partitioner import CubedSpherePartitioner, Partitioner, TilePartitioner
from .quantity import Quantity, QuantityMetadata
from .types import NumpyModule
from .utils import device_synchronize


logger = logging.getLogger("pace.util")


def bcast_metadata_list(comm, quantity_list):
    is_root = comm.Get_rank() == constants.ROOT_RANK
    if is_root:
        metadata_list = []
        for quantity in quantity_list:
            metadata_list.append(quantity.metadata)
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.ROOT_RANK)


def bcast_metadata(comm, array):
    return bcast_metadata_list(comm, [array])[0]


class Communicator:
    def __init__(self, comm, partitioner, force_cpu: bool = False):
        self.comm = comm
        self.partitioner: Partitioner = partitioner
        self._force_cpu = force_cpu

    @property
    def rank(self) -> int:
        """rank of the current process within this communicator"""
        return self.comm.Get_rank()

    def _maybe_force_cpu(self, module: NumpyModule) -> NumpyModule:
        """
        Get a numpy-like module depending on configuration and
        Quantity original allocator.
        """
        if self._force_cpu:
            return np
        return module

    @staticmethod
    def _device_synchronize():
        """Wait for all work that could be in-flight to finish."""
        # this is a method so we can profile it separately from other device syncs
        device_synchronize()

    def _Scatter(self, numpy_module, sendbuf, recvbuf, **kwargs):
        with send_buffer(numpy_module.zeros, sendbuf) as send, recv_buffer(
            numpy_module.zeros, recvbuf
        ) as recv:
            self.comm.Scatter(send, recv, **kwargs)

    def _Gather(self, numpy_module, sendbuf, recvbuf, **kwargs):
        with send_buffer(numpy_module.zeros, sendbuf) as send, recv_buffer(
            numpy_module.zeros, recvbuf
        ) as recv:
            self.comm.Gather(send, recv, **kwargs)

    def scatter(
        self,
        send_quantity: Optional[Quantity] = None,
        recv_quantity: Optional[Quantity] = None,
    ) -> Quantity:
        """Transfer subtile regions of a full-tile quantity
        from the tile root rank to all subtiles.

        Args:
            send_quantity: quantity to send, only required/used on the tile root rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        if self.rank == constants.ROOT_RANK and send_quantity is None:
            raise TypeError("send_quantity is a required argument on the root rank")
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            metadata = self.comm.bcast(send_quantity.metadata, root=constants.ROOT_RANK)
        else:
            metadata = self.comm.bcast(None, root=constants.ROOT_RANK)
        shape = self.partitioner.subtile_extent(metadata, self.rank)
        if recv_quantity is None:
            recv_quantity = self._get_scatter_recv_quantity(shape, metadata)
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            with array_buffer(
                self._maybe_force_cpu(metadata.np).zeros,
                (self.partitioner.total_ranks,) + shape,
                dtype=metadata.dtype,
            ) as sendbuf:
                for rank in range(0, self.partitioner.total_ranks):
                    subtile_slice = self.partitioner.subtile_slice(
                        rank=rank,
                        global_dims=metadata.dims,
                        global_extent=metadata.extent,
                        overlap=True,
                    )
                    sendbuf.assign_from(
                        send_quantity.view[subtile_slice],
                        buffer_slice=np.index_exp[rank, :],
                    )
                self._Scatter(
                    metadata.np,
                    sendbuf.array,
                    recv_quantity.view[:],
                    root=constants.ROOT_RANK,
                )
        else:
            self._Scatter(
                metadata.np,
                None,
                recv_quantity.view[:],
                root=constants.ROOT_RANK,
            )
        return recv_quantity

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather"""
        recv_quantity = Quantity(
            send_metadata.np.empty(global_extent, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            origin=tuple([0 for dim in send_metadata.dims]),
            extent=global_extent,
            gt4py_backend=send_metadata.gt4py_backend,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter"""
        recv_quantity = Quantity(
            send_metadata.np.empty(shape, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            gt4py_backend=send_metadata.gt4py_backend,
        )
        return recv_quantity

    def gather(
        self, send_quantity: Quantity, recv_quantity: Quantity = None
    ) -> Optional[Quantity]:
        """Transfer subtile regions of a full-tile quantity
        from each rank to the tile root rank.

        Args:
            send_quantity: quantity to send
            recv_quantity: if provided, assign received data into this Quantity (only
                used on the tile root rank)
        Returns:
            recv_quantity: quantity if on root rank, otherwise None
        """
        result: Optional[Quantity]
        if self.rank == constants.ROOT_RANK:
            with array_buffer(
                send_quantity.np.empty,
                (self.partitioner.total_ranks,) + tuple(send_quantity.extent),
                dtype=send_quantity.data.dtype,
            ) as recvbuf:
                self._Gather(
                    send_quantity.np,
                    send_quantity.view[:],
                    recvbuf.array,
                    root=constants.ROOT_RANK,
                )
                if recv_quantity is None:
                    global_extent = self.partitioner.global_extent(
                        send_quantity.metadata
                    )
                    recv_quantity = self._get_gather_recv_quantity(
                        global_extent, send_quantity.metadata
                    )
                for rank in range(self.partitioner.total_ranks):
                    to_slice = self.partitioner.subtile_slice(
                        rank=rank,
                        global_dims=recv_quantity.dims,
                        global_extent=recv_quantity.extent,
                        overlap=True,
                    )
                    recvbuf.assign_to(
                        recv_quantity.view[to_slice], buffer_slice=np.index_exp[rank, :]
                    )
                result = recv_quantity
        else:
            self._Gather(
                send_quantity.np,
                send_quantity.view[:],
                None,
                root=constants.ROOT_RANK,
            )
            result = None
        return result

    def gather_state(self, send_state=None, recv_state=None):
        """Transfer a state dictionary from subtile ranks to the tile root rank.

        'time' is assumed to be the same on all ranks, and its value will be set
        to the value from the root rank.

        Args:
            send_state: the model state to be sent containing the subtile data
            recv_state: the pre-allocated state in which to recieve the full tile
                state. Only variables which are scattered will be written to.
        Returns:
            recv_state: on the root rank, the state containing the entire tile
        """
        if self.rank == constants.ROOT_RANK and recv_state is None:
            recv_state = {}
        for name, quantity in send_state.items():
            if name == "time":
                if self.rank == constants.ROOT_RANK:
                    recv_state["time"] = send_state["time"]
            else:
                if recv_state is not None and name in recv_state:
                    tile_quantity = self.gather(
                        quantity, recv_quantity=recv_state[name]
                    )
                else:
                    tile_quantity = self.gather(quantity)
                if self.rank == constants.ROOT_RANK:
                    recv_state[name] = tile_quantity
        return recv_state

    def scatter_state(self, send_state=None, recv_state=None):
        """Transfer a state dictionary from the tile root rank to all subtiles.

        Args:
            send_state: the model state to be sent containing the entire tile,
                required only from the root rank
            recv_state: the pre-allocated state in which to recieve the scattered
                state. Only variables which are scattered will be written to.
        Returns:
            rank_state: the state corresponding to this rank's subdomain
        """

        def scatter_root():
            if send_state is None:
                raise TypeError("send_state is a required argument on the root rank")
            name_list = list(send_state.keys())
            while "time" in name_list:
                name_list.remove("time")
            name_list = self.comm.bcast(name_list, root=constants.ROOT_RANK)
            array_list = [send_state[name] for name in name_list]
            for name, array in zip(name_list, array_list):
                if name in recv_state:
                    self.scatter(send_quantity=array, recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter(send_quantity=array)
            recv_state["time"] = self.comm.bcast(
                send_state.get("time", None), root=constants.ROOT_RANK
            )

        def scatter_client():
            name_list = self.comm.bcast(None, root=constants.ROOT_RANK)
            for name in name_list:
                if name in recv_state:
                    self.scatter(recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter()
            recv_state["time"] = self.comm.bcast(None, root=constants.ROOT_RANK)

        if recv_state is None:
            recv_state = {}
        if self.rank == constants.ROOT_RANK:
            scatter_root()
        else:
            scatter_client()
        if recv_state["time"] is None:
            recv_state.pop("time")
        return recv_state


class TileCommunicator(Communicator):
    """Performs communications within a single tile or region of a tile"""

    def __init__(self, comm, partitioner: TilePartitioner, force_cpu: bool = False):
        """Initialize a TileCommunicator.

        Args:
            comm: communication object behaving like mpi4py.Comm
            partitioner: tile partitioner
            force_cpu: force all communication to go through central memory
        """
        super(TileCommunicator, self).__init__(comm, partitioner, force_cpu=force_cpu)
        self.partitioner: TilePartitioner = partitioner


class CubedSphereCommunicator(Communicator):
    """Performs communications within a cubed sphere"""

    timer: Timer
    partitioner: CubedSpherePartitioner

    def __init__(
        self,
        comm,
        partitioner: CubedSpherePartitioner,
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
        if comm.Get_size() != partitioner.total_ranks:
            raise ValueError(
                f"was given a partitioner for {partitioner.total_ranks} ranks but a "
                f"comm object with only {comm.Get_size()} ranks, are we running "
                "with mpi and the correct number of ranks?"
            )
        self.timer: Timer = timer if timer is not None else NullTimer()
        self._tile_communicator: Optional[TileCommunicator] = None
        self._boundaries: Optional[Mapping[int, Boundary]] = None
        self._last_halo_tag = 0
        self._force_cpu = force_cpu
        super(CubedSphereCommunicator, self).__init__(comm, partitioner, force_cpu)
        self.partitioner: CubedSpherePartitioner = partitioner

    def _get_halo_tag(self) -> int:
        self._last_halo_tag += 1
        return self._last_halo_tag

    @classmethod
    def from_layout(
        cls,
        comm,
        layout: Tuple[int, int],
        force_cpu: bool = False,
        timer: Optional[Timer] = None,
    ) -> "CubedSphereCommunicator":
        partitioner = CubedSpherePartitioner(tile=TilePartitioner(layout=layout))
        return cls(comm=comm, partitioner=partitioner, force_cpu=force_cpu, timer=timer)

    @property
    def boundaries(self) -> Mapping[int, Boundary]:
        """boundaries of this tile with neighboring tiles"""
        if self._boundaries is None:
            self._boundaries = {}
            for boundary_type in constants.BOUNDARY_TYPES:
                boundary = self.partitioner.boundary(boundary_type, self.rank)
                if boundary is not None:
                    self._boundaries[boundary_type] = boundary
        return self._boundaries

    @property
    def tile(self) -> TileCommunicator:
        """communicator for within a tile"""
        if self._tile_communicator is None:
            self._initialize_tile_communicator()
        return cast(TileCommunicator, self._tile_communicator)

    def _initialize_tile_communicator(self):
        tile_comm = self.comm.Split(
            color=self.partitioner.tile_index(self.rank), key=self.rank
        )
        self._tile_communicator = TileCommunicator(tile_comm, self.partitioner.tile)

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather

        Args:
            shape: ndarray shape, numpy-style
            metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we add a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            metadata.np.empty(global_extent, dtype=metadata.dtype),
            dims=(constants.TILE_DIM,) + metadata.dims,
            units=metadata.units,
            origin=(0,) + tuple([0 for dim in metadata.dims]),
            extent=global_extent,
            gt4py_backend=metadata.gt4py_backend,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter

        Args:
            shape: ndarray shape, numpy-style
            metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we remove a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            metadata.np.empty(shape, dtype=metadata.dtype),
            dims=metadata.dims[1:],
            units=metadata.units,
            gt4py_backend=metadata.gt4py_backend,
        )
        return recv_quantity

    def halo_update(self, quantity: Union[Quantity, List[Quantity]], n_points: int):
        """Perform a halo update on a quantity or quantities

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior
        """
        if isinstance(quantity, Quantity):
            quantities = [quantity]
        else:
            quantities = quantity

        halo_updater = self.start_halo_update(quantities, n_points)
        halo_updater.wait()

    def start_halo_update(
        self, quantity: Union[Quantity, List[Quantity]], n_points: int
    ) -> HaloUpdater:
        """Start an asynchronous halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if isinstance(quantity, Quantity):
            quantities = [quantity]
        else:
            quantities = quantity

        specifications = []
        for quantity in quantities:
            specification = QuantityHaloSpec(
                n_points=n_points,
                shape=quantity.data.shape,
                strides=quantity.data.strides,
                itemsize=quantity.data.itemsize,
                origin=quantity.origin,
                extent=quantity.extent,
                dims=quantity.dims,
                numpy_module=self._maybe_force_cpu(quantity.np),
                dtype=quantity.metadata.dtype,
            )
            specifications.append(specification)

        halo_updater = self.get_scalar_halo_updater(specifications)
        halo_updater.force_finalize_on_wait()
        halo_updater.start(quantities)
        return halo_updater

    def finish_halo_update(self, quantity: Quantity, n_points: int):
        """Deprecated, do not use."""
        raise NotImplementedError(
            "finish_halo_update has been removed, use .wait() on the request object "
            "returned by start_halo_update"
        )

    def vector_halo_update(
        self,
        x_quantity: Union[Quantity, List[Quantity]],
        y_quantity: Union[Quantity, List[Quantity]],
        n_points: int,
    ):
        """Perform a halo update of a horizontal vector quantity or quantities.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior
        """
        if isinstance(x_quantity, Quantity):
            x_quantities = [x_quantity]
        else:
            x_quantities = x_quantity
        if isinstance(y_quantity, Quantity):
            y_quantities = [y_quantity]
        else:
            y_quantities = y_quantity

        halo_updater = self.start_vector_halo_update(
            x_quantities, y_quantities, n_points
        )
        halo_updater.wait()

    def start_vector_halo_update(
        self,
        x_quantity: Union[Quantity, List[Quantity]],
        y_quantity: Union[Quantity, List[Quantity]],
        n_points: int,
    ) -> HaloUpdater:
        """Start an asynchronous halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if isinstance(x_quantity, Quantity):
            x_quantities = [x_quantity]
        else:
            x_quantities = x_quantity
        if isinstance(y_quantity, Quantity):
            y_quantities = [y_quantity]
        else:
            y_quantities = y_quantity

        x_specifications = []
        y_specifications = []
        for x_quantity, y_quantity in zip(x_quantities, y_quantities):
            x_specification = QuantityHaloSpec(
                n_points=n_points,
                shape=x_quantity.data.shape,
                strides=x_quantity.data.strides,
                itemsize=x_quantity.data.itemsize,
                origin=x_quantity.metadata.origin,
                extent=x_quantity.metadata.extent,
                dims=x_quantity.metadata.dims,
                numpy_module=self._maybe_force_cpu(x_quantity.np),
                dtype=x_quantity.metadata.dtype,
            )
            x_specifications.append(x_specification)
            y_specification = QuantityHaloSpec(
                n_points=n_points,
                shape=y_quantity.data.shape,
                strides=y_quantity.data.strides,
                itemsize=y_quantity.data.itemsize,
                origin=y_quantity.metadata.origin,
                extent=y_quantity.metadata.extent,
                dims=y_quantity.metadata.dims,
                numpy_module=self._maybe_force_cpu(y_quantity.np),
                dtype=y_quantity.metadata.dtype,
            )
            y_specifications.append(y_specification)

        halo_updater = self.get_vector_halo_updater(x_specifications, y_specifications)
        halo_updater.force_finalize_on_wait()
        halo_updater.start(x_quantities, y_quantities)
        return halo_updater

    def synchronize_vector_interfaces(self, x_quantity: Quantity, y_quantity: Quantity):
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
        """
        req = self.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req.wait()

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
        halo_updater = VectorInterfaceHaloUpdater(
            comm=self.comm,
            boundaries=self.boundaries,
            force_cpu=self._force_cpu,
            timer=self.timer,
        )
        req = halo_updater.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        return req

    def get_scalar_halo_updater(self, specifications: List[QuantityHaloSpec]):
        if len(specifications) == 0:
            raise RuntimeError("Cannot create updater with specifications list")
        if specifications[0].n_points == 0:
            raise ValueError("cannot perform a halo update on zero halo points")
        return HaloUpdater.from_scalar_specifications(
            self,
            self._maybe_force_cpu(specifications[0].numpy_module),
            specifications,
            self.boundaries.values(),
            self._get_halo_tag(),
            self.timer,
        )

    def get_vector_halo_updater(
        self,
        specifications_x: List[QuantityHaloSpec],
        specifications_y: List[QuantityHaloSpec],
    ):
        if len(specifications_x) == 0 and len(specifications_y) == 0:
            raise RuntimeError("Cannot create updater with empty specifications list")
        if specifications_x[0].n_points == 0 and specifications_y[0].n_points == 0:
            raise ValueError("Cannot perform a halo update on zero halo points")
        return HaloUpdater.from_vector_specifications(
            self,
            self._maybe_force_cpu(specifications_x[0].numpy_module),
            specifications_x,
            specifications_y,
            self.boundaries.values(),
            self._get_halo_tag(),
            self.timer,
        )
