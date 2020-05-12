from typing import Iterable
from .quantity import Quantity
from .partitioner import CubedSpherePartitioner, TilePartitioner
from . import constants
from .boundary import Boundary
from .rotate import rotate_scalar_data, rotate_vector_data
from .buffer import array_buffer, send_buffer, recv_buffer
import logging

__all__ = [
    "TileCommunicator",
    "CubedSphereCommunicator",
    "Communicator",
    "HaloUpdateRequest",
]

logger = logging.getLogger("fv3util")


def bcast_metadata_list(comm, quantity_list):
    is_master = comm.Get_rank() == constants.MASTER_RANK
    if is_master:
        metadata_list = []
        for quantity in quantity_list:
            metadata_list.append(quantity.metadata)
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.MASTER_RANK)


def bcast_metadata(comm, array):
    return bcast_metadata_list(comm, [array])[0]


class Communicator:
    def __init__(self, comm):
        self.comm = comm

    @property
    def rank(self) -> int:
        """rank of the current process within this communicator"""
        return self.comm.Get_rank()


class FunctionRequest:
    def __init__(self, function):
        self._function = function

    def wait(self):
        self._function()


class HaloUpdateRequest:
    """asynchronous request object for halo updates"""

    def __init__(self, send_requests, recv_requests):
        self._send_requests = send_requests
        self._recv_requests = recv_requests

    def wait(self):
        for request in self._recv_requests:
            request.wait()
        for request in self._send_requests:
            request.wait()


class TileCommunicator(Communicator):
    """Performs communications within a single tile or region of a tile"""

    def __init__(self, comm, partitioner: TilePartitioner):
        self.partitioner = partitioner
        super(TileCommunicator, self).__init__(comm)

    def _Scatter(self, numpy, sendbuf, recvbuf, **kwargs):
        with send_buffer(numpy, sendbuf) as send, recv_buffer(numpy, recvbuf) as recv:
            self.comm.Scatter(send, recv, **kwargs)

    def _Gather(self, numpy, sendbuf, recvbuf, **kwargs):
        with send_buffer(numpy, sendbuf) as send, recv_buffer(numpy, recvbuf) as recv:
            self.comm.Gather(send, recv, **kwargs)

    def scatter(
        self, send_quantity: Quantity = None, recv_quantity: Quantity = None
    ) -> Quantity:
        """Transfer subtile regions of a full-tile quantity
        from the tile master rank to all subtiles.
        
        Args:
            send_quantity: quantity to send, only required/used on the tile master rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        if self.rank == constants.MASTER_RANK and send_quantity is None:
            raise TypeError("send_quantity is a required argument on the master rank")
        if self.rank == constants.MASTER_RANK:
            metadata = self.comm.bcast(
                send_quantity.metadata, root=constants.MASTER_RANK
            )
        else:
            metadata = self.comm.bcast(None, root=constants.MASTER_RANK)
        shape = self.partitioner.subtile_extent(metadata)
        if recv_quantity is None:
            recv_quantity = Quantity(
                metadata.np.empty(shape, dtype=metadata.dtype),
                dims=metadata.dims,
                units=metadata.units,
            )
        if self.rank == constants.MASTER_RANK:
            with array_buffer(
                metadata.np.empty,
                (self.partitioner.total_ranks,) + shape,
                dtype=metadata.dtype,
            ) as sendbuf:
                for rank in range(0, self.partitioner.total_ranks):
                    subtile_slice = self.partitioner.subtile_slice(
                        rank,
                        tile_dims=metadata.dims,
                        tile_extent=metadata.extent,
                        overlap=True,
                    )
                    sendbuf[rank, :] = send_quantity.view[subtile_slice]
                self._Scatter(
                    metadata.np,
                    sendbuf,
                    recv_quantity.view[:],
                    root=constants.MASTER_RANK,
                )
        else:
            self._Scatter(
                metadata.np, None, recv_quantity.view[:], root=constants.MASTER_RANK
            )
        return recv_quantity

    def gather(
        self, send_quantity: Quantity, recv_quantity: Quantity = None
    ) -> Quantity:
        """Transfer subtile regions of a full-tile quantity
        from each rank to the tile master rank.
        
        Args:
            send_quantity: quantity to send
            recv_quantity: if provided, assign received data into this Quantity (only
                used on the tile master rank)
        Returns:
            recv_quantity
        """
        if self.rank == constants.MASTER_RANK:
            with array_buffer(
                send_quantity.np.empty,
                (self.partitioner.total_ranks,) + tuple(send_quantity.extent),
                dtype=send_quantity.data.dtype,
            ) as recvbuf:
                self._Gather(
                    send_quantity.np,
                    send_quantity.view[:],
                    recvbuf,
                    root=constants.MASTER_RANK,
                )
                if recv_quantity is None:
                    tile_extent = self.partitioner.tile_extent(send_quantity.metadata)
                    recv_quantity = Quantity(
                        send_quantity.np.empty(
                            tile_extent, dtype=send_quantity.data.dtype
                        ),
                        dims=send_quantity.dims,
                        units=send_quantity.units,
                        origin=tuple([0 for dim in send_quantity.dims]),
                        extent=tile_extent,
                    )
                for rank in range(self.partitioner.total_ranks):
                    to_slice = self.partitioner.subtile_slice(
                        rank,
                        tile_dims=recv_quantity.dims,
                        tile_extent=recv_quantity.extent,
                        overlap=True,
                    )
                    recv_quantity.view[to_slice] = recvbuf[rank, :]
                result = recv_quantity
        else:
            result = self._Gather(
                send_quantity.np,
                send_quantity.view[:],
                None,
                root=constants.MASTER_RANK,
            )
        return result

    def gather_state(self, send_state: dict = None, recv_state: dict = None):
        """Transfer a state dictionary from subtile ranks to the tile master rank.

        'time' is assumed to be the same on all ranks, and its value will be set
        to the value from the master rank.

        Args:
            send_state: the model state to be sent containing the subtile data
            recv_state: the pre-allocated state in which to recieve the full tile
                state. Only variables which are scattered will be written to.
        Returns:
            recv_state: on the master rank, the state containing the entire tile
        """
        if self.rank == constants.MASTER_RANK and recv_state is None:
            recv_state = {}
        for name, quantity in send_state.items():
            if name == "time":
                if self.rank == constants.MASTER_RANK:
                    recv_state["time"] = send_state["time"]
            else:
                if recv_state is not None and name in recv_state:
                    tile_quantity = self.gather(
                        quantity, recv_quantity=recv_state[name]
                    )
                else:
                    tile_quantity = self.gather(quantity)
                if self.rank == constants.MASTER_RANK:
                    recv_state[name] = tile_quantity
        return recv_state

    def scatter_state(self, send_state: dict = None, recv_state: dict = None):
        """Transfer a state dictionary from the tile master rank to all subtiles.
        
        Args:
            send_state: the model state to be sent containing the entire tile,
                required only from the master rank
            recv_state: the pre-allocated state in which to recieve the scattered
                state. Only variables which are scattered will be written to.
        Returns:
            rank_state: the state corresponding to this rank's subdomain
        """

        def scatter_master():
            if send_state is None:
                raise TypeError("send_state is a required argument on the master rank")
            name_list = list(send_state.keys())
            while "time" in name_list:
                name_list.remove("time")
            name_list = self.comm.bcast(name_list, root=constants.MASTER_RANK)
            array_list = [send_state[name] for name in name_list]
            for name, array in zip(name_list, array_list):
                if name in recv_state:
                    self.scatter(send_quantity=array, recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter(send_quantity=array)
            recv_state["time"] = self.comm.bcast(
                send_state.get("time", None), root=constants.MASTER_RANK
            )

        def scatter_client():
            name_list = self.comm.bcast(None, root=constants.MASTER_RANK)
            for name in name_list:
                if name in recv_state:
                    self.scatter(recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter()
            time = self.comm.bcast(None, root=constants.MASTER_RANK)
            if time is not None:
                recv_state["time"] = time

        if recv_state is None:
            recv_state = {}
        if self.rank == constants.MASTER_RANK:
            scatter_master()
        else:
            scatter_client()
        return recv_state


class CubedSphereCommunicator(Communicator):
    """Performs communications within a cubed sphere"""

    def __init__(self, comm, partitioner: CubedSpherePartitioner):
        """Initialize a CubedSphereCommunicator.
        
        Args:
            comm: mpi4py.Comm object
            partitioner: cubed sphere partitioner
        """
        self.partitioner = partitioner
        self._tile_communicator = None
        self._boundaries = None
        super(CubedSphereCommunicator, self).__init__(comm)

    @property
    def boundaries(self) -> Iterable[Boundary]:
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
        return self._tile_communicator

    def _initialize_tile_communicator(self):
        tile_comm = self.comm.Split(
            color=self.partitioner.tile_index(self.rank), key=self.rank
        )
        self._tile_communicator = TileCommunicator(tile_comm, self.partitioner.tile)

    def halo_update(self, quantity: Quantity, n_points: int):
        """Perform a halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior
        """
        req = self.start_halo_update(quantity, n_points)
        req.wait()

    def start_halo_update(self, quantity: Quantity, n_points: int) -> HaloUpdateRequest:
        """Start an asynchronous halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if n_points == 0:
            raise ValueError("cannot perform a halo update on zero halo points")
        send_requests = self._Isend_halos(quantity, n_points)
        recv_requests = self._Irecv_halos(quantity, n_points)
        return HaloUpdateRequest(send_requests, recv_requests)

    def _Isend_halos(self, quantity: Quantity, n_points: int):
        send_requests = []
        for boundary_type, boundary in self.boundaries.items():
            data = boundary.send_view(quantity, n_points=n_points)
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation.\
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            data = rotate_scalar_data(
                data, quantity.dims, quantity.np, -boundary.n_clockwise_rotations
            )
            send_requests.append(self._Isend(quantity.np, data, dest=boundary.to_rank))
        return send_requests

    def _Irecv_halos(self, quantity: Quantity, n_points: int):
        recv_requests = []
        for boundary_type, boundary in self.boundaries.items():
            dest_view = boundary.recv_view(quantity, n_points=n_points)
            logger.debug(
                "finish_halo_update: retrieving boundary_type=%s shape=%s from_rank=%s to_rank=%s",
                boundary_type,
                dest_view.shape,
                boundary.to_rank,
                self.rank,
            )
            recv_requests.append(
                self._Irecv(quantity.np, dest_view, source=boundary.to_rank)
            )
        return recv_requests

    def finish_halo_update(self, quantity: Quantity, n_points: int):
        """Deprecated, do not use."""
        raise NotImplementedError(
            "finish_halo_update has been removed, use .wait() on the request object "
            "returned by start_halo_update"
        )

    def vector_halo_update(
        self, x_quantity: Quantity, y_quantity: Quantity, n_points: int,
    ):
        """Perform a halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior
        """
        req = self.start_vector_halo_update(x_quantity, y_quantity, n_points)
        req.wait()

    def start_vector_halo_update(
        self, x_quantity: Quantity, y_quantity: Quantity, n_points: int,
    ) -> HaloUpdateRequest:
        """Start an asynchronous halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if n_points == 0:
            raise ValueError("cannot perform a halo update on zero halo points")
        send_requests = self._Isend_vector_halos(x_quantity, y_quantity, n_points)
        recv_requests = self._Irecv_halos(x_quantity, n_points)
        recv_requests.extend(self._Irecv_halos(y_quantity, n_points))
        return HaloUpdateRequest(send_requests, recv_requests)

    def _Isend_vector_halos(self, x_quantity, y_quantity, n_points):
        send_requests = []
        for boundary_type, boundary in self.boundaries.items():
            x_data = boundary.send_view(x_quantity, n_points=n_points)
            y_data = boundary.send_view(y_quantity, n_points=n_points)
            logger.debug("%s %s", x_data.shape, y_data.shape)
            x_data, y_data = rotate_vector_data(
                x_data,
                y_data,
                -boundary.n_clockwise_rotations,
                x_quantity.dims,
                x_quantity.np,
            )
            logger.debug(
                "%s %s %s %s %s",
                boundary.from_rank,
                boundary.to_rank,
                boundary.n_clockwise_rotations,
                x_data.shape,
                y_data.shape,
            )
            send_requests.append(
                self._Isend(x_quantity.np, x_data, dest=boundary.to_rank)
            )
            send_requests.append(
                self._Isend(y_quantity.np, y_data, dest=boundary.to_rank)
            )
        return send_requests

    def _Isend(self, numpy, in_array, **kwargs):
        # don't want to use a buffer here, because we leave this scope and can't close
        # the context manager. might figure out a way to do it later
        array = numpy.ascontiguousarray(in_array)
        return self.comm.Isend(array, **kwargs)

    def _Send(self, numpy, in_array, **kwargs):
        with send_buffer(numpy, in_array) as sendbuf:
            self.comm.Send(sendbuf, **kwargs)

    def _Recv(self, numpy, out_array, **kwargs):
        with recv_buffer(numpy, out_array) as recvbuf:
            self.comm.Recv(recvbuf, **kwargs)

    def _Irecv(self, numpy, out_array, **kwargs):
        # we can't perform a true Irecv because we need to receive the data into a
        # buffer and then copy that buffer into the output array. Instead we will
        # just do a Recv() when wait is called.
        def recv():
            with recv_buffer(numpy, out_array) as recvbuf:
                self.comm.Recv(recvbuf, **kwargs)

        return FunctionRequest(recv)

    def finish_vector_halo_update(
        self, x_quantity: Quantity, y_quantity: Quantity, n_points: int,
    ):
        """Deprecated, do not use."""
        raise NotImplementedError(
            "finish_vector_halo_update has been removed, use .wait() on the request object "
            "returned by start_vector_halo_update"
        )
