from typing import Iterable
from .quantity import Quantity, QuantityMetadata
from .partitioner import CubedSpherePartitioner, TilePartitioner
from . import constants
from .boundary import Boundary

__all__ = ['TileCommunicator', 'CubedSphereCommunicator']


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


class TileCommunicator(Communicator):
    """Performs communications within a single tile or region of a tile"""

    def __init__(self, comm, partitioner: TilePartitioner):
        self.partitioner = partitioner
        super(TileCommunicator, self).__init__(comm)

    def scatter(
            self,
            metadata: QuantityMetadata,
            send_quantity: Quantity = None,
            recv_quantity: Quantity = None) -> Quantity:
        """Transfer a quantity from the tile master rank to all subtiles.
        
        Args:
            metadata: the metadata of the quantity being transferred, used to
                initialize the default recieve buffer
            send_quantity: quantity to send, only used on the tile master rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        shape = self.partitioner.subtile_extent(metadata)
        if self.rank == constants.MASTER_RANK:
            sendbuf = metadata.np.empty(
                (self.partitioner.total_ranks,) + shape,
                dtype=metadata.dtype
            )
            for rank in range(0, self.partitioner.total_ranks):
                subtile_slice = self.partitioner.subtile_slice(
                    rank,
                    tile_metadata=metadata,
                    overlap=True,
                )
                sendbuf[rank, :] = send_quantity.view[subtile_slice]
        else:
            sendbuf = None
        if recv_quantity is None:
            recv_quantity = Quantity(
                metadata.np.empty(shape, dtype=metadata.dtype),
                dims=metadata.dims,
                units=metadata.units,
            )
        self.comm.Scatter(sendbuf, recv_quantity.data, root=0)
        return recv_quantity

    def scatter_state(self, tile_state: dict = None):
        """Transfer a state dictionary from the tile master rank to all subtiles.
        
        Args:
            tile_state: the model state to be sent containing the entire tile,
                required only from the master rank
        Returns:
            rank_state: the state corresponding to this rank's subdomain
        """
        def broadcast_master():
            if tile_state is None:
                raise TypeError('tile_state is a required argument on the master rank')
            name_list = list(tile_state.keys())
            while 'time' in name_list:
                name_list.remove('time')
            name_list = self.comm.bcast(name_list, root=constants.MASTER_RANK)
            array_list = [tile_state[name] for name in name_list]
            metadata_list = bcast_metadata_list(self.comm, array_list)
            for name, array, metadata in zip(name_list, array_list, metadata_list):
                state[name] = self.scatter(metadata, send_quantity=array)
            state['time'] = self.comm.bcast(tile_state.get('time', None), root=constants.MASTER_RANK)

        def broadcast_client():
            name_list = self.comm.bcast(None, root=constants.MASTER_RANK)
            metadata_list = bcast_metadata_list(self.comm, None)
            for name, metadata in zip(name_list, metadata_list):
                state[name] = self.scatter(metadata)
            time = self.comm.bcast(None, root=constants.MASTER_RANK)
            if time is not None:
                state['time'] = time

        state = {}
        if self.rank == constants.MASTER_RANK:
            broadcast_master()
        else:
            broadcast_client()
        return state


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
        tile_comm = self.comm.Split(color=self.partitioner.tile_index(self.rank), key=self.rank)
        self._tile_communicator = TileCommunicator(tile_comm, self.partitioner.tile)

    def start_halo_update(self, quantity: Quantity, n_points: int):
        """Initiate an asynchronous halo update of a quantity."""
        if n_points == 0:
            raise ValueError('cannot perform a halo update on zero halo points')
        for boundary_type, boundary in self.boundaries.items():
            data = boundary.send_view(quantity, n_points=n_points)
            data = quantity.np.ascontiguousarray(
                rotate_data(data, quantity.metadata, boundary.n_clockwise_rotations)
            )
            self.comm.Isend(data, dest=boundary.to_rank)

    def finish_halo_update(self, quantity: Quantity, n_points: int):
        """Complete an asynchronous halo update of a quantity."""
        for boundary_type, boundary in self.boundaries.items():
            dest_view = boundary.recv_view(quantity, n_points=n_points)
            dest_buffer = quantity.np.empty(dest_view.shape, dtype=dest_view.dtype)
            self.comm.Recv(dest_buffer, source=boundary.to_rank)
            dest_view[:] = dest_buffer


def rotate_data(data, metadata, n_clockwise_rotations):
    n_clockwise_rotations = n_clockwise_rotations % 4
    if n_clockwise_rotations == 0:
        pass
    elif n_clockwise_rotations in (1, 3):
        x_dim, y_dim = None, None
        for i, dim in enumerate(metadata.dims):
            if dim in constants.X_DIMS:
                x_dim = i
            elif dim in constants.Y_DIMS:
                y_dim = i
        if n_clockwise_rotations == 1:
            data = metadata.np.rot90(data, axes=(x_dim, y_dim))
        elif n_clockwise_rotations == 3:
            data = metadata.np.rot90(data, axes=(y_dim, x_dim))
    elif n_clockwise_rotations == 2:
        slice_list = []
        for dim in metadata.dims:
            if dim in constants.HORIZONTAL_DIMS:
                slice_list.append(slice(None, None, -1))
            else:
                slice_list.append(slice(None, None))
        data = data[slice_list]
    return data

