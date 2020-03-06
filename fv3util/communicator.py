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
            metadata_list.append(QuantityMetadata.from_quantity(quantity))
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.MASTER_RANK)


def bcast_metadata(comm, array):
    return bcast_metadata_list(comm, [array])[0]


class TileCommunicator:
    """Performs communications within a single tile or region of a tile"""

    def __init__(self, tile_comm, partitioner: TilePartitioner):
        self.partitioner = partitioner
        self.tile_comm = tile_comm

    def scatter(
            self,
            metadata: QuantityMetadata,
            send_quantity: Quantity = None,
            recv_quantity: Quantity = None) -> Quantity:
        """Transfer data from the tile master rank to all subtiles.
        
        Args:
            metadata: the metadata of the quantity being transferred, used to
                initialize the default recieve buffer
            send_quantity: quantity to send, only used on the tile master rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        shape = self.partitioner.subtile_extent(metadata)
        if self.tile_comm.Get_rank() == constants.MASTER_RANK:
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
        self.tile_comm.Scatter(sendbuf, recv_quantity.data, root=0)
        return recv_quantity

    @property
    def rank(self) -> int:
        """rank of the current process within the tile"""
        return self.tile_comm.Get_rank()
    

class CubedSphereCommunicator:
    """Performs communications within a cubed sphere"""

    def __init__(self, comm, partitioner: CubedSpherePartitioner):
        """Initialize a CubedSphereCommunicator.
        
        Args:
            comm: mpi4py.Comm object
            partitioner: cubed sphere partitioner
        """
        self.comm = comm
        self.partitioner = partitioner
        self._tile_communicator = None
        self._boundaries = None

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
            self._initialize_tile_communicator
        return self._tile_communicator
    
    def _initialize_tile_communicator(self):
        raise NotImplementedError()

    def start_halo_update(self, quantity: Quantity, n_ghost: int):
        """Initiate an asynchronous halo update of a quantity."""
        if n_ghost == 0:
            raise ValueError('cannot perform a halo update on zero ghost cells')
        for boundary_type, boundary in self.boundaries.items():
            data = boundary.send_view(quantity, n_points=n_ghost)
            data = quantity.np.ascontiguousarray(
                rotate_data(data, quantity.metadata, boundary.n_clockwise_rotations)
            )
            self.comm.Isend(data, dest=boundary.to_rank)

    def finish_halo_update(self, quantity: Quantity, n_ghost: int):
        """Complete an asynchronous halo update of a quantity."""
        for boundary_type, boundary in self.boundaries.items():
            dest_view = boundary.recv_view(quantity, n_points=n_ghost)
            dest_buffer = quantity.np.empty(dest_view.shape, dtype=dest_view.dtype)
            self.comm.Recv(dest_buffer, source=boundary.to_rank)
            dest_view[:] = dest_buffer

    @property
    def rank(self) -> int:
        """rank of the current process on the cubed sphere"""
        return self.comm.Get_rank()


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

