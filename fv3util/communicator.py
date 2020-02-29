from .quantity import Quantity, QuantityMetadata
from .partitioner import CubedSpherePartitioner, TilePartitioner
from . import constants
import functools


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

    def __init__(self, tile_comm, partitioner: TilePartitioner):
        self.partitioner = partitioner
        self.tile_comm = tile_comm

    def scatter_tile(
            self,
            metadata: QuantityMetadata,
            send_quantity: Quantity = None,
            recv_quantity: Quantity = None):
        shape = self.partitioner.subtile_extent(metadata)
        if self.tile_comm.Get_rank() == constants.MASTER_RANK:
            sendbuf = metadata.np.empty(
                (self.partitioner.total_ranks,) + shape,
                dtype=metadata.dtype
            )
            for rank in range(0, self.partitioner.total_ranks):
                subtile_slice = self.partitioner.subtile_slice(
                    rank,
                    metadata=metadata,
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
    def rank(self):
        return self.tile_comm.Get_rank()
    

class CubedSphereCommunicator:

    def __init__(self, comm, partitioner: CubedSpherePartitioner):
        self.comm = comm
        self.partitioner = partitioner
        self._tile_communicator = None
        self._boundaries = None

    @property
    def boundaries(self):
        if self._boundaries is None:
            self._boundaries = {}
            for boundary_type in constants.BOUNDARY_TYPES:
                boundary = self.partitioner.boundary(boundary_type, self.rank)
                if boundary is not None:
                    self._boundaries[boundary_type] = boundary
        return self._boundaries

    @property
    def tile_communicator(self):
        if self._tile_communicator is None:
            self._initialize_tile_communicator
        return self._tile_communicator
    
    def _initialize_tile_communicator(self):
        raise NotImplementedError()

    def start_halo_update(self, quantity, n_ghost):
        if n_ghost == 0:
            raise ValueError('cannot perform a halo update on zero ghost cells')
        for boundary_type, boundary in self.boundaries.items():
            data = quantity.boundary_data(
                boundary_type, n_points=n_ghost, interior=True
            )
            data = quantity.np.ascontiguousarray(
                rotate_data(data, quantity.metadata, boundary.n_clockwise_rotations)
            )
            self.comm.Isend(data, dest=boundary.to_rank)

    def finish_halo_update(self, quantity, n_ghost):
        for boundary_type, boundary in self.boundaries.items():
            dest_view = quantity.boundary_data(
                boundary_type, n_points=n_ghost, interior=False
            )
            dest_buffer = quantity.np.empty(dest_view.shape, dtype=dest_view.dtype)
            self.comm.Recv(dest_buffer, source=boundary.to_rank)
            dest_view[:] = dest_buffer

    @property
    def rank(self):
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

