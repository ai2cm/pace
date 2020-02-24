from .quantity import Quantity, QuantityMetadata
from .partitioner import Partitioner
from . import constants


class TileCommunicator:

    def __init__(self, tile_comm, partitioner: Partitioner):
        self.partitioner = partitioner
        self.tile_comm = tile_comm

    def scatter_tile(
            self,
            metadata: QuantityMetadata,
            send_quantity: Quantity = None,
            recv_quantity: Quantity = None):
        shape = self.partitioner.subtile_extent(array_dims=metadata.dims)
        if self.tile_comm.Get_rank() == constants.MASTER_RANK:
            sendbuf = metadata.np.empty(
                (self.partitioner.ranks_per_tile,) + shape,
                dtype=metadata.dtype
            )
            for rank in range(0, self.partitioner.ranks_per_tile):
                subtile_slice = self.partitioner.subtile_slice(
                    rank,
                    array_dims=metadata.dims,
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
        return self.tile_comm.rank
    

class CubedSphereCommunicator:

    def __init__(self, comm, partitioner):
        self.comm = comm
        self.partitioner = partitioner
        self._tile_communicator = None

    @property
    def tile_communicator(self):
        if self._tile_communicator is None:
            self._initialize_tile_communicator
        return self._tile_communicator
    
    def _initialize_tile_communicator(self):
        raise NotImplementedError()

    def start_halo_update(self, comm, quantity, n_ghost):
        raise NotImplementedError()

    def finish_halo_update(self, comm, quantity, n_ghost):
        raise NotImplementedError()
