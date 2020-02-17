from typing import Tuple
import dataclasses
from . import constants
import numpy as np
import xarray as xr


@dataclasses.dataclass
class ArrayMetadata:
    dims: Tuple[str, ...]
    units: str
    dtype: type


def get_tile_index(rank, total_ranks):
    """Returns the tile number for a given rank and total number of ranks.
    """
    if total_ranks % 6 != 0:
        raise ValueError(f'total_ranks {total_ranks} is not evenly divisible by 6')
    ranks_per_tile = total_ranks // 6
    return rank // ranks_per_tile


def get_tile_number(tile_rank, total_ranks):
    """Returns the tile number for a given rank and total number of ranks.
    """
    FutureWarning(
        'get_tile_number will be removed in a later version, '
        'use get_tile_index(rank, total_ranks) + 1 instead'
    )
    if total_ranks % 6 != 0:
        raise ValueError(f'total_ranks {total_ranks} is not evenly divisible by 6')
    ranks_per_tile = total_ranks // 6
    return tile_rank // ranks_per_tile + 1


class Partitioner:

    def __init__(
            self,
            rank: int,
            total_ranks: int,
            nz: int,
            ny: int,
            nx: int,
            layout: Tuple[int, int]
    ):
        """Create a Grid2D object for horizontal domain decomposition on a single tile.
        
        Args:
            rank: the rank of the process
            total_ranks: total number of processes
            ny: number of grid cells along the y-direction
            nx: number of grid cells along the x-direction
            layout: (x_subtiles, y_subtiles) specifying how the tile is split in the
                horizontal across multiple processes each with their own subtile.
        """
        self.rank = rank
        self.total_ranks = total_ranks
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self._validate_layout(layout)
        self.layout = layout

    @classmethod
    def from_namelist(cls, rank, total_ranks, namelist):
        return cls(
            rank=rank,
            total_ranks=total_ranks,
            nz=namelist['fv_core_nml']['npz'],
            ny=namelist['fv_core_nml']['npy'] - 1,
            nx=namelist['fv_core_nml']['npx'] - 1,
            layout=namelist['fv_core_nml']['layout'])

    def _validate_layout(self, layout):
        if layout[0] * layout[1] != self.ranks_per_tile:
            raise ValueError(
                f"layout must include {self.ranks_per_tile} ranks per tile, "
                f"but {layout} was given"
            )

    @property
    def ny_rank(self):
        return self.ny // self.layout[0]

    @property
    def nx_rank(self):
        return self.nx // self.layout[1]

    @property
    def tile(self):
        return get_tile_index(self.rank, self.total_ranks)

    @property
    def ranks_per_tile(self):
        return self.total_ranks // 6

    @property
    def tile_master_rank(self):
        return self.ranks_per_tile * (self.rank // self.ranks_per_tile)

    @property
    def subtile_index(self):
        return subtile_index(self.rank, self.ranks_per_tile, self.layout)

    def tile_extent(self, array_dims):
        return tile_extent(self.nz, self.ny, self.nx, array_dims)

    def subtile_range(
            self,
            array_dims: Tuple[str, ...],
            overlap: bool = False) -> Tuple[slice, slice]:
        """Given array dimensions, return the range of the subtile within the tile domain.

        Assumes 2D arrays are shape [ny, nx] and higher dimensional arrays
        are shape [nz, ny, nx, ...].

        Args:
            array_dims: the array dimensions
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            y_range: the y range of the array on the tile
            x_range: the x range of the array on the tile
        """
        return subtile_range(
            array_dims, self.nz, self.ny_rank, self.nx_rank, self.layout, self.subtile_index,
            overlap=overlap,
        )

    def scatter_tile(self, tile_comm, array, metadata):
        shape = tile_extent(nz=self.nz, nx=self.nx_rank, ny=self.ny_rank, array_dims=metadata.dims)
        if tile_comm.Get_rank() == constants.MASTER_RANK:
            sendbuf = np.empty((self.ranks_per_tile,) + shape, dtype=metadata.dtype)
            for add_rank in range(0, self.ranks_per_tile):
                rank = self.rank + add_rank
                idx = subtile_index(rank, self.ranks_per_tile, self.layout)
                subtile_slice = subtile_range(
                    array_dims=metadata.dims,
                    nz=self.nz,
                    ny_rank=self.ny_rank,
                    nx_rank=self.nx_rank,
                    layout=self.layout,
                    subtile_index=idx,
                    overlap=True
                )
                sendbuf[add_rank, :] = np.ascontiguousarray(array[subtile_slice])
        else:
            sendbuf = None
        recvbuf = np.empty(shape, dtype=metadata.dtype)
        tile_comm.Scatter(sendbuf, recvbuf, root=0)
        return xr.DataArray(
            recvbuf,
            dims=metadata.dims,
            attrs={'units': metadata.units}
        )


def subtile_index(rank, ranks_per_tile, layout):
    within_tile_rank = rank % ranks_per_tile
    j = within_tile_rank // layout[0]
    i = within_tile_rank % layout[0]
    return j, i


def bcast_metadata_list(comm, array_list):
    is_master = comm.Get_rank() == constants.MASTER_RANK
    if is_master:
        metadata_list = []
        for array in array_list:
            metadata_list.append(ArrayMetadata(dims=array.dims, units=array.attrs['units'], dtype=array.dtype))
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.MASTER_RANK)


def bcast_metadata(comm, array):
    return bcast_metadata_list(comm, [array])[0]


def tile_extent(nz, ny, nx, array_dims):
    dim_extents = {
        constants.X_DIM: nx,
        constants.X_INTERFACE_DIM: nx + 1,
        constants.Y_DIM: ny,
        constants.Y_INTERFACE_DIM: ny + 1,
        constants.Z_DIM: nz,
        constants.Z_INTERFACE_DIM: nz + 1,
    }
    return_extents = [dim_extents[dim] for dim in array_dims]
    return tuple(return_extents)


def subtile_range(array_dims, nz, ny_rank, nx_rank, layout, subtile_index, overlap=False):
    j_subtile, i_subtile = subtile_index
    y_start, x_start = j_subtile * ny_rank, i_subtile * nx_rank
    subtile_extent = tile_extent(nz, ny_rank, nx_rank, array_dims)
    # discard last index for interface variables, unless you're the last rank
    # done so that only one rank is responsible for the shared interface point
    return_list = []
    for dim, extent in zip(array_dims, subtile_extent):
        if dim in (constants.Z_DIM, constants.Z_INTERFACE_DIM):
            return_list.append(slice(0, extent))
        elif not overlap and (dim == constants.Y_INTERFACE_DIM and j_subtile != layout[0] - 1):
            return_list.append(slice(y_start, y_start + extent - 1))
        elif dim in (constants.Y_DIM, constants.Y_INTERFACE_DIM):
            return_list.append(slice(y_start, y_start + extent))
        elif not overlap and (dim == constants.X_INTERFACE_DIM and i_subtile != layout[1] - 1):
            return_list.append(slice(x_start, x_start + extent - 1))
        elif dim in (constants.X_DIM, constants.X_INTERFACE_DIM):
            return_list.append(slice(x_start, x_start + extent))
    return tuple(return_list)
