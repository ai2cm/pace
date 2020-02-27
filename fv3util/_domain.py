from typing import Tuple, Dict
import copy
import dataclasses
from . import constants
import numpy as np
import xarray as xr


@dataclasses.dataclass
class ArrayMetadata:
    dims: Tuple[str, ...]  # dimension names
    dim_lengths: Dict[str, int]  # defines lengths of non-horizontal dimensions
    units: str
    dtype: type

    @classmethod
    def from_data_array(cls, data_array):
        dim_lengths = dict(zip(data_array.dims, data_array.shape))
        for dim in constants.HORIZONTAL_DIMS:
            if dim in dim_lengths:
                dim_lengths.pop(dim)
        return cls(
            data_array.dims, dim_lengths, data_array.attrs['units'], data_array.dtype
        )


def get_tile_index(rank, total_ranks):
    """Returns the tile index for a given rank and total number of ranks.
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
    return get_tile_index(tile_rank, total_ranks) + 1


class CubedSpherePartitioner:

    def __init__(
            self,
            ny: int,
            nx: int,
            layout: Tuple[int, int]
    ):
        """Create an object for fv3gfs domain decomposition.
        
        Args:
            ny: number of grid cell centers along the y-direction
            nx: number of grid cell centers along the x-direction
            layout: (y_subtiles, x_subtiles) specifying how the tile is split in the
                horizontal across multiple processes each with their own subtile.
        """
        self.ny = ny
        self.nx = nx
        self.layout = layout

    @classmethod
    def from_namelist(cls, namelist):
        """Create a Partitioner from a Fortran namelist. Infers dimensions in number
        of grid cell centers based on namelist parameters.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(
            ny=namelist['fv_core_nml']['npy'] - 1,
            nx=namelist['fv_core_nml']['npx'] - 1,
            layout=namelist['fv_core_nml']['layout'])

    @property
    def ny_rank(self):
        """the number of cell centers in the y direction on each rank/subtile"""
        return self.ny // self.layout[0]

    @property
    def nx_rank(self):
        """the number of cell centers in the x direction on each rank/subtile"""
        return self.nx // self.layout[1]

    def tile(self, rank):
        """Return the tile index of a given rank"""
        return int(rank // self.ranks_per_tile)

    @property
    def ranks_per_tile(self):
        """the number of ranks per tile"""
        return self.layout[0] * self.layout[1]

    @property
    def cube_total_ranks(self):
        """the number of ranks per tile"""
        return self.ranks_per_tile * 6

    def tile_master_rank(self, rank):
        """Return the lowest rank on the same tile as a given rank."""
        return self.ranks_per_tile * (rank // self.ranks_per_tile)

    def subtile_index(self, rank):
        """Return the (y, x) subtile position of a given rank as an integer number of subtiles."""
        return subtile_index(rank, self.ranks_per_tile, self.layout)

    def tile_extent(self, array_metadata: ArrayMetadata) -> Tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            array_metadata

        Returns:
            extent: shape of full tile representation
        """
        return tile_extent(
            self.ny, self.nx, array_metadata.dims, array_metadata.dim_lengths
        )

    def subtile_slice(
            self,
            rank,
            array_metadata: ArrayMetadata,
            overlap: bool = False) -> Tuple[slice, slice]:
        """Return the subtile slice of a given rank on an array.

        Args:
            rank: the rank of the process
            array_metadata: the array metadata
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            y_range: the y range of the array on the tile
            x_range: the x range of the array on the tile
        """
        subtile_index = self.subtile_index(rank)
        return subtile_slice(
            array_metadata, self.ny_rank, self.nx_rank, self.layout, subtile_index,
            overlap=overlap,
        )

    def scatter_tile(self, tile_comm, array, metadata):
        shape = tile_extent(
            nx=self.nx_rank,
            ny=self.ny_rank,
            array_dims=metadata.dims,
            dim_lengths=metadata.dim_lengths
        )
        if tile_comm.Get_rank() == constants.MASTER_RANK:
            sendbuf = np.empty((self.ranks_per_tile,) + shape, dtype=metadata.dtype)
            for rank in range(0, self.ranks_per_tile):
                subtile_slice = self.subtile_slice(
                    rank,
                    array_metadata=metadata,
                    overlap=True,
                )
                sendbuf[rank, :] = np.ascontiguousarray(array.values[subtile_slice])
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
    j = within_tile_rank // layout[1]
    i = within_tile_rank % layout[1]
    return j, i


def bcast_metadata_list(comm, array_list):
    is_master = comm.Get_rank() == constants.MASTER_RANK
    if is_master:
        metadata_list = []
        for array in array_list:
            metadata_list.append(ArrayMetadata.from_data_array(array))
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.MASTER_RANK)


def bcast_metadata(comm, array):
    return bcast_metadata_list(comm, [array])[0]


def tile_extent(ny, nx, array_dims, dim_lengths):
    dim_extents = copy.deepcopy(dim_lengths)
    dim_extents.update({
        constants.X_DIM: nx,
        constants.X_INTERFACE_DIM: nx + 1,
        constants.Y_DIM: ny,
        constants.Y_INTERFACE_DIM: ny + 1,
    })
    return_extents = [dim_extents[dim] for dim in array_dims]
    return tuple(return_extents)


def subtile_slice(array_metadata, ny_rank, nx_rank, layout, subtile_index, overlap=False):
    j_subtile, i_subtile = subtile_index
    y_start, x_start = j_subtile * ny_rank, i_subtile * nx_rank
    subtile_extent = tile_extent(
        ny_rank, nx_rank, array_metadata.dims, array_metadata.dim_lengths
    )
    # discard last index for interface variables, unless you're the last rank
    # done so that only one rank is responsible for the shared interface point
    return_list = []
    for dim, extent in zip(array_metadata.dims, subtile_extent):
        if not overlap and (dim == constants.Y_INTERFACE_DIM and j_subtile != layout[0] - 1):
            return_list.append(slice(y_start, y_start + extent - 1))
        elif dim in (constants.Y_DIM, constants.Y_INTERFACE_DIM):
            return_list.append(slice(y_start, y_start + extent))
        elif not overlap and (dim == constants.X_INTERFACE_DIM and i_subtile != layout[1] - 1):
            return_list.append(slice(x_start, x_start + extent - 1))
        elif dim in (constants.X_DIM, constants.X_INTERFACE_DIM):
            return_list.append(slice(x_start, x_start + extent))
        else:
            return_list.append(slice(0, extent))
    return tuple(return_list)
