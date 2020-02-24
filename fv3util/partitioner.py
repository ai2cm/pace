from typing import Tuple
import functools
import dataclasses
from . import constants
import numpy as np
import xarray as xr
from .quantity import QuantityMetadata, Quantity

BOUNDARY_CACHE_SIZE=None


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


@dataclasses.dataclass
class Boundary:
    from_rank: int
    to_rank: int
    n_clockwise_rotations: int

    def rotate(self, y_data, x_data):
        if self.n_clockwise_rotations % 4 == 0:
            pass
        elif self.n_clockwise_rotations % 4 == 1:
            y_data[:], x_data[:] = -x_data[:], y_data[:]
        elif self.n_clockwise_rotations % 4 == 2:
            y_data[:] = -y_data[:]
            x_data[:] = -x_data[:]
        elif self.n_clockwise_rotations % 4 == 3:
            y_data, x_data = x_data[:], -y_data[:]


class Partitioner:

    def __init__(
            self,
            nz: int,
            ny: int,
            nx: int,
            layout: Tuple[int, int]
    ):
        """Create an object for fv3gfs domain decomposition.
        
        Args:
            nz: number of grid cell centers along the y-direction
            ny: number of grid cell centers along the y-direction
            nx: number of grid cell centers along the x-direction
            layout: (x_subtiles, y_subtiles) specifying how the tile is split in the
                horizontal across multiple processes each with their own subtile.
        """
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self._layout = layout
        self.total_ranks = 6 * layout[0] * layout[1]

    @property
    def layout(self):
        return self._layout

    def _ensure_square_layout(self):
        if self.layout[0] != self.layout[1]:
            raise NotImplementedError('currently only square layouts are supported')

    @classmethod
    def from_namelist(cls, namelist):
        """Create a Partitioner from a Fortran namelist. Infers dimensions in number
        of grid cell centers based on namelist parameters.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(
            nz=namelist['fv_core_nml']['npz'],
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
        return get_tile_index(rank, self.total_ranks)

    @property
    def ranks_per_tile(self):
        """the number of ranks per tile"""
        return self.total_ranks // 6

    def tile_master_rank(self, rank):
        """Return the lowest rank on the same tile as a given rank."""
        return self.ranks_per_tile * (rank // self.ranks_per_tile)

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def left_edge(self, rank):
        self._ensure_square_layout()
        if on_tile_left(self.subtile_index(rank)):
            if is_even(self.tile(rank)):
                to_master_rank = self.tile_master_rank(rank - 2 * self.ranks_per_tile)
                tile_rank = rank % self.ranks_per_tile
                to_tile_rank = fliplr_subtile_rank(
                    rotate_subtile_rank(
                        tile_rank, self.layout, n_clockwise_rotations=1
                    ),
                    self.layout
                )
                to_rank = to_master_rank + to_tile_rank
                rotations = 1
            else:
                to_rank = rank - self.ranks_per_tile + self.layout[0] - 1
                rotations = 0
        else:
            to_rank = rank - 1
            rotations = 0
        to_rank = to_rank % self.total_ranks
        return Boundary(from_rank=rank, to_rank=to_rank, n_clockwise_rotations=rotations)

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def right_edge(self, rank):
        self._ensure_square_layout()
        self._ensure_square_layout()
        if on_tile_right(self.subtile_index(rank), self.layout):
            if not is_even(self.tile(rank)):
                to_master_rank = self.tile_master_rank(rank + 2 * self.ranks_per_tile)
                tile_rank = rank % self.ranks_per_tile
                to_tile_rank = fliplr_subtile_rank(
                    rotate_subtile_rank(
                        tile_rank, self.layout, n_clockwise_rotations=1
                    ),
                    self.layout
                )
                to_rank = to_master_rank + to_tile_rank
                rotations = 1
            else:
                to_rank = rank + self.ranks_per_tile - self.layout[0] + 1
                rotations = 0
        else:
            to_rank = rank + 1
            rotations = 0
        to_rank = to_rank % self.total_ranks
        return Boundary(from_rank=rank, to_rank=to_rank, n_clockwise_rotations=rotations)

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def top_edge(self, rank):
        self._ensure_square_layout()
        if on_tile_top(self.subtile_index(rank), self.layout):
            if is_even(self.tile(rank)):
                to_master_rank = (self.tile(rank) + 2) * self.ranks_per_tile
                tile_rank = rank % self.ranks_per_tile
                to_tile_rank = fliplr_subtile_rank(
                    rotate_subtile_rank(
                        tile_rank, self.layout, n_clockwise_rotations=1
                    ),
                    self.layout
                )
                to_rank = to_master_rank + to_tile_rank
                rotations = 3
            else:
                to_master_rank = (self.tile(rank) + 1) * self.ranks_per_tile
                tile_rank = rank % self.ranks_per_tile
                to_tile_rank = flipud_subtile_rank(tile_rank, self.layout)
                to_rank = to_master_rank + to_tile_rank
                rotations = 0
        else:
            to_rank = rank + self.layout[1]
            rotations = 0
        to_rank = to_rank % self.total_ranks
        return Boundary(from_rank=rank, to_rank=to_rank, n_clockwise_rotations=rotations)

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def bottom_edge(self, rank):
        self._ensure_square_layout()
        if (
                on_tile_bottom(self.subtile_index(rank)) and
                not is_even(self.tile(rank))
        ):
            to_master_rank = (self.tile(rank) - 2) * self.ranks_per_tile
            tile_rank = rank % self.ranks_per_tile
            to_tile_rank = fliplr_subtile_rank(
                rotate_subtile_rank(
                    tile_rank, self.layout, n_clockwise_rotations=1
                ),
                self.layout
            )
            to_rank = to_master_rank + to_tile_rank
            rotations = 3
        else:
            to_rank = rank - self.layout[1]
            rotations = 0
        to_rank = to_rank % self.total_ranks
        return Boundary(from_rank=rank, to_rank=to_rank, n_clockwise_rotations=rotations)

    def top_left_corner(self, rank):
        if (on_tile_top(self.subtile_index(rank), self.layout) and
                on_tile_left(self.subtile_index(rank))):
            corner = None
        else:
            if is_even(self.tile(rank)) and on_tile_left(self.subtile_index(rank)):
                second_edge = self.left_edge
            else:
                second_edge = self.top_edge
            corner = self._get_corner(rank, self.left_edge, second_edge)
        return corner

    def top_right_corner(self, rank):
        if (on_tile_top(self.subtile_index(rank), self.layout) and
                on_tile_right(self.subtile_index(rank), self.layout)):
            corner = None
        else:
            if is_even(self.tile(rank)) and on_tile_top(self.subtile_index(rank), self.layout):
                second_edge = self.bottom_edge
            else:
                second_edge = self.right_edge
            corner = self._get_corner(rank, self.top_edge, second_edge)
        return corner

    def bottom_left_corner(self, rank):
        if (on_tile_bottom(self.subtile_index(rank)) and
                on_tile_left(self.subtile_index(rank))):
            corner = None
        else:
            if not is_even(self.tile(rank)) and on_tile_bottom(self.subtile_index(rank)):
                second_edge = self.top_edge
            else:
                second_edge = self.left_edge
            corner = self._get_corner(rank, self.bottom_edge, second_edge)
        return corner

    def bottom_right_corner(self, rank):
        if (on_tile_bottom(self.subtile_index(rank)) and
                on_tile_right(self.subtile_index(rank), self.layout)):
            corner = None
        else:
            if not is_even(self.tile(rank)) and on_tile_bottom(self.subtile_index(rank)):
                second_edge = self.bottom_edge
            else:
                second_edge = self.right_edge
            corner = self._get_corner(rank, self.bottom_edge, second_edge)
        return corner

    def _get_corner(self, rank, edge_func_1, edge_func_2):
        edge_1 = edge_func_1(rank)
        edge_2 = edge_func_2(edge_1.to_rank)
        rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
        return Boundary(
            from_rank=rank,
            to_rank=edge_2.to_rank,
            n_clockwise_rotations=rotations
        )

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def subtile_index(self, rank):
        """Return the (y, x) subtile position of a given rank as an integer number of subtiles."""
        return subtile_index(rank, self.ranks_per_tile, self.layout)

    def tile_extent(self, array_dims):
        """Return the shape of a full tile representation for the given dimensions."""
        return tile_extent(self.nz, self.ny, self.nx, array_dims)

    def subtile_extent(self, array_dims):
        """Return the shape of a single rank representation for the given dimensions."""
        return tile_extent(self.nz, self.ny_rank, self.nx_rank, array_dims)

    def subtile_slice(
            self,
            rank,
            array_dims: Tuple[str, ...],
            overlap: bool = False) -> Tuple[slice, slice]:
        """Return the subtile slice of a given rank on an array.

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
        subtile_index = self.subtile_index(rank)
        return subtile_slice(
            array_dims, self.nz, self.ny_rank, self.nx_rank, self.layout, subtile_index,
            overlap=overlap,
        )


def on_tile_left(subtile_index):
    return subtile_index[1] == 0


def on_tile_right(subtile_index, layout):
    return subtile_index[1] == layout[1] - 1


def on_tile_top(subtile_index, layout):
    return subtile_index[0] == layout[0] - 1


def on_tile_bottom(subtile_index):
    return subtile_index[0] == 0


def rotate_subtile_rank(rank, layout, n_clockwise_rotations):
    if n_clockwise_rotations == 0:
        to_tile_rank = rank
    elif n_clockwise_rotations == 1:
        total_ranks = layout[0] * layout[1]
        rank_array = np.arange(total_ranks).reshape(layout)
        rotated_rank_array = np.rot90(rank_array)
        to_tile_rank = rank_array[np.where(rotated_rank_array == rank)][0]
    else:
        raise NotImplementedError()
    return to_tile_rank


def transpose_subtile_rank(rank, layout):
    return transform_subtile_rank(np.transpose, rank, layout)


def fliplr_subtile_rank(rank, layout):
    return transform_subtile_rank(np.fliplr, rank, layout)


def flipud_subtile_rank(rank, layout):
    return transform_subtile_rank(np.flipud, rank, layout)


def transform_subtile_rank(transform_func, rank, layout):
    total_ranks = layout[0] * layout[1]
    rank_array = np.arange(total_ranks).reshape(layout)
    fliplr_rank_array = transform_func(rank_array)
    return rank_array[np.where(fliplr_rank_array == rank)][0]


def subtile_index(rank, ranks_per_tile, layout):
    within_tile_rank = rank % ranks_per_tile
    j = within_tile_rank // layout[1]
    i = within_tile_rank % layout[1]
    return j, i


def is_even(value):
    return value % 2 == 0


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


def subtile_slice(array_dims, nz, ny_rank, nx_rank, layout, subtile_index, overlap=False):
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
