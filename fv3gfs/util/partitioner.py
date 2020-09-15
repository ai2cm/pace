from typing import Tuple, Callable, Iterable
import warnings
import copy
import functools
import dataclasses
from . import constants, utils
from .constants import (
    NORTH,
    SOUTH,
    WEST,
    EAST,
    NORTHWEST,
    NORTHEAST,
    SOUTHWEST,
    SOUTHEAST,
)
import numpy as np
from . import boundary as bd
from .quantity import QuantityMetadata

BOUNDARY_CACHE_SIZE = None


__all__ = ["TilePartitioner", "CubedSpherePartitioner", "get_tile_index"]


def get_tile_index(rank, total_ranks):
    """Returns the tile number for a given rank and total number of ranks.
    """
    if total_ranks % 6 != 0:
        raise ValueError(f"total_ranks {total_ranks} is not evenly divisible by 6")
    ranks_per_tile = total_ranks // 6
    return rank // ranks_per_tile


def get_tile_number(tile_rank, total_ranks):
    """Returns the tile number for a given rank and total number of ranks.
    """
    FutureWarning(
        "get_tile_number will be removed in a later version, "
        "use get_tile_index(rank, total_ranks) + 1 instead"
    )
    if total_ranks % 6 != 0:
        raise ValueError(f"total_ranks {total_ranks} is not evenly divisible by 6")
    ranks_per_tile = total_ranks // 6
    return tile_rank // ranks_per_tile + 1


class TilePartitioner:
    def __init__(
        self, layout: Tuple[int, int],
    ):
        """Create an object for fv3gfs tile decomposition.
        """
        self.layout = layout

    @classmethod
    def from_namelist(cls, namelist):
        """Initialize a TilePartitioner from a Fortran namelist.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(layout=namelist["fv_core_nml"]["layout"])

    def subtile_index(self, rank: int) -> Tuple[int, int]:
        """Return the (y, x) subtile position of a given rank as an integer number of subtiles."""
        return subtile_index(rank, self.total_ranks, self.layout)

    @property
    def total_ranks(self) -> int:
        return self.layout[0] * self.layout[1]

    def tile_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full tile representation
        """
        return tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_extent(self, tile_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions."""
        return rank_extent_from_tile_metadata(
            tile_metadata.dims, tile_metadata.extent, self.layout
        )

    def subtile_nx(self, nx):
        warnings.warn(
            "subtile_nx method may be removed soon as it only supports constant-size subdomains, use subtile_slice instead",
            warnings.DeprecationWarning,
        )
        return nx // self.layout[1]

    def subtile_ny(self, ny):
        warnings.warn(
            "subtile_ny method may be removed soon as it only supports constant-size subdomains, use subtile_slice instead",
            warnings.DeprecationWarning,
        )
        return ny // self.layout[0]

    def subtile_slice(
        self,
        rank: int,
        tile_dims: Iterable[str],
        tile_extent: Iterable[int],
        overlap: bool = False,
    ) -> Tuple[slice, slice]:
        """Return the subtile slice of a given rank on an array.

        Args:
            rank: the rank of the process
            tile_metadata: the metadata for a quantity on a tile
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            y_range: the y range of the array on the tile
            x_range: the x range of the array on the tile
        """
        return subtile_slice(
            tile_dims,
            tile_extent,
            self.layout,
            self.subtile_index(rank),
            overlap=overlap,
        )

    def on_tile_top(self, rank: int) -> bool:
        return on_tile_top(self.subtile_index(rank), self.layout)

    def on_tile_bottom(self, rank: int) -> bool:
        return on_tile_bottom(self.subtile_index(rank))

    def on_tile_left(self, rank: int) -> bool:
        return on_tile_left(self.subtile_index(rank))

    def on_tile_right(self, rank: int) -> bool:
        return on_tile_right(self.subtile_index(rank), self.layout)

    def boundary(self, boundary_type: str, rank: int) -> bd.SimpleBoundary:
        """Returns a boundary of the requested type for a given rank.

        Target ranks will be on the same tile as the given rank, wrapping around as
        in a doubly-periodic boundary condition.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        boundary = copy.copy(self._cached_boundary(boundary_type, rank))
        return boundary

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def _cached_boundary(self, boundary_type: str, rank: int) -> bd.SimpleBoundary:
        boundary = {
            WEST: self._left_edge,
            EAST: self._right_edge,
            NORTH: self._top_edge,
            SOUTH: self._bottom_edge,
            NORTHWEST: self._top_left_corner,
            NORTHEAST: self._top_right_corner,
            SOUTHWEST: self._bottom_left_corner,
            SOUTHEAST: self._bottom_right_corner,
        }[boundary_type](rank)
        return boundary

    def _left_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_left(rank):
            to_rank = rank + self.layout[1] - 1
        else:
            to_rank = rank - 1
        return bd.SimpleBoundary(
            boundary_type=constants.WEST,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _right_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_right(rank):
            to_rank = rank - self.layout[1] + 1
        else:
            to_rank = rank + 1
        return bd.SimpleBoundary(
            boundary_type=constants.EAST,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _top_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_top(rank):
            to_rank = rank - (self.layout[0] - 1) * self.layout[1]
        else:
            to_rank = rank + self.layout[1]
        return bd.SimpleBoundary(
            boundary_type=constants.NORTH,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _bottom_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_bottom(rank):
            to_rank = rank + (self.layout[0] - 1) * self.layout[1]
        else:
            to_rank = rank - self.layout[1]
        return bd.SimpleBoundary(
            boundary_type=constants.SOUTH,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _top_left_corner(self, rank: int) -> bd.SimpleBoundary:
        return _get_corner(constants.NORTHWEST, rank, self._left_edge, self._top_edge)

    def _top_right_corner(self, rank: int) -> bd.SimpleBoundary:
        return _get_corner(constants.NORTHEAST, rank, self._right_edge, self._top_edge)

    def _bottom_left_corner(self, rank: int) -> bd.SimpleBoundary:
        return _get_corner(
            constants.SOUTHWEST, rank, self._left_edge, self._bottom_edge
        )

    def _bottom_right_corner(self, rank: int) -> bd.SimpleBoundary:
        return _get_corner(
            constants.SOUTHEAST, rank, self._right_edge, self._bottom_edge
        )

    def fliplr_rank(self, rank: int) -> int:
        return fliplr_subtile_rank(rank, self.layout)

    def rotate_rank(self, rank: int, n_clockwise_rotations: int) -> int:
        return rotate_subtile_rank(rank, self.layout, n_clockwise_rotations)


def _get_corner(
    boundary_type: int,
    rank: int,
    edge_func_1: Callable[[int], bd.Boundary],
    edge_func_2: Callable[[int], bd.Boundary],
):
    edge_1 = edge_func_1(rank)
    edge_2 = edge_func_2(edge_1.to_rank)
    rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
    return bd.SimpleBoundary(
        boundary_type=boundary_type,
        from_rank=rank,
        to_rank=edge_2.to_rank,
        n_clockwise_rotations=rotations,
    )


class CubedSpherePartitioner:
    def __init__(self, tile: TilePartitioner):
        """Create an object for fv3gfs cubed-sphere domain decomposition.
        
        Args:
            tile: partitioner for the cube faces
        """
        if not isinstance(tile, TilePartitioner):
            raise TypeError("tile must be a TilePartitioner")
        self.tile = tile

    @classmethod
    def from_namelist(cls, namelist):
        """Initialize a CubedSpherePartitioner from a Fortran namelist.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(TilePartitioner.from_namelist(namelist))

    def _ensure_square_layout(self) -> None:
        if not self.tile.layout[0] == self.tile.layout[1]:
            raise NotImplementedError("currently only square layouts are supported")

    def tile_index(self, rank: int) -> Tuple[int, int]:
        """Returns the tile index of a given rank"""
        return get_tile_index(rank, self.total_ranks)

    def tile_master_rank(self, rank: int) -> int:
        """Returns the lowest rank on the same tile as a given rank."""
        return self.tile.total_ranks * (rank // self.tile.total_ranks)

    @property
    def layout(self):
        return self.tile.layout

    @property
    def total_ranks(self) -> int:
        """the number of ranks on the cubed sphere"""
        return 6 * self.tile.total_ranks

    def boundary(self, boundary_type: str, rank: int) -> bd.SimpleBoundary:
        """Returns a boundary of the requested type for a given rank.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        boundary = copy.copy(self._cached_boundary(boundary_type, rank))
        return boundary

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def _cached_boundary(self, boundary_type: str, rank: int) -> bd.SimpleBoundary:
        boundary = {
            WEST: self._left_edge,
            EAST: self._right_edge,
            NORTH: self._top_edge,
            SOUTH: self._bottom_edge,
            NORTHWEST: self._top_left_corner,
            NORTHEAST: self._top_right_corner,
            SOUTHWEST: self._bottom_left_corner,
            SOUTHEAST: self._bottom_right_corner,
        }[boundary_type](rank)
        if boundary is not None:
            boundary.to_rank = boundary.to_rank % self.total_ranks
        return boundary

    def _left_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_left(rank):
            if is_even(self.tile_index(rank)):
                to_master_rank = self.tile_master_rank(rank - 2 * self.tile.total_ranks)
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                to_rank = to_master_rank + to_tile_rank
                rotations = 1
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.WEST,
                    from_rank=rank,
                    to_rank=to_rank,
                    n_clockwise_rotations=rotations,
                )
            else:
                boundary = self.tile.boundary(WEST, rank=rank)
                boundary.to_rank -= self.tile.total_ranks
        else:
            boundary = self.tile.boundary(WEST, rank=rank)
        return boundary

    def _right_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_right(rank):
            if not is_even(self.tile_index(rank)):
                to_master_rank = self.tile_master_rank(rank + 2 * self.tile.total_ranks)
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.EAST,
                    from_rank=rank,
                    to_rank=to_master_rank + to_tile_rank,
                    n_clockwise_rotations=1,
                )
            else:
                boundary = self.tile.boundary(EAST, rank=rank)
                boundary.to_rank += self.tile.total_ranks
        else:
            boundary = self.tile.boundary(EAST, rank=rank)
        return boundary

    def _top_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_top(rank):
            if is_even(self.tile_index(rank)):
                to_master_rank = (self.tile_index(rank) + 2) * self.tile.total_ranks
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.NORTH,
                    from_rank=rank,
                    to_rank=to_master_rank + to_tile_rank,
                    n_clockwise_rotations=3,
                )
            else:
                boundary = self.tile.boundary(NORTH, rank)
                boundary.to_rank += self.tile.total_ranks
        else:
            boundary = self.tile.boundary(NORTH, rank=rank)
        return boundary

    def _bottom_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_bottom(rank) and not is_even(self.tile_index(rank)):
            to_master_rank = (self.tile_index(rank) - 2) * self.tile.total_ranks
            tile_rank = rank % self.tile.total_ranks
            to_tile_rank = self.tile.fliplr_rank(self.tile.rotate_rank(tile_rank, 1))
            boundary = bd.SimpleBoundary(
                boundary_type=constants.SOUTH,
                from_rank=rank,
                to_rank=to_master_rank + to_tile_rank,
                n_clockwise_rotations=3,
            )
        else:
            boundary = self.tile.boundary(SOUTH, rank=rank)
            if self.tile.on_tile_bottom(rank):
                boundary.to_rank -= self.tile.total_ranks
        return boundary

    def _top_left_corner(self, rank: int) -> bd.SimpleBoundary:
        if self.tile.on_tile_top(rank) and self.tile.on_tile_left(rank):
            corner = None
        else:
            if is_even(self.tile_index(rank)) and on_tile_left(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._left_edge
            else:
                second_edge = self._top_edge
            corner = self._get_corner(
                constants.NORTHWEST, rank, self._left_edge, second_edge
            )
        return corner

    def _top_right_corner(self, rank: int) -> bd.SimpleBoundary:
        if on_tile_top(self.tile.subtile_index(rank), self.layout) and on_tile_right(
            self.tile.subtile_index(rank), self.layout
        ):
            corner = None
        else:
            if is_even(self.tile_index(rank)) and on_tile_top(
                self.tile.subtile_index(rank), self.layout
            ):
                second_edge = self._bottom_edge
            else:
                second_edge = self._right_edge
            corner = self._get_corner(
                constants.NORTHEAST, rank, self._top_edge, second_edge
            )
        return corner

    def _bottom_left_corner(self, rank: int) -> bd.SimpleBoundary:
        if on_tile_bottom(self.tile.subtile_index(rank)) and on_tile_left(
            self.tile.subtile_index(rank)
        ):
            corner = None
        else:
            if not is_even(self.tile_index(rank)) and on_tile_bottom(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._top_edge
            else:
                second_edge = self._left_edge
            corner = self._get_corner(
                constants.SOUTHWEST, rank, self._bottom_edge, second_edge
            )
        return corner

    def _bottom_right_corner(self, rank: int) -> bd.SimpleBoundary:
        if on_tile_bottom(self.tile.subtile_index(rank)) and on_tile_right(
            self.tile.subtile_index(rank), self.layout
        ):
            corner = None
        else:
            if not is_even(self.tile_index(rank)) and on_tile_bottom(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._bottom_edge
            else:
                second_edge = self._right_edge
            corner = self._get_corner(
                constants.SOUTHEAST, rank, self._bottom_edge, second_edge
            )
        return corner

    def _get_corner(
        self,
        boundary_type: int,
        rank: int,
        edge_func_1: Callable[[int], bd.Boundary],
        edge_func_2: Callable[[int], bd.Boundary],
    ):
        edge_1 = edge_func_1(rank)
        edge_2 = edge_func_2(edge_1.to_rank)
        rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
        return bd.SimpleBoundary(
            boundary_type=boundary_type,
            from_rank=rank,
            to_rank=edge_2.to_rank,
            n_clockwise_rotations=rotations,
        )


def on_tile_left(subtile_index: Tuple[int, int]) -> bool:
    return subtile_index[1] == 0


def on_tile_right(subtile_index: Tuple[int, int], layout: Tuple[int, int]) -> bool:
    return subtile_index[1] == layout[1] - 1


def on_tile_top(subtile_index: Tuple[int, int], layout: Tuple[int, int]) -> bool:
    return subtile_index[0] == layout[0] - 1


def on_tile_bottom(subtile_index: Tuple[int, int]) -> bool:
    return subtile_index[0] == 0


def rotate_subtile_rank(
    rank: int, layout: Tuple[int, int], n_clockwise_rotations: int
) -> int:
    """Returns the rank position where this rank would be if you rotated the
    tile n_clockwise_rotations times.
    """
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
    """Returns the rank position where this rank would be if you transposed
    the tile.
    """
    return transform_subtile_rank(np.transpose, rank, layout)


def fliplr_subtile_rank(rank, layout):
    """Returns the rank position where this rank would be if you flipped the
    tile along a vertical axis
    """
    return transform_subtile_rank(np.fliplr, rank, layout)


def flipud_subtile_rank(rank, layout):
    """Returns the rank position where this rank would be if you flipped the
    tile along a horizontal axis
    """
    return transform_subtile_rank(np.flipud, rank, layout)


def transform_subtile_rank(
    transform_func: Callable[[np.ndarray], np.ndarray],
    rank: int,
    layout: Tuple[int, int],
):
    """Returns the rank position where this rank would be if you performed
    a transformation on the tile which strictly moves ranks.
    """
    total_ranks = layout[0] * layout[1]
    rank_array = np.arange(total_ranks).reshape(layout)
    transformed_rank_array = transform_func(rank_array)
    return rank_array[np.where(transformed_rank_array == rank)][0]


def subtile_index(
    rank: int, ranks_per_tile: int, layout: Tuple[int, int]
) -> Tuple[int, int]:
    within_tile_rank = rank % ranks_per_tile
    j = within_tile_rank // layout[1]
    i = within_tile_rank % layout[1]
    return j, i


def is_even(value: [int, float]) -> bool:
    return value % 2 == 0


def tile_extent_from_rank_metadata(
    dims: Iterable[str], rank_extent: Iterable[int], layout: Tuple[int, int]
) -> Tuple[int]:
    """
    Returns the extent of a tile given data about a single rank, and the tile
    layout.

    Args:
        dims: dimension names
        rank_extent: the extent of one rank
        layout: the (y, x) number of ranks along each tile axis

    Returns:
        tile_extent: the extent of one tile
    """
    layout_factors = np.asarray(
        utils.list_by_dims(dims, layout, non_horizontal_value=1)
    )
    return extent_from_metadata(dims, rank_extent, layout_factors)


def rank_extent_from_tile_metadata(
    dims: Iterable[str], tile_extent: Iterable[int], layout: Tuple[int, int]
) -> Tuple[int, ...]:
    """
    Returns the extent of a rank given data about a tile, and the tile
    layout.

    Args:
        dims: dimension names
        rank_extent: the extent of a tile
        layout: the (y, x) number of ranks along each tile axis

    Returns:
        rank_extent: the extent of one rank
    """
    layout_factors = 1 / np.asarray(
        utils.list_by_dims(dims, layout, non_horizontal_value=1)
    )
    return extent_from_metadata(dims, tile_extent, layout_factors)


def extent_from_metadata(
    dims: Tuple[str, ...], extent: Tuple[int, ...], layout_factors: np.ndarray
) -> Tuple[int, ...]:
    return_extents = []
    for dim, rank_extent, layout_factor in zip(dims, extent, layout_factors):
        if dim in constants.INTERFACE_DIMS:
            add_extent = -1
        else:
            add_extent = 0
        tile_extent = (rank_extent + add_extent) * layout_factor - add_extent
        return_extents.append(int(tile_extent))  # layout_factor is float, need to cast
    return tuple(return_extents)


@dataclasses.dataclass
class _IndexData1D:
    dim: str
    extent: int
    i_subtile: int
    n_ranks: int

    @property
    def base_extent(self):
        return self.extent - self.extent_minus_gridcell_count

    @property
    def extent_minus_gridcell_count(self):
        if self.dim in constants.INTERFACE_DIMS:
            return 1
        else:
            return 0

    @property
    def is_end_index(self):
        return self.i_subtile == self.n_ranks - 1


def _index_generator(dims, tile_extent, subtile_index, horizontal_layout):
    subtile_extent = rank_extent_from_tile_metadata(
        dims, tile_extent, horizontal_layout
    )
    quantity_layout = utils.list_by_dims(
        dims, horizontal_layout, non_horizontal_value=1
    )
    quantity_subtile_index = utils.list_by_dims(
        dims, subtile_index, non_horizontal_value=0
    )
    for dim, extent, i_subtile, n_ranks in zip(
        dims, subtile_extent, quantity_subtile_index, quantity_layout
    ):
        yield _IndexData1D(dim, extent, i_subtile, n_ranks)


def subtile_slice(
    dims: Iterable[str],
    tile_extent: Iterable[int],
    layout: Tuple[int, int],
    subtile_index: Tuple[int, int],
    overlap: bool = False,
) -> Tuple[slice, ...]:
    """
    Returns the slice of data within a tile's computational domain belonging
    to a single rank.

    Args:
        dims: dimension names for each axis
        tile_extent: size of the tile's computational domain
        layout: the (y, x) number of ranks along each tile axis
        subtile_index: the (y, x) position of the rank on the tile
        overlap: whether to assign regions which belong to multiple ranks
            to both ranks, or only to the higher rank (default)
    """
    return_list = []
    # discard last index for interface variables, unless you're the last rank
    # done so that only one rank is responsible for the shared interface point
    for index in _index_generator(dims, tile_extent, subtile_index, layout):
        start = index.i_subtile * index.base_extent
        if index.is_end_index or overlap:
            end = start + index.extent
        else:
            end = start + index.base_extent
        return_list.append(slice(start, end))
    return tuple(return_list)
