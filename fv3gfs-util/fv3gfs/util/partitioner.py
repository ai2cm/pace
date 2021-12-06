import abc
import copy
import dataclasses
import functools
from typing import Callable, Optional, Sequence, Tuple, Union, cast

import numpy as np

from . import boundary as bd
from . import constants, utils
from .constants import (
    EAST,
    NORTH,
    NORTHEAST,
    NORTHWEST,
    SOUTH,
    SOUTHEAST,
    SOUTHWEST,
    WEST,
)
from .quantity import QuantityMetadata


BOUNDARY_CACHE_SIZE = None


__all__ = ["TilePartitioner", "CubedSpherePartitioner", "get_tile_index"]


def get_tile_index(rank: int, total_ranks: int) -> int:
    """
    Returns the zero-indexed tile number, given a rank and total number of ranks.
    """
    if total_ranks % 6 != 0:
        raise ValueError(f"total_ranks {total_ranks} is not evenly divisible by 6")
    ranks_per_tile = total_ranks // 6
    return rank // ranks_per_tile


def get_tile_number(tile_rank: int, total_ranks: int) -> int:
    """Deprecated: use get_tile_index.

    Returns the tile number for a given rank and total number of ranks.
    """
    FutureWarning(
        "get_tile_number will be removed in a later version, "
        "use get_tile_index(rank, total_ranks) + 1 instead"
    )
    if total_ranks % 6 != 0:
        raise ValueError(f"total_ranks {total_ranks} is not evenly divisible by 6")
    ranks_per_tile = total_ranks // 6
    return tile_rank // ranks_per_tile + 1


class Partitioner(abc.ABC):
    @abc.abstractmethod
    def global_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full tile representation
        """
        pass

    @abc.abstractmethod
    def subtile_slice(
        self,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        rank: int,
        overlap: bool = False,
    ) -> Tuple[Union[int, slice], ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            rank: the rank of the process
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the slice of the global compute domain corresponding
                to the subtile compute domain
        """
        pass

    @abc.abstractmethod
    def subtile_extent(
        self,
        global_metadata: QuantityMetadata,
        rank: int,
        overlap: bool = False,
        is_cube_metadata: bool = False,
    ) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions."""
        pass

    @abc.abstractproperty
    def total_ranks(self) -> int:
        pass


class TilePartitioner(Partitioner):
    def __init__(
        self,
        layout: Tuple[int, int],
        edge_interior_ratio: float = 1.0,
    ):
        """Create an object for fv3gfs tile decomposition."""
        self.layout = layout
        self.edge_interior_ratio = edge_interior_ratio

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

    def global_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full tile representation
        """
        return tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_extent(
        self,
        global_metadata: QuantityMetadata,
        rank: int,
        overlap: bool = False,
        is_cube_metadata: bool = False,
    ) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions."""
        if is_cube_metadata:
            cube_metadata_offset = 1
        else:
            cube_metadata_offset = 0

        tile_decomposition = subtile_extents_from_tile_metadata(
            global_metadata.dims[cube_metadata_offset:],
            global_metadata.extent[cube_metadata_offset:],
            self.layout,
        )
        num_decomposed_dims = int(len(tile_decomposition) / 2)
        horizontal_dim_index = 0
        return_extent = []
        for num_dim, dim in enumerate(global_metadata.dims[cube_metadata_offset:]):
            edge_index_offset = 0
            is_end_index = False
            if dim in constants.HORIZONTAL_DIMS:
                if dim in constants.Y_DIMS:
                    is_end_index = (
                        rank
                        >= (self.layout[horizontal_dim_index] - 1)
                        * self.layout[1 - horizontal_dim_index]
                    )
                    rank_is_on_axis_edge = (
                        rank < self.layout[horizontal_dim_index] or is_end_index
                    )
                elif dim in constants.X_DIMS:
                    position_on_axis = rank % self.layout[horizontal_dim_index]
                    is_end_index = (
                        position_on_axis == self.layout[horizontal_dim_index] - 1
                    )
                    rank_is_on_axis_edge = position_on_axis == 0 or is_end_index
                if rank_is_on_axis_edge:
                    edge_index_offset = num_decomposed_dims
                horizontal_dim_index = horizontal_dim_index + 1

            return_extent.append(
                _interface_overlap_extent(
                    dim,
                    is_end_index,
                    tile_decomposition[num_dim + edge_index_offset],
                    overlap,
                )
            )

        return tuple(return_extent)

    def subtile_slice(
        self,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        rank: int,
        overlap: bool = False,
    ) -> Tuple[slice, ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            rank: the rank of the process
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the slice of the global compute domain corresponding
                to the subtile compute domain
        """
        return subtile_slice(
            global_dims,
            global_extent=global_extent,
            layout=self.layout,
            subtile_index=self.subtile_index(rank),
            edge_interior_ratio=self.edge_interior_ratio,
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

    def boundary(self, boundary_type: int, rank: int) -> Optional[bd.SimpleBoundary]:
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
    def _cached_boundary(
        self, boundary_type: int, rank: int
    ) -> Optional[bd.SimpleBoundary]:
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

    def _top_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        return _get_corner(constants.NORTHWEST, rank, self._left_edge, self._top_edge)

    def _top_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        return _get_corner(constants.NORTHEAST, rank, self._right_edge, self._top_edge)

    def _bottom_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        return _get_corner(
            constants.SOUTHWEST, rank, self._left_edge, self._bottom_edge
        )

    def _bottom_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
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


class CubedSpherePartitioner(Partitioner):
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

    def tile_index(self, rank: int) -> int:
        """Returns the tile index of a given rank"""
        return get_tile_index(rank, self.total_ranks)

    def tile_root_rank(self, rank: int) -> int:
        """Returns the lowest rank on the same tile as a given rank."""
        return self.tile.total_ranks * (rank // self.tile.total_ranks)

    @property
    def layout(self) -> Tuple[int, int]:
        return self.tile.layout

    @property
    def total_ranks(self) -> int:
        """the number of ranks on the cubed sphere"""
        return 6 * self.tile.total_ranks

    def boundary(self, boundary_type: int, rank: int) -> Optional[bd.SimpleBoundary]:
        """Returns a boundary of the requested type for a given rank, or None.

        On tile corners, the boundary across that corner does not exist.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        boundary = copy.copy(self._cached_boundary(boundary_type, rank))
        return boundary

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def _cached_boundary(
        self, boundary_type: int, rank: int
    ) -> Optional[bd.SimpleBoundary]:
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
                to_root_rank = self.tile_root_rank(rank - 2 * self.tile.total_ranks)
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                to_rank = to_root_rank + to_tile_rank
                rotations = 1
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.WEST,
                    from_rank=rank,
                    to_rank=to_rank,
                    n_clockwise_rotations=rotations,
                )
            else:
                boundary = cast(bd.SimpleBoundary, self.tile.boundary(WEST, rank=rank))
                boundary.to_rank -= self.tile.total_ranks
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(WEST, rank=rank))
        return boundary

    def _right_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_right(rank):
            if not is_even(self.tile_index(rank)):
                to_root_rank = self.tile_root_rank(rank + 2 * self.tile.total_ranks)
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.EAST,
                    from_rank=rank,
                    to_rank=to_root_rank + to_tile_rank,
                    n_clockwise_rotations=1,
                )
            else:
                boundary = cast(bd.SimpleBoundary, self.tile.boundary(EAST, rank=rank))
                boundary.to_rank += self.tile.total_ranks
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(EAST, rank=rank))
        return boundary

    def _top_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_top(rank):
            if is_even(self.tile_index(rank)):
                to_root_rank = (self.tile_index(rank) + 2) * self.tile.total_ranks
                tile_rank = rank % self.tile.total_ranks
                to_tile_rank = self.tile.fliplr_rank(
                    self.tile.rotate_rank(tile_rank, 1)
                )
                boundary = bd.SimpleBoundary(
                    boundary_type=constants.NORTH,
                    from_rank=rank,
                    to_rank=to_root_rank + to_tile_rank,
                    n_clockwise_rotations=3,
                )
            else:
                boundary = cast(bd.SimpleBoundary, self.tile.boundary(NORTH, rank))
                boundary.to_rank += self.tile.total_ranks
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(NORTH, rank=rank))
        return boundary

    def _bottom_edge(self, rank: int) -> bd.SimpleBoundary:
        self._ensure_square_layout()
        if self.tile.on_tile_bottom(rank) and not is_even(self.tile_index(rank)):
            to_root_rank = (self.tile_index(rank) - 2) * self.tile.total_ranks
            tile_rank = rank % self.tile.total_ranks
            to_tile_rank = self.tile.fliplr_rank(self.tile.rotate_rank(tile_rank, 1))
            boundary = bd.SimpleBoundary(
                boundary_type=constants.SOUTH,
                from_rank=rank,
                to_rank=to_root_rank + to_tile_rank,
                n_clockwise_rotations=3,
            )
        else:
            boundary = cast(bd.SimpleBoundary, self.tile.boundary(SOUTH, rank=rank))
            if self.tile.on_tile_bottom(rank):
                boundary.to_rank -= self.tile.total_ranks
        return boundary

    def _top_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
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

    def _top_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
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

    def _bottom_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
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

    def _bottom_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
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
    ) -> bd.SimpleBoundary:
        edge_1 = edge_func_1(rank)
        edge_2 = edge_func_2(edge_1.to_rank)
        rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
        return bd.SimpleBoundary(
            boundary_type=boundary_type,
            from_rank=rank,
            to_rank=edge_2.to_rank,
            n_clockwise_rotations=rotations,
        )

    def global_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full cube representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full cube representation
        """
        return (6,) + tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_extent(
        self,
        cube_metadata: QuantityMetadata,
        rank: int,
        overlap: bool = False,
        is_cube_metadata: bool = True,
    ) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions."""
        if cube_metadata.dims[0] != constants.TILE_DIM:
            raise NotImplementedError(
                "currently only supports tile dimension {constants.TILE_DIM} as the "
                "first dimension, got dims {cube_metadata.dims}"
            )
        if not is_cube_metadata:
            is_cube_metadata = True

        return self.tile.subtile_extent(cube_metadata, rank, overlap, is_cube_metadata)

    def subtile_slice(
        self,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        rank: int,
        overlap: bool = False,
    ) -> Tuple[Union[int, slice], ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            rank: the rank of the process
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the tuple slice of the global compute domain corresponding
                to the subtile compute domain
        """
        if global_dims[0] != constants.TILE_DIM:
            raise NotImplementedError(
                "currently only supports tile dimension {constants.TILE_DIM} as the "
                "first dimension, got dims {cube_metadata.dims}"
            )
        i_tile = self.tile_index(rank)
        return (i_tile,) + self.tile.subtile_slice(
            global_dims=global_dims[1:],
            global_extent=global_extent[1:],
            rank=rank,
            overlap=overlap,
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


def is_even(value: Union[int, float]) -> bool:
    return value % 2 == 0


def tile_extent_from_rank_metadata(
    dims: Sequence[str],
    rank_extent: Sequence[int],
    layout: Tuple[int, int],
    edge_interior_ratio: float = 1.0,
) -> Tuple[int, ...]:
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
    if edge_interior_ratio < 1.0:
        print(f"{edge_interior_ratio}", flush=True)
        raise NotImplementedError("Only equal sized subdomains are supported.")
    layout_factors = np.asarray(
        utils.list_by_dims(dims, layout, non_horizontal_value=1)
    )
    return extent_from_metadata(dims, rank_extent, layout_factors)


def listArgsToTupleArgs(function):
    def wrapper(*args):
        """Wrapper ensures hashable function arguments (e.g. lists become tuples)"""
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result

    return wrapper


@listArgsToTupleArgs
@functools.lru_cache(maxsize=64)
def subtile_extents_from_tile_metadata(
    dims: Sequence[str],
    tile_extent: Sequence[int],
    layout: Tuple[int, int],
    edge_interior_ratio: float = 1.0,
) -> Tuple[int, ...]:
    """
    Returns the extent of a given rank given data about a tile, and the tile
    layout.

    Args:
        dims: dimension names
        tile_extent: the extent of a tile
        layout: the (y, x) number of ranks along each tile axis
        edge_interior_ratio (optional): ratio between interior and boundary tile sizes.
            Assumes full integer divisibility for both interior
            and boundary tile sizes.

    Returns:
        subtile_extents: the extents of first all interior tiles, then all edge tiles along all dimensions.
    """

    def _valid_edge_tile_sizes(dim_extent: int, subtile_count: int, start: int):
        """ "Returns a list of valid edge tile sizes, counting down from the starting edge size to the smallest possible one
        that lets the interior tile sizes still be an integer. After that, it counts up from the starting edge size.
        """
        bottom = 1
        top = int((dim_extent - subtile_count + 2) / 2) + 1
        unsorted_valid_sizes = range(bottom, top)
        valid_sizes = []

        index = start
        offset = 0
        factor = -1

        for i in range(len(unsorted_valid_sizes) + 1):
            index = start + factor * offset
            if index in unsorted_valid_sizes and index not in valid_sizes:
                valid_sizes.append(index)
            offset = offset + 1
            if index == 1:
                offset = 0
                factor = 1
        return valid_sizes

    layout_factors = np.asarray(
        utils.list_by_dims(dims, layout, non_horizontal_value=1)
    )

    return_extents = []
    edge_extents = []
    for dim, subtile_count, dim_extent in zip(dims, layout_factors, tile_extent):
        dim_edge_interior_ratio = edge_interior_ratio
        if dim in constants.INTERFACE_DIMS:
            dim_extent = dim_extent - 1
        if (not subtile_count % 2) and dim_extent % 2:
            raise ValueError(
                f"Cannot find valid decomposition for odd ({dim_extent}) gridpoints along an even count ({subtile_count}) of ranks."
            )
        if (
            subtile_count >= 3 and dim in constants.HORIZONTAL_DIMS
        ):  # only do shrinked edges in x,y and if there is interior
            edge_subtile_size = round(
                dim_extent / (2 + (subtile_count - 2) / dim_edge_interior_ratio)
            )

            # searching of a valid integer pair for edge and interior tile sizes that add up to the entire dimension extent.
            found = False
            for edge_size in _valid_edge_tile_sizes(
                dim_extent, subtile_count, edge_subtile_size
            ):
                dim_edge_interior_ratio = edge_size / (
                    (dim_extent - 2 * edge_size) / (subtile_count - 2)
                )
                if (
                    edge_size * 2
                    + (subtile_count - 2) * int(edge_size / dim_edge_interior_ratio)
                    == dim_extent
                ):
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"No valid subdomain assignment found for dimension {dim} with {dim_extent} gridpoints along {subtile_count} ranks."
                )
            return_extents.append(int(edge_size / dim_edge_interior_ratio))
            edge_extents.append(edge_size)
        else:
            edge_size = int(dim_extent / subtile_count)
            return_extents.append(edge_size)
            edge_extents.append(edge_size)
    return_extents.extend(edge_extents)
    return tuple(return_extents)


def extent_from_metadata(
    dims: Sequence[str], extent: Sequence[int], layout_factors: np.ndarray
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


def subtile_slice(
    dims: Sequence[str],
    global_extent: Sequence[int],
    layout: Tuple[int, int],
    subtile_index: Tuple[int, int],
    edge_interior_ratio: float = 1.0,
    overlap: bool = False,
) -> Tuple[slice, ...]:
    """
    Returns the slice of data within a tile's computational domain belonging
    to a single rank.

    Args:
        dims: dimension names for each axis
        global_extent: size of the tile or cube's computational domain
        layout: the (y, x) number of ranks along each tile axis
        subtile_index: the (y, x) position of the rank on the tile
        edge_interior_ratio: ratio between interior and boundary tile sizes.
            Assumes full integer divisibility for both interior
            and boundary tile sizes.
        overlap: whether to assign regions which belong to multiple ranks
            to both ranks, or only to the higher rank (default)
    """

    return_list = []
    subtile_extent = subtile_extents_from_tile_metadata(
        dims, global_extent, layout, edge_interior_ratio
    )
    # discard last index for interface variables, unless you're the last rank
    # done so that only one rank is responsible for the shared interface point
    num_decomposed_dims = int(len(subtile_extent) / 2)
    horizontal_dim_index = 0

    for num_dim, (dim, dim_extent) in enumerate(zip(dims, global_extent)):
        if dim in constants.HORIZONTAL_DIMS:
            if (
                subtile_index[horizontal_dim_index] == 0
                or layout[horizontal_dim_index] < 3
            ):
                # this is technically not the edge tile size, but does not matter as subtile_index for that dim is 0.
                # alternatively equal sized tiles go here, which can take both subdomain sizes.
                start = subtile_index[horizontal_dim_index] * subtile_extent[num_dim]
            else:
                start = (subtile_index[horizontal_dim_index] - 1) * subtile_extent[
                    num_dim
                ] + subtile_extent[num_dim + num_decomposed_dims]

            is_end_index = subtile_index[horizontal_dim_index] == (
                layout[horizontal_dim_index] - 1
            )
            if layout[horizontal_dim_index] < 3 or (
                subtile_index[horizontal_dim_index] == 0 or is_end_index
            ):
                end = start + _interface_overlap_extent(
                    dim,
                    is_end_index,
                    subtile_extent[num_dim + num_decomposed_dims],
                    overlap,
                )
            else:
                end = start + _interface_overlap_extent(
                    dim, False, subtile_extent[num_dim], overlap
                )
            horizontal_dim_index = horizontal_dim_index + 1
        else:
            start = 0
            end = dim_extent

        return_list.append(slice(int(start), int(end)))
    return tuple(return_list)


def _interface_overlap_extent(dim: str, is_end_index: bool, extent: int, overlap: bool):
    if dim in constants.INTERFACE_DIMS and (is_end_index or overlap):
        extent_plus_overlap_gridcell_count = 1
    else:
        extent_plus_overlap_gridcell_count = 0
    return extent + extent_plus_overlap_gridcell_count
