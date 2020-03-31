import dataclasses
import functools
from .quantity import Quantity
from . import constants


@dataclasses.dataclass
class Boundary:
    """Maps part of a subtile domain to another rank which shares halo points"""

    from_rank: int
    to_rank: int
    n_clockwise_rotations: int
    """
    number of clockwise rotations data undergoes if it moves from the from_rank
    to the to_rank. The same as the number of clockwise rotations to get from the
    orientation of the axes in from_rank to the orientation of the axes in to_rank.
    """

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

    def send_view(self, quantity: Quantity, n_points: int):
        """Return a sliced view of points which should be sent at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
        """
        return self._view(quantity, n_points, interior=True)

    def recv_view(self, quantity: Quantity, n_points: int):
        """Return a sliced view of points which should be recieved at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
        """
        return self._view(quantity, n_points, interior=False)

    def _view(self, quantity: Quantity, n_points: int, interior: bool):
        """Return a sliced view of points in the given quantity at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
            interior: if True, give points inside the computational domain (default),
                otherwise give points in the halo
        """
        raise NotImplementedError()


@dataclasses.dataclass
class SimpleBoundary(Boundary):
    """A boundary representing an edge or corner of a subtile."""

    boundary_type: str

    def _view(self, quantity: Quantity, n_points: int, interior: bool):
        boundary_slice = _get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            self.boundary_type,
            n_points,
            interior,
        )
        return quantity.data[tuple(boundary_slice)]


@functools.lru_cache(maxsize=None)
def _get_boundary_slice(dims, origin, extent, boundary_type, n_halo, interior):
    if boundary_type in constants.EDGE_BOUNDARY_TYPES:
        dim_to_starts = DIM_TO_START_EDGE
        dim_to_ends = DIM_TO_END_EDGE
    elif boundary_type in constants.CORNER_BOUNDARY_TYPES:
        dim_to_starts = DIM_TO_START_CORNERS
        dim_to_ends = DIM_TO_END_CORNERS
    else:
        raise ValueError(
            f"invalid boundary type {boundary_type}, "
            f"must be one of {constants.BOUNDARY_TYPES}"
        )
    boundary_slice = []
    for dim, origin_1d, extent_1d in zip(dims, origin, extent):
        if dim in constants.INTERFACE_DIMS:
            n_overlap = 1
        else:
            n_overlap = 0
        n_points = n_halo
        if dim not in constants.HORIZONTAL_DIMS:
            boundary_slice.append(slice(origin_1d, origin_1d + extent_1d))
        elif boundary_type in dim_to_starts[dim]:
            edge_index = origin_1d
            if interior:
                edge_index += n_overlap
                boundary_slice.append(slice(edge_index, edge_index + n_points))
            else:
                boundary_slice.append(slice(edge_index - n_points, edge_index))
        elif boundary_type in dim_to_ends[dim]:
            edge_index = origin_1d + extent_1d
            if interior:
                edge_index -= n_overlap
                boundary_slice.append(slice(edge_index - n_points, edge_index))
            else:
                boundary_slice.append(slice(edge_index, edge_index + n_points))
        else:
            boundary_slice.append(slice(origin_1d, origin_1d + extent_1d))
    return tuple(boundary_slice)


DIM_TO_START_EDGE = {
    constants.X_DIM: (constants.WEST,),
    constants.X_INTERFACE_DIM: (constants.WEST,),
    constants.Y_DIM: (constants.SOUTH,),
    constants.Y_INTERFACE_DIM: (constants.SOUTH,),
}

DIM_TO_END_EDGE = {
    constants.X_DIM: (constants.EAST,),
    constants.X_INTERFACE_DIM: (constants.EAST,),
    constants.Y_DIM: (constants.NORTH,),
    constants.Y_INTERFACE_DIM: (constants.NORTH,),
}


DIM_TO_START_CORNERS = {
    constants.X_DIM: (constants.NORTHWEST, constants.SOUTHWEST),
    constants.X_INTERFACE_DIM: (constants.NORTHWEST, constants.SOUTHWEST),
    constants.Y_DIM: (constants.SOUTHWEST, constants.SOUTHEAST),
    constants.Y_INTERFACE_DIM: (constants.SOUTHWEST, constants.SOUTHEAST),
}

DIM_TO_END_CORNERS = {
    constants.X_DIM: (constants.NORTHEAST, constants.SOUTHEAST),
    constants.X_INTERFACE_DIM: (constants.NORTHEAST, constants.SOUTHEAST),
    constants.Y_DIM: (constants.NORTHWEST, constants.NORTHEAST),
    constants.Y_INTERFACE_DIM: (constants.NORTHWEST, constants.NORTHEAST),
}
