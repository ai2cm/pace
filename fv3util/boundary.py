import dataclasses
import functools
from .quantity import Quantity
from . import constants


@dataclasses.dataclass
class Boundary:
    """Maps part of a subtile domain to another rank which shares ghost cells"""
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
            quantity.dims, quantity.origin, quantity.extent,
            self.boundary_type, n_points, interior
        )
        return quantity.data[tuple(boundary_slice)]


@functools.lru_cache(maxsize=None)
def _get_boundary_slice(dims, origin, extent, boundary_type, n_points, interior):
    if boundary_type in constants.EDGE_BOUNDARY_TYPES:
        dim_to_starts = DIM_TO_START_EDGE
        dim_to_ends = DIM_TO_END_EDGE
    elif boundary_type in constants.CORNER_BOUNDARY_TYPES:
        dim_to_starts = DIM_TO_START_CORNERS
        dim_to_ends = DIM_TO_END_CORNERS
    else:
        raise ValueError(
            f'invalid boundary type {boundary_type}, '
            f'must be one of {constants.BOUNDARY_TYPES}'
        )
    boundary_slice = []
    for dim, origin_1d, extent_1d in zip(dims, origin, extent):
        if dim not in constants.HORIZONTAL_DIMS:
            boundary_slice.append(slice(None, None))
        elif boundary_type in dim_to_starts[dim]:
            edge_index = origin_1d
            if interior:
                boundary_slice.append(slice(edge_index, edge_index + n_points))
            else:
                boundary_slice.append(slice(edge_index - n_points, edge_index))
        elif boundary_type in dim_to_ends[dim]:
            edge_index = origin_1d + extent_1d
            if interior:
                boundary_slice.append(slice(edge_index - n_points, edge_index))
            else:
                boundary_slice.append(slice(edge_index, edge_index + n_points))
        else:
            boundary_slice.append(slice(origin_1d, origin_1d + extent_1d))
    return tuple(boundary_slice)


DIM_TO_START_EDGE = {
    constants.X_DIM: (constants.LEFT,),
    constants.X_INTERFACE_DIM: (constants.LEFT,),
    constants.Y_DIM: (constants.BOTTOM,),
    constants.Y_INTERFACE_DIM: (constants.BOTTOM,),
}

DIM_TO_END_EDGE = {
    constants.X_DIM: (constants.RIGHT,),
    constants.X_INTERFACE_DIM: (constants.RIGHT,),
    constants.Y_DIM: (constants.TOP,),
    constants.Y_INTERFACE_DIM: (constants.TOP,),
}


DIM_TO_START_CORNERS = {
    constants.X_DIM: (constants.TOP_LEFT, constants.BOTTOM_LEFT),
    constants.X_INTERFACE_DIM: (constants.TOP_LEFT, constants.BOTTOM_LEFT),
    constants.Y_DIM: (constants.BOTTOM_LEFT, constants.BOTTOM_RIGHT),
    constants.Y_INTERFACE_DIM: (constants.BOTTOM_LEFT, constants.BOTTOM_RIGHT),
}

DIM_TO_END_CORNERS = {
    constants.X_DIM: (constants.TOP_RIGHT, constants.BOTTOM_RIGHT),
    constants.X_INTERFACE_DIM: (constants.TOP_RIGHT, constants.BOTTOM_RIGHT),
    constants.Y_DIM: (constants.TOP_LEFT, constants.TOP_RIGHT),
    constants.Y_INTERFACE_DIM: (constants.TOP_LEFT, constants.TOP_RIGHT),
}
