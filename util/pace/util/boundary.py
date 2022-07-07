import dataclasses
from typing import Tuple

from pace.util.halo_data_transformer import QuantityHaloSpec

from ._boundary_utils import get_boundary_slice
from .quantity import Quantity


@dataclasses.dataclass
class Boundary:
    """Maps part of a subtile domain to another rank which shares halo points."""

    from_rank: int
    to_rank: int
    n_clockwise_rotations: int
    """
    number of clockwise rotations data undergoes if it moves from the from_rank
    to the to_rank. The same as the number of clockwise rotations to get from the
    orientation of the axes in from_rank to the orientation of the axes in to_rank.
    """

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

    def send_slice(self, specification: QuantityHaloSpec) -> Tuple[slice]:
        """Return the index slices which shoud be sent at this boundary.

        Args:
            specification: data specifications for the halo. Including shape
            and number of halo points.

        Returns:
            A tuple of slices (one per dimensions)
        """
        return self._slice(specification, interior=True)

    def recv_slice(self, specification: QuantityHaloSpec) -> Tuple[slice]:
        """Return the index slices which should be received at this boundary.

        Args:
            quantity: quantity for which to return slices
            n_points: the width of boundary to include

        Returns:
            A tuple of slices (one per dimensions)
        """
        return self._slice(specification, interior=False)

    def _slice(self, specification: QuantityHaloSpec, interior: bool) -> Tuple[slice]:
        """Returns a tuple of slices (one per dimensions) indexing the data to be exchange.

        Args:
            specification: memory information on this halo, including halo size

        Return:
            A tuple of slices (one per dimensions)
        """
        raise NotImplementedError()

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

    boundary_type: int

    def _view(self, quantity: Quantity, n_points: int, interior: bool):
        boundary_slice = get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.data.shape,
            self.boundary_type,
            n_points,
            interior,
        )
        return quantity.data[tuple(boundary_slice)]

    def _slice(self, specification: QuantityHaloSpec, interior: bool) -> Tuple[slice]:
        return get_boundary_slice(
            specification.dims,
            specification.origin,
            specification.extent,
            specification.shape,
            self.boundary_type,
            specification.n_points,
            interior,
        )
