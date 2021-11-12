"""
fill_scalar_corners

The fill_corners routines put data into tile corners to make stencil operations more
convenient.

The corners themselves do not map on to anything meaningful. At, say, the southwest
corner of a tile, the south edge and west edge correspond to different cube faces.
However, there is no cube face across the southwest corner - it's the edge between
the two adjacent tiles.

(Figures as below depict grid cells near the edge of a tile domain. All symbols except
for C represent halo data.)

W: west neighbor tile
S: south neighbor tile
C: compute domain
?: corner data, with no meaningful tile

WWWCCC
WWWCCC
WWWCCC
???SSS
???SSS
???SSS

However, when applying stencils to the data, it is useful to make use of these corners.
Take for example a stencil along the x-axis. This stencil might be applied in the
south halo, as below:

X: 3-by-1 stencil being moved along the domain

WWWCCC
WWWCCC
WWWCCC
???SSS
???XXX
???SSS

Something interesting happens when the stencil tries to compute at the outermost
point. It wants to grab data from the western tile face, across the edge boundary
separating those two tiles!

X: 3-by-1 stencil being applied across tile edge

WWWCCC
WWWCCC
WXWCCC
???SSS
???XXS
???SSS

In order to do this, the west tile data is copied into the corner before the 3-by-1
stencil is called. Numbering the west-tile points, the data is copied like so:

123CCC
456CCC
789CCC
369SSS
258SSS
147SSS

This allows the 3-by-1 stencil to be applied naturally. Note the new X position within
the corner is being applied on the same number as the point across the edge
shown earlier.

123CCC
456CCC
789CCC
369SSS
25XXXS
147SSS

"""
from typing import Sequence

from typing_extensions import Literal

from . import constants
from .partitioner import TilePartitioner
from .quantity import Quantity


def fill_scalar_corners(
    quantity: Quantity,
    direction: Literal["x", "y"],
    tile_partitioner: TilePartitioner,
    rank: int,
    n_halo: int,
):
    """
    At the corners of tile faces, copy data from halo edges into halo corners to allow
    stencils to be translated along those edges in a computationally-relevant way.

    The quantity is modified in-place.

    Args:
        quantity: the quantity to modify, whose first two dimensions must be along
            the x and y directions, respectively
        direction: the direction along which we want to enable stencils to compute.
            For example, calling with "x" would allow a stencil with length > 1 along
            the x-direction to be convolved with Quantity. Note it is not possible
            to use corner filling to convolve with stencils having length > 1 along
            both x and y dimensions.
        tile_partitioner: object to determine tile positions of ranks
        rank: rank on which the quantity exists
        n_halo: number of halo points to fill
    """
    if quantity.dims[0] not in constants.X_DIMS:
        raise ValueError("first dimension must be in x-direction")
    elif quantity.dims[1] not in constants.Y_DIMS:
        raise ValueError("second dimension must be in y-direction")
    if direction not in ("x", "y"):
        raise TypeError(f"direction must be one of 'x' or 'y', received {direction}")
    on_north = tile_partitioner.on_tile_top(rank)
    on_south = tile_partitioner.on_tile_bottom(rank)
    on_east = tile_partitioner.on_tile_right(rank)
    on_west = tile_partitioner.on_tile_left(rank)
    # for interface variables, the edge exists on both sides of the corner, so
    # we need to copy data *after* that edge. shift=1 does this, shift=0
    # does nothing
    shift = _shared_edge_points(direction, quantity.dims)
    if on_south and on_west:
        if direction == "y":
            quantity.view.southwest[-n_halo:0, -n_halo:0] = quantity.np.rot90(
                quantity.view.southwest[shift : n_halo + shift, -n_halo:0], k=-1
            )
        else:
            quantity.view.southwest[-n_halo:0, -n_halo:0] = quantity.np.rot90(
                quantity.view.southwest[-n_halo:0, shift : n_halo + shift], k=1
            )
    if on_north and on_west:
        if direction == "y":
            quantity.view.northwest[-n_halo:0, 0:n_halo] = quantity.np.rot90(
                quantity.view.northwest[shift : n_halo + shift, 0:n_halo], k=1
            )
        else:
            quantity.view.northwest[-n_halo:0, 0:n_halo] = quantity.np.rot90(
                quantity.view.northwest[-n_halo:0, -n_halo - shift : -shift], k=-1
            )
    if on_south and on_east:
        if direction == "y":
            quantity.view.southeast[0:n_halo, -n_halo:0] = quantity.np.rot90(
                quantity.view.southeast[-n_halo - shift : -shift, -n_halo:0], k=1
            )
        else:
            quantity.view.southeast[0:n_halo, -n_halo:0] = quantity.np.rot90(
                quantity.view.southeast[0:n_halo, shift : n_halo + shift], k=-1
            )
    if on_north and on_east:
        if direction == "y":
            quantity.view.northeast[0:n_halo, 0:n_halo] = quantity.np.rot90(
                quantity.view.northeast[-n_halo - shift : -shift, 0:n_halo], k=-1
            )
        else:
            quantity.view.northeast[0:n_halo, 0:n_halo] = quantity.np.rot90(
                quantity.view.northeast[0:n_halo, -n_halo - shift : -shift], k=1
            )


def _shared_edge_points(fill_direction: Literal["x", "y"], dims: Sequence[str]):
    """
    Returns the number of edge points shared by adjacent tiles along the direction
    that is going to be copied when a given fill_direction is passed to a corner
    filling routine.

    Note that the direction along which data is copied is the opposite of the
    fill_direction, as the fill direction corresponds to the direction along which
    stencils are going to be subsequently called.
    """
    if fill_direction == "y":
        # for interface variables, the edge exists on both sides of the corner, so
        # we need to copy data *after* that edge. shift=1 does this, shift=0
        # does nothing
        shift = int(dims[0] in constants.INTERFACE_DIMS)
    else:
        shift = int(dims[1] in constants.INTERFACE_DIMS)
    return shift
