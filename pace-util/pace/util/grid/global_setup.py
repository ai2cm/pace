import math

from pace.util.constants import PI, RADIUS

from .generation import MetricTerms
from .gnomonic import (
    _cart_to_latlon,
    _check_shapes,
    _latlon2xyz,
    _mirror_latlon,
    symm_ed,
)
from .mirror import _rot_3d


def gnomonic_grid(grid_type: int, lon, lat, np):
    """
    Apply gnomonic grid to lon and lat arrays for all tiles. Tiles must then be rotated
    and mirrored to the correct orientations before use.
    This global mesh generation is the way the Fortran code initializes the lon/lat
    grids and is reproduced here for testing purposes.

    args:
        grid_type: type of grid to apply
        lon: longitute array with dimensions [x, y]
        lat: latitude array with dimensionos [x, y]
    """
    _check_shapes(lon, lat)
    if grid_type == 0:
        global_gnomonic_ed(lon, lat, np)
    elif grid_type == 1:
        raise NotImplementedError()
    elif grid_type == 2:
        raise NotImplementedError()
    if grid_type < 3:
        symm_ed(lon, lat)
        lon[:] -= PI


# A tile global version of gnomonic_ed
# closer to the Fortran code
def global_gnomonic_ed(lon, lat, np):
    im = lon.shape[0] - 1
    alpha = np.arcsin(3 ** -0.5)
    dely = np.multiply(2.0, alpha) / float(im)
    pp = np.zeros((3, im + 1, im + 1))

    for j in range(0, im + 1):
        lon[0, j] = 0.75 * PI  # West edge
        lon[im, j] = 1.25 * PI  # East edge
        lat[0, j] = -alpha + dely * float(j)  # West edge
        lat[im, j] = lat[0, j]  # East edge

    # Get North-South edges by symmetry
    for i in range(1, im):
        lon[i, 0], lat[i, 0] = _mirror_latlon(
            lon[0, 0], lat[0, 0], lon[im, im], lat[im, im], lon[0, i], lat[0, i], np
        )
        lon[i, im] = lon[i, 0]
        lat[i, im] = -lat[i, 0]

    # set 4 corners
    pp[:, 0, 0] = _latlon2xyz(lon[0, 0], lat[0, 0], np)
    pp[:, im, 0] = _latlon2xyz(lon[im, 0], lat[im, 0], np)
    pp[:, 0, im] = _latlon2xyz(lon[0, im], lat[0, im], np)
    pp[:, im, im] = _latlon2xyz(lon[im, im], lat[im, im], np)

    # map edges on the sphere back to cube: intersection at x = -1/sqrt(3)
    i = 0
    for j in range(1, im):
        pp[:, i, j] = _latlon2xyz(lon[i, j], lat[i, j], np)
        pp[1, i, j] = -pp[1, i, j] * (3 ** -0.5) / pp[0, i, j]
        pp[2, i, j] = -pp[2, i, j] * (3 ** -0.5) / pp[0, i, j]

    j = 0
    for i in range(1, im):
        pp[:, i, j] = _latlon2xyz(lon[i, j], lat[i, j], np)
        pp[1, i, j] = -pp[1, i, j] * (3 ** -0.5) / pp[0, i, j]
        pp[2, i, j] = -pp[2, i, j] * (3 ** -0.5) / pp[0, i, j]

    pp[0, :, :] = -(3 ** -0.5)
    for j in range(1, im + 1):
        # copy y-z face of the cube along j=0
        pp[1, 1:, j] = pp[1, 1:, 0]
        # copy along i=0
        pp[2, 1:, j] = pp[2, 0, j]
    _cart_to_latlon(im + 1, pp, lon, lat, np)


# A tile global version of mirror_grid
# Closer to the Fortran code
def global_mirror_grid(
    grid_global, ng: int, npx: int, npy: int, np, right_hand_grid: bool
):
    """
    Mirrors and rotates all tiles of a lon/lat grid to the correct orientation.
    The tiles must then be partitioned onto the appropriate ranks.
    This global mesh generation is the way the Fortran code initializes the lon/lat
    grids and is reproduced here for testing purposes.
    """
    # first fix base region
    nreg = 0
    for j in range(0, math.ceil(npy / 2)):
        for i in range(0, math.ceil(npx / 2)):
            x1 = np.multiply(
                0.25,
                np.abs(grid_global[ng + i, ng + j, 0, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + j, 0, nreg])
                + np.abs(grid_global[ng + i, ng + npy - (j + 1), 0, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg]),
            )
            grid_global[ng + i, ng + j, 0, nreg] = np.copysign(
                x1, grid_global[ng + i, ng + j, 0, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + j, 0, nreg] = np.copysign(
                x1, grid_global[ng + npx - (i + 1), ng + j, 0, nreg]
            )
            grid_global[ng + i, ng + npy - (j + 1), 0, nreg] = np.copysign(
                x1, grid_global[ng + i, ng + npy - (j + 1), 0, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg] = np.copysign(
                x1, grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 0, nreg]
            )

            y1 = np.multiply(
                0.25,
                np.abs(grid_global[ng + i, ng + j, 1, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + j, 1, nreg])
                + np.abs(grid_global[ng + i, ng + npy - (j + 1), 1, nreg])
                + np.abs(grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg]),
            )

            grid_global[ng + i, ng + j, 1, nreg] = np.copysign(
                y1, grid_global[ng + i, ng + j, 1, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + j, 1, nreg] = np.copysign(
                y1, grid_global[ng + npx - (i + 1), ng + j, 1, nreg]
            )
            grid_global[ng + i, ng + npy - (j + 1), 1, nreg] = np.copysign(
                y1, grid_global[ng + i, ng + npy - (j + 1), 1, nreg]
            )
            grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg] = np.copysign(
                y1, grid_global[ng + npx - (i + 1), ng + npy - (j + 1), 1, nreg]
            )

            # force dateline/greenwich-meridion consistency
            if npx % 2 != 0:
                if i == (npx - 1) // 2:
                    grid_global[ng + i, ng + j, 0, nreg] = 0.0
                    grid_global[ng + i, ng + npy - (j + 1), 0, nreg] = 0.0

    i_mid = (npx - 1) // 2
    j_mid = (npy - 1) // 2
    for nreg in range(1, MetricTerms.N_TILES):
        for j in range(0, npy):
            x1 = grid_global[ng : ng + npx, ng + j, 0, 0]
            y1 = grid_global[ng : ng + npx, ng + j, 1, 0]
            z1 = np.add(RADIUS, np.multiply(0.0, x1))

            if nreg == 1:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3,
                    [x1, y1, z1],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
            elif nreg == 2:
                ang = -90.0
                x2, y2, z2 = _rot_3d(
                    3,
                    [x1, y1, z1],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1,
                    [x2, y2, z2],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
                # force North Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == i_mid:
                        x2[i_mid] = 0.0
                        y2[i_mid] = PI / 2.0
                    if j == j_mid:
                        x2[: i_mid + 1] = 0.0
                        x2[i_mid + 1 :] = PI
            elif nreg == 3:
                ang = -180.0
                x2, y2, z2 = _rot_3d(
                    3,
                    [x1, y1, z1],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    1,
                    [x2, y2, z2],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
                # force dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == (npy - 1) // 2:
                        x2[:] = PI
            elif nreg == 4:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    3,
                    [x1, y1, z1],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2,
                    [x2, y2, z2],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
            elif nreg == 5:
                ang = 90.0
                x2, y2, z2 = _rot_3d(
                    2,
                    [x1, y1, z1],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
                ang = 0.0
                x2, y2, z2 = _rot_3d(
                    3,
                    [x2, y2, z2],
                    ang,
                    np,
                    right_hand_grid,
                    degrees=True,
                    convert=True,
                )
                # force South Pole and dateline/Greenwich-Meridian consistency
                if npx % 2 != 0:
                    if j == i_mid:
                        x2[i_mid] = 0.0
                        y2[i_mid] = -PI / 2.0
                    if j > j_mid:
                        x2[i_mid] = 0.0
                    elif j < j_mid:
                        x2[i_mid] = PI

            grid_global[ng : ng + npx, ng + j, 0, nreg] = x2
            grid_global[ng : ng + npx, ng + j, 1, nreg] = y2

    return grid_global
