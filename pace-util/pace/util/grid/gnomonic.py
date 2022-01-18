import math

from pace.util.constants import PI


def _check_shapes(lon, lat):
    if len(lon.shape) != 2:
        raise ValueError(f"longitude must be 2D, has shape {lon.shape}")
    elif len(lat.shape) != 2:
        raise ValueError(f"latitude must be 2D, has shape {lat.shape}")
    elif lon.shape[0] != lon.shape[1]:
        raise ValueError(f"longitude must be square, has shape {lon.shape}")
    elif lat.shape[0] != lat.shape[1]:
        raise ValueError(f"latitude must be square, has shape {lat.shape}")
    elif lon.shape[0] != lat.shape[0]:
        raise ValueError(
            "longitude and latitude must have same shape, but they are "
            f"{lon.shape} and {lat.shape}"
        )


def lat_tile_east_west_edge(alpha, dely, south_north_tile_index):
    return -alpha + dely * float(south_north_tile_index)


def local_gnomonic_ed(
    lon,
    lat,
    npx,
    west_edge,
    east_edge,
    south_edge,
    north_edge,
    global_is,
    global_js,
    np,
    rank,
):
    _check_shapes(lon, lat)
    # tile_im, wedge_dict, corner_dict, global_is, global_js
    im = lon.shape[0] - 1
    alpha = np.arcsin(3 ** -0.5)
    tile_im = npx - 1
    dely = np.multiply(2.0, alpha / float(tile_im))
    halo = 3
    pp = np.zeros((3, im + 1, im + 1))
    pp_west_tile_edge = np.zeros((3, 1, im + 1))
    pp_south_tile_edge = np.zeros((3, im + 1, 1))
    lon_west_tile_edge = np.zeros((1, im + 1))
    lon_south_tile_edge = np.zeros((im + 1, 1))
    lat_west_tile_edge = np.zeros((1, im + 1))
    lat_south_tile_edge = np.zeros((im + 1, 1))
    lat_west_tile_edge_mirror = np.zeros((1, im + 1))

    lon_west = 0.75 * PI
    lon_east = 1.25 * PI
    lat_south = lat_tile_east_west_edge(alpha, dely, 0)
    lat_north = lat_tile_east_west_edge(alpha, dely, tile_im)

    start_i = 1 if west_edge else 0
    end_i = im if east_edge else im + 1
    start_j = 1 if south_edge else 0
    lon_west_tile_edge[0, :] = lon_west
    for j in range(0, im + 1):
        lat_west_tile_edge[0, j] = lat_tile_east_west_edge(
            alpha, dely, global_js - halo + j
        )
        lat_west_tile_edge_mirror[0, j] = lat_tile_east_west_edge(
            alpha, dely, global_is - halo + j
        )

    if east_edge:
        lon_south_tile_edge[im, 0] = 1.25 * PI
        lat_south_tile_edge[im, 0] = lat_tile_east_west_edge(
            alpha, dely, global_js - halo
        )

    # Get North-South edges by symmetry
    for i in range(start_i, end_i):
        edge_lon, edge_lat = _mirror_latlon(
            lon_west,
            lat_south,
            lon_east,
            lat_north,
            lon_west_tile_edge[0, i],
            lat_west_tile_edge_mirror[0, i],
            np,
        )
        lon_south_tile_edge[i, 0] = edge_lon
        lat_south_tile_edge[i, 0] = edge_lat

    # map edges on the sphere back to cube: intersection at x = -1/sqrt(3)
    i = 0
    for j in range(im + 1):
        pp_west_tile_edge[:, i, j] = _latlon2xyz(
            lon_west_tile_edge[i, j], lat_west_tile_edge[i, j], np
        )
        pp_west_tile_edge[1, i, j] = (
            -pp_west_tile_edge[1, i, j] * (3 ** -0.5) / pp_west_tile_edge[0, i, j]
        )
        pp_west_tile_edge[2, i, j] = (
            -pp_west_tile_edge[2, i, j] * (3 ** -0.5) / pp_west_tile_edge[0, i, j]
        )
    if west_edge:
        pp[:, 0, :] = pp_west_tile_edge[:, 0, :]

    j = 0
    for i in range(im + 1):
        pp_south_tile_edge[:, i, j] = _latlon2xyz(
            lon_south_tile_edge[i, j], lat_south_tile_edge[i, j], np
        )
        pp_south_tile_edge[1, i, j] = (
            -pp_south_tile_edge[1, i, j] * (3 ** -0.5) / pp_south_tile_edge[0, i, j]
        )
        pp_south_tile_edge[2, i, j] = (
            -pp_south_tile_edge[2, i, j] * (3 ** -0.5) / pp_south_tile_edge[0, i, j]
        )
    if south_edge:
        pp[:, :, 0] = pp_south_tile_edge[:, :, 0]

    # set 4 corners
    if south_edge or west_edge:
        sw_xyz = _latlon2xyz(lon_west, lat_south, np)
        if south_edge and west_edge:
            pp[:, 0, 0] = sw_xyz
        if south_edge:
            pp_west_tile_edge[:, 0, 0] = sw_xyz
        if west_edge:
            pp_south_tile_edge[:, 0, 0] = sw_xyz
    if east_edge:
        se_xyz = _latlon2xyz(lon_east, lat_south, np)
        pp_south_tile_edge[:, im, 0] = se_xyz

    if north_edge:
        nw_xyz = _latlon2xyz(lon_west, lat_north, np)
        pp_west_tile_edge[:, 0, im] = nw_xyz

    if north_edge and east_edge:
        pp[:, im, im] = _latlon2xyz(lon_east, lat_north, np)

    pp[0, :, :] = -(3 ** -0.5)
    for j in range(start_j, im + 1):
        # copy y-z face of the cube along j=0
        pp[1, start_i:, j] = pp_south_tile_edge[1, start_i:, 0]  # pp[1,:,0]
        # copy along i=0
        pp[2, start_i:, j] = pp_west_tile_edge[2, 0, j]  # pp[4,0,j]

    _cart_to_latlon(im + 1, pp, lon, lat, np)
    # TODO replicating the last step of gnomonic_grid until api is finalized
    # remove this if this method is called from gnomonic_grid
    # if grid_type < 3:
    symm_ed(lon, lat)
    lon[:] -= PI


def _corner_to_center_mean(corner_array):
    """Given a 2D array on cell corners, return a 2D array on cell centers with the
    mean value of each of the corners."""
    return xyz_midpoint(
        corner_array[1:, 1:],
        corner_array[:-1, :-1],
        corner_array[1:, :-1],
        corner_array[:-1, 1:],
    )


def normalize_vector(np, *vector_components):
    scale = np.divide(
        1.0, np.sum(np.asarray([item ** 2.0 for item in vector_components])) ** 0.5
    )
    return np.asarray([item * scale for item in vector_components])


def normalize_xyz(xyz):
    # double transpose to broadcast along last dimension instead of first
    return (xyz.T / ((xyz ** 2).sum(axis=-1) ** 0.5).T).T


def lon_lat_midpoint(lon1, lon2, lat1, lat2, np):
    p1 = lon_lat_to_xyz(lon1, lat1, np)
    p2 = lon_lat_to_xyz(lon2, lat2, np)
    midpoint = xyz_midpoint(p1, p2)
    return xyz_to_lon_lat(midpoint, np)


def xyz_midpoint(*points):
    return normalize_xyz(sum(points))


def lon_lat_corner_to_cell_center(lon, lat, np):
    # just perform the mean in x-y-z space and convert back
    xyz = lon_lat_to_xyz(lon, lat, np)
    center = _corner_to_center_mean(xyz)
    return xyz_to_lon_lat(center, np)


def lon_lat_to_xyz(lon, lat, np):
    """map (lon, lat) to (x, y, z)
    Args:
        lon: 2d array of longitudes
        lat: 2d array of latitudes
        np: numpy-like module for arrays
    Returns:
        xyz: 3d array whose last dimension is length 3 and indicates x/y/z value
    """
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    x, y, z = normalize_vector(np, x, y, z)
    if len(lon.shape) == 2:
        xyz = np.concatenate([arr[:, :, None] for arr in (x, y, z)], axis=-1)
    elif len(lon.shape) == 1:
        xyz = np.concatenate([arr[:, None] for arr in (x, y, z)], axis=-1)
    return xyz


def xyz_to_lon_lat(xyz, np):
    """map (x, y, z) to (lon, lat)
    Returns:
        xyz: 3d array whose last dimension is length 3 and indicates x/y/z value
        np: numpy-like module for arrays
    Returns:
        lon: 2d array of longitudes
        lat: 2d array of latitudes
    """
    xyz = normalize_xyz(xyz)
    # double transpose to index last dimension, regardless of number of dimensions
    x = xyz.T[0, :].T
    y = xyz.T[1, :].T
    z = xyz.T[2, :].T
    lon = 0.0 * x
    nonzero_lon = np.abs(x) + np.abs(y) >= 1.0e-10
    lon[nonzero_lon] = np.arctan2(y[nonzero_lon], x[nonzero_lon])
    negative_lon = lon < 0.0
    while np.any(negative_lon):
        lon[negative_lon] += 2 * PI
        negative_lon = lon < 0.0
    lat = np.arcsin(z)
    return lon, lat


def _latlon2xyz(lon, lat, np):
    """map (lon, lat) to (x, y, z)"""
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return normalize_vector(np, x, y, z)


def _xyz2latlon(x, y, z, np):
    """map (x, y, z) to (lon, lat)"""
    x, y, z = normalize_vector(np, x, y, z)
    lon = 0.0 * x
    nonzero_lon = np.abs(x) + np.abs(y) >= 1.0e-10
    lon[nonzero_lon] = np.arctan2(y[nonzero_lon], x[nonzero_lon])
    negative_lon = lon < 0.0
    while np.any(negative_lon):
        lon[negative_lon] += 2 * PI
        negative_lon = lon < 0.0
    lat = np.arcsin(z)

    return lon, lat


def _cart_to_latlon(im, q, xs, ys, np):
    """map (x, y, z) to (lon, lat)"""

    esl = 1.0e-10

    for j in range(im):
        for i in range(im):
            p = q[:, i, j]
            dist = np.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
            p = p / dist

            if np.abs(p[0]) + np.abs(p[1]) < esl:
                lon = 0.0
            else:
                lon = np.arctan2(p[1], p[0])  # range [-PI, PI]

            if lon < 0.0:
                lon = np.add(2.0 * PI, lon)

            lat = np.arcsin(p[2])

            xs[i, j] = lon
            ys[i, j] = lat

            q[:, i, j] = p


def _mirror_latlon(lon1, lat1, lon2, lat2, lon0, lat0, np):

    p0 = _latlon2xyz(lon0, lat0, np)
    p1 = _latlon2xyz(lon1, lat1, np)
    p2 = _latlon2xyz(lon2, lat2, np)
    nb = _vect_cross(p1, p2, np)

    pdot = np.sqrt(nb[0] ** 2 + nb[1] ** 2 + nb[2] ** 2)
    nb = nb / pdot

    pdot = p0[0] * nb[0] + p0[1] * nb[1] + p0[2] * nb[2]
    pp = p0 - np.multiply(2.0, pdot) * nb

    lon3 = np.empty((1, 1))
    lat3 = np.empty((1, 1))
    pp3 = np.empty((3, 1, 1))
    pp3[:, 0, 0] = pp
    _cart_to_latlon(1, pp3, lon3, lat3, np)

    return lon3[0, 0], lat3[0, 0]


def _vect_cross(p1, p2, np):
    return np.asarray(
        [
            p1[1] * p2[2] - p1[2] * p2[1],
            p1[2] * p2[0] - p1[0] * p2[2],
            p1[0] * p2[1] - p1[1] * p2[0],
        ]
    )


def symm_ed(lon, lat):
    pass


def _great_circle_beta_lon_lat(lon1, lon2, lat1, lat2, np):
    """Returns the great-circle distance between points along the desired axis,
    as a fraction of the radius of the sphere."""
    return (
        np.arcsin(
            np.sqrt(
                np.sin((lat1 - lat2) / 2.0) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2.0) ** 2
            )
        )
        * 2.0
    )


def great_circle_distance_along_axis(lon, lat, radius, np, axis=0):
    """Returns the great-circle distance between points along the desired axis."""
    lon, lat = np.broadcast_arrays(lon, lat)
    if len(lon.shape) == 1:
        case_1d = True
        # add singleton dimension so we can use the same indexing notation as n-d cases
        lon, lat = lon[:, None], lat[:, None]
    else:
        case_1d = False
    swap_dims = list(range(len(lon.shape)))
    swap_dims[axis], swap_dims[0] = swap_dims[0], swap_dims[axis]
    # below code computes distance along first axis, so we put the desired axis there
    lon, lat = lon.transpose(swap_dims), lat.transpose(swap_dims)
    result = great_circle_distance_lon_lat(
        lon[:-1, :], lon[1:, :], lat[:-1, :], lat[1:, :], radius, np
    )
    result = result.transpose(swap_dims)  # remember to swap back
    if case_1d:
        result = result[:, 0]  # remove the singleton dimension we added
    return result


def great_circle_distance_lon_lat(lon1, lon2, lat1, lat2, radius, np):
    return radius * _great_circle_beta_lon_lat(lon1, lon2, lat1, lat2, np)


def great_circle_distance_xyz(p1, p2, radius, np):
    lon1, lat1 = xyz_to_lon_lat(p1, np)
    lon2, lat2 = xyz_to_lon_lat(p2, np)
    return great_circle_distance_lon_lat(lon1, lon2, lat1, lat2, radius, np)


def get_area(lon, lat, radius, np):
    """
    Given latitude and longitude on cell corners, return the area of each cell.
    """
    xyz = lon_lat_to_xyz(lon, lat, np)
    lower_left = xyz[(slice(None, -1), slice(None, -1), slice(None, None))]
    lower_right = xyz[(slice(1, None), slice(None, -1), slice(None, None))]
    upper_left = xyz[(slice(None, -1), slice(1, None), slice(None, None))]
    upper_right = xyz[(slice(1, None), slice(1, None), slice(None, None))]
    return get_rectangle_area(
        lower_left, upper_left, upper_right, lower_right, radius, np
    )


def set_corner_area_to_triangle_area(
    lon, lat, area, tile_partitioner, rank, radius, np
):
    """
    Given latitude and longitude on cell corners, and an array of cell areas, set the
    four corner areas to the area of the inner triangle at those corners.
    """
    xyz = lon_lat_to_xyz(lon, lat, np)
    lower_left = xyz[(slice(None, -1), slice(None, -1), slice(None, None))]
    lower_right = xyz[(slice(1, None), slice(None, -1), slice(None, None))]
    upper_left = xyz[(slice(None, -1), slice(1, None), slice(None, None))]
    upper_right = xyz[(slice(1, None), slice(1, None), slice(None, None))]
    if tile_partitioner.on_tile_left(rank) and tile_partitioner.on_tile_bottom(rank):
        area[0, 0] = get_triangle_area(
            upper_left[0, 0], upper_right[0, 0], lower_right[0, 0], radius, np
        )
    if tile_partitioner.on_tile_right(rank) and tile_partitioner.on_tile_bottom(rank):
        area[-1, 0] = get_triangle_area(
            upper_right[-1, 0], upper_left[-1, 0], lower_left[-1, 0], radius, np
        )
    if tile_partitioner.on_tile_right(rank) and tile_partitioner.on_tile_top(rank):
        area[-1, -1] = get_triangle_area(
            lower_right[-1, -1], lower_left[-1, -1], upper_left[-1, -1], radius, np
        )
    if tile_partitioner.on_tile_left(rank) and tile_partitioner.on_tile_top(rank):
        area[0, -1] = get_triangle_area(
            lower_left[0, -1], lower_right[0, -1], upper_right[0, -1], radius, np
        )


def set_c_grid_tile_border_area(
    xyz_dgrid, xyz_agrid, radius, area_cgrid, tile_partitioner, rank, np
):
    """
    Using latitude and longitude without halo points, fix C-grid area at tile edges and
    corners.
    Naively, the c-grid area is calculated as the area between the rectangle at the
    four corners of the grid cell. At tile edges however, this is not accurate,
    because the area makes a butterfly-like shape as it crosses the tile boundary.
    Instead we calculate the area on one side of that shape, and multiply it by two.
    At corners, the corner is composed of three rectangles from each tile bordering
    the corner. We calculate the area from one tile and multiply it by three.
    Args:
        xyz_dgrid: d-grid cartesian coordinates as a 3-d array, last dimension
            of length 3 indicating x/y/z
        xyz_agrid: a-grid cartesian coordinates as a 3-d array, last dimension
            of length 3 indicating x/y/z
        area_cgrid: 2d array of c-grid areas
        radius: radius of Earth in metres
        tile_partitioner: partitioner class to determine subtile position
        rank: rank of current tile
        np: numpy-like module to interact with arrays
    """

    if tile_partitioner.on_tile_left(rank):
        _set_c_grid_west_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np)

    if tile_partitioner.on_tile_top(rank):
        _set_c_grid_north_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np)

    if tile_partitioner.on_tile_right(rank):
        _set_c_grid_east_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np)

    if tile_partitioner.on_tile_bottom(rank):
        _set_c_grid_south_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np)

    """
# TODO add these back if we change the fortran side, or
#  decide the 'if sw_corner' should happen
    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_top(rank):
             _set_c_grid_northwest_corner_area(
                 xyz_dgrid, xyz_agrid, area_cgrid, radius, np
             )
        if tile_partitioner.on_tile_bottom(rank):
            _set_c_grid_southwest_corner_area_mod(
                xyz_dgrid, xyz_agrid, area_cgrid, radius, np
            )
    if tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _set_c_grid_southeast_corner_area(
                xyz_dgrid, xyz_agrid, area_cgrid, radius, np
            )
        if tile_partitioner.on_tile_top(rank):
            _set_c_grid_northeast_corner_area(
                xyz_dgrid, xyz_agrid, area_cgrid, radius, np
            )
    """


def _set_c_grid_west_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    xyz_y_center = 0.5 * (xyz_dgrid[1, :-1] + xyz_dgrid[1, 1:])
    area_cgrid[0, :] = 2 * get_rectangle_area(
        xyz_y_center[:-1],
        xyz_agrid[1, :-1],
        xyz_agrid[1, 1:],
        xyz_y_center[1:],
        radius,
        np,
    )


def _set_c_grid_east_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    _set_c_grid_west_edge_area(
        xyz_dgrid[::-1, :], xyz_agrid[::-1, :], area_cgrid[::-1, :], radius, np
    )


def _set_c_grid_north_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    _set_c_grid_south_edge_area(
        xyz_dgrid[:, ::-1], xyz_agrid[:, ::-1], area_cgrid[:, ::-1], radius, np
    )


def _set_c_grid_south_edge_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    _set_c_grid_west_edge_area(
        xyz_dgrid.transpose(1, 0, 2),
        xyz_agrid.transpose(1, 0, 2),
        area_cgrid.transpose(1, 0),
        radius,
        np,
    )


def _set_c_grid_southwest_corner_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    lower_right = normalize_xyz((xyz_dgrid[0, 0, :] + xyz_dgrid[1, 0, :]))  # Fortran P2
    upper_right = xyz_agrid[0, 0, :]  # Fortran P3
    upper_left = normalize_xyz((xyz_dgrid[0, 0, :] + xyz_dgrid[0, 1, :]))  # Fortran P4
    lower_left = xyz_dgrid[0, 0, :]  # Fortran P1
    area_cgrid[0, 0] = 3.0 * get_rectangle_area(
        lower_left, upper_left, upper_right, lower_right, radius, np
    )


def _set_c_grid_southwest_corner_area_mod(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    _set_c_grid_southwest_corner_area(
        xyz_dgrid[1:, 1:], xyz_agrid[1:, 1:], area_cgrid[:, :], radius, np
    )


def _set_c_grid_northwest_corner_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    _set_c_grid_southwest_corner_area(
        xyz_dgrid[1:, ::-1], xyz_agrid[1:, ::-1], area_cgrid[:, ::-1], radius, np
    )


def _set_c_grid_northeast_corner_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    _set_c_grid_southwest_corner_area(
        xyz_dgrid[::-1, ::-1], xyz_agrid[::-1, ::-1], area_cgrid[::-1, ::-1], radius, np
    )


def _set_c_grid_southeast_corner_area(xyz_dgrid, xyz_agrid, area_cgrid, radius, np):
    _set_c_grid_southwest_corner_area(
        xyz_dgrid[::-1, 1:], xyz_agrid[::-1, 1:], area_cgrid[::-1, :], radius, np
    )


def set_tile_border_dxc(xyz_dgrid, xyz_agrid, radius, dxc, tile_partitioner, rank, np):
    if tile_partitioner.on_tile_left(rank):
        _set_tile_west_dxc(xyz_dgrid, xyz_agrid, radius, dxc, np)
    if tile_partitioner.on_tile_right(rank):
        _set_tile_east_dxc(xyz_dgrid, xyz_agrid, radius, dxc, np)


def _set_tile_west_dxc(xyz_dgrid, xyz_agrid, radius, dxc, np):
    tile_edge_point = 0.5 * (xyz_dgrid[0, 1:] + xyz_dgrid[0, :-1])
    cell_center_point = xyz_agrid[0, :]
    dxc[0, :] = 2 * great_circle_distance_xyz(
        tile_edge_point, cell_center_point, radius, np
    )


def _set_tile_east_dxc(xyz_dgrid, xyz_agrid, radius, dxc, np):
    _set_tile_west_dxc(xyz_dgrid[::-1, :], xyz_agrid[::-1, :], radius, dxc[::-1, :], np)


def set_tile_border_dyc(xyz_dgrid, xyz_agrid, radius, dyc, tile_partitioner, rank, np):
    if tile_partitioner.on_tile_top(rank):
        _set_tile_north_dyc(xyz_dgrid, xyz_agrid, radius, dyc, np)
    if tile_partitioner.on_tile_bottom(rank):
        _set_tile_south_dyc(xyz_dgrid, xyz_agrid, radius, dyc, np)


def _set_tile_north_dyc(xyz_dgrid, xyz_agrid, radius, dyc, np):
    _set_tile_east_dxc(
        xyz_dgrid.transpose(1, 0, 2),
        xyz_agrid.transpose(1, 0, 2),
        radius,
        dyc.transpose(1, 0),
        np,
    )


def _set_tile_south_dyc(xyz_dgrid, xyz_agrid, radius, dyc, np):
    _set_tile_west_dxc(
        xyz_dgrid.transpose(1, 0, 2),
        xyz_agrid.transpose(1, 0, 2),
        radius,
        dyc.transpose(1, 0),
        np,
    )


def get_rectangle_area(p1, p2, p3, p4, radius, np):
    """
    Given four point arrays whose last dimensions are x/y/z in clockwise or
    counterclockwise order, return an array of spherical rectangle areas.
    NOTE, this is not the exact same order of operations as the Fortran code
    This results in some errors in the last digit, but the spherical_angle
    is an exact match. The errors in the last digit multipled out by the radius
    end up causing relative errors larger than 1e-14, but still wtihin 1e-12.
    """
    total_angle = spherical_angle(p2, p3, p1, np)
    for (
        q1,
        q2,
        q3,
    ) in ((p3, p2, p4), (p4, p3, p1), (p1, p4, p2)):
        total_angle += spherical_angle(q1, q2, q3, np)

    return (total_angle - 2 * PI) * radius ** 2


def get_triangle_area(p1, p2, p3, radius, np):
    """
    Given three point arrays whose last dimensions are x/y/z, return an array of
    spherical triangle areas.
    """

    total_angle = spherical_angle(p1, p2, p3, np)
    for q1, q2, q3 in ((p2, p3, p1), (p3, p1, p2)):
        total_angle += spherical_angle(q1, q2, q3, np)
    return (total_angle - PI) * radius ** 2


def fortran_vector_spherical_angle(e1, e2, e3):
    """
   The Fortran version
    Given x/y/z tuples, compute the spherical angle between
    them according to:
!           p3
!         /
!        /
!       p_center ---> angle
!         \
!          \
!           p2
    This angle will always be less than Pi.
    """

    # Vector P:
    px = e1[1] * e2[2] - e1[2] * e2[1]
    py = e1[2] * e2[0] - e1[0] * e2[2]
    pz = e1[0] * e2[1] - e1[1] * e2[0]
    # Vector Q:
    qx = e1[1] * e3[2] - e1[2] * e3[1]
    qy = e1[2] * e3[0] - e1[0] * e3[2]
    qz = e1[0] * e3[1] - e1[1] * e3[0]
    ddd = (px * px + py * py + pz * pz) * (qx * qx + qy * qy + qz * qz)

    if ddd <= 0.0:
        angle = 0.0
    else:
        ddd = (px * qx + py * qy + pz * qz) / math.sqrt(ddd)
        if abs(ddd) > 1.0:
            # FIX (lmh) to correctly handle co-linear points (angle near pi or 0)
            if ddd < 0.0:
                angle = 4.0 * math.atan(1.0)  # should be pi
            else:
                angle = 0.0
        else:
            angle = math.acos(ddd)
    return angle


def spherical_angle(p_center, p2, p3, np):
    """
    Given ndarrays whose last dimension is x/y/z, compute the spherical angle between
    them according to:
!           p3
!         /
!        /
!       p_center ---> angle
!         \
!          \
!           p2
    This angle will always be less than Pi.
    """

    p = np.cross(p_center, p2)
    q = np.cross(p_center, p3)
    angle = np.arccos(
        np.sum(p * q, axis=-1)
        / np.sqrt(np.sum(p ** 2, axis=-1) * np.sum(q ** 2, axis=-1))
    )
    if not np.isscalar(angle):
        angle[np.isnan(angle)] = 0.0
    elif math.isnan(angle):
        angle = 0.0

    return angle


def spherical_cos(p_center, p2, p3, np):
    """
    As Spherical angle, but returns cos(angle)
    """
    p = np.cross(p_center, p2)
    q = np.cross(p_center, p3)
    return np.sum(p * q, axis=-1) / np.sqrt(
        np.sum(p ** 2, axis=-1) * np.sum(q ** 2, axis=-1)
    )


def get_unit_vector_direction(p1, p2, np):
    """
    Returms the unit vector pointing from a set of lonlat points p1 to lonlat points p2
    """
    xyz1 = lon_lat_to_xyz(p1[:, :, 0], p1[:, :, 1], np)
    xyz2 = lon_lat_to_xyz(p2[:, :, 0], p2[:, :, 1], np)
    midpoint = xyz_midpoint(xyz1, xyz2)
    p3 = np.cross(xyz2, xyz1)
    return normalize_xyz(np.cross(midpoint, p3))


def get_lonlat_vect(lonlat_grid, np):
    """
    Calculates the unit vectors pointing in the longitude/latitude directions
    for a set of longitude/latitude points
    """
    lon_vector = np.array(
        [
            -np.sin(lonlat_grid[:, :, 0]),
            np.cos(lonlat_grid[:, :, 0]),
            np.zeros(lonlat_grid[:, :, 0].shape),
        ]
    ).transpose([1, 2, 0])
    lat_vector = np.array(
        [
            -np.sin(lonlat_grid[:, :, 1]) * np.cos(lonlat_grid[:, :, 0]),
            -np.sin(lonlat_grid[:, :, 1]) * np.sin(lonlat_grid[:, :, 0]),
            np.cos(lonlat_grid[:, :, 1]),
        ]
    ).transpose([1, 2, 0])
    return lon_vector, lat_vector
