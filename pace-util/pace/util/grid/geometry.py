from pace.util import Quantity, TilePartitioner

from .gnomonic import (
    get_lonlat_vect,
    get_unit_vector_direction,
    great_circle_distance_lon_lat,
    lon_lat_midpoint,
    normalize_xyz,
    spherical_cos,
    xyz_midpoint,
)


def get_center_vector(
    xyz_gridpoints,
    grid_type: int,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
    np,
):
    """
    Calculates the cartesian unit vectors at the center of each grid cell.

    Returns:
        vector1: the horizontal unit vector
        vector2: the vertical unit vector
    """
    big_number = 1.0e8

    if grid_type < 3:
        if False:  # ifdef OLD_VECT
            vector1 = (
                xyz_gridpoints[1:, :-1, :]
                + xyz_gridpoints[1:, 1:, :]
                - xyz_gridpoints[:-1, :-1, :]
                - xyz_gridpoints[:-1, 1:, :]
            )
            vector2 = (
                xyz_gridpoints[:-1, 1:, :]
                + xyz_gridpoints[1:, 1:, :]
                - xyz_gridpoints[:-1, :-1, :]
                - xyz_gridpoints[1:, :-1, :]
            )
        else:
            center_points = xyz_midpoint(
                xyz_gridpoints[:-1, :-1, :],
                xyz_gridpoints[1:, :-1, :],
                xyz_gridpoints[:-1, 1:, :],
                xyz_gridpoints[1:, 1:, :],
            )

            p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, :], xyz_gridpoints[:-1, 1:, :])
            p2 = xyz_midpoint(xyz_gridpoints[1:, :-1, :], xyz_gridpoints[1:, 1:, :])
            p3 = np.cross(p2, p1)
            vector1 = normalize_xyz(np.cross(center_points, p3))

            p1 = xyz_midpoint(xyz_gridpoints[:-1, :-1, :], xyz_gridpoints[1:, :-1, :])
            p2 = xyz_midpoint(xyz_gridpoints[:-1, 1:, :], xyz_gridpoints[1:, 1:, :])
            p3 = np.cross(p2, p1)
            vector2 = normalize_xyz(np.cross(center_points, p3))

        # fill ghost on ec1 and ec2:
        _fill_halo_corners(vector1, big_number, nhalo, tile_partitioner, rank)
        _fill_halo_corners(vector2, big_number, nhalo, tile_partitioner, rank)

    else:
        shape_dgrid = xyz_gridpoints.shape
        vector1 = np.zeros((shape_dgrid[0] - 1, shape_dgrid[1] - 1, 3))
        vector2 = np.zeros((shape_dgrid[0] - 1, shape_dgrid[1] - 1, 3))
        vector1[:, :, 0] = 1
        vector2[:, :, 1] = 1

    return vector1, vector2


def calc_unit_vector_west(
    xyz_dgrid,
    xyz_agrid,
    grid_type: int,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
    np,
):
    """
    Calculates the cartesian unit vectors at the left/right edges of each grid cell.

    Returns:
        vector1: the horizontal unit vector
        vector2: the vertical unit vector

    """
    ew1 = np.zeros((xyz_dgrid.shape[0], xyz_agrid.shape[1], 3))
    ew2 = np.zeros((xyz_dgrid.shape[0], xyz_agrid.shape[1], 3))
    if grid_type < 3:

        pp = xyz_midpoint(xyz_dgrid[1:-1, :-1, :3], xyz_dgrid[1:-1, 1:, :3])

        p2 = np.cross(xyz_agrid[:-1, :, :3], xyz_agrid[1:, :, :3])
        if tile_partitioner.on_tile_left(rank):
            p2[nhalo - 1] = np.cross(pp[nhalo - 1], xyz_agrid[nhalo, :, :3])
        if tile_partitioner.on_tile_right(rank):
            p2[-nhalo] = np.cross(xyz_agrid[-nhalo - 1, :, :3], pp[-nhalo])

        ew1[1:-1, :, :] = normalize_xyz(np.cross(p2, pp))
        p1 = np.cross(xyz_dgrid[1:-1, :-1, :], xyz_dgrid[1:-1, 1:, :])
        ew2[1:-1, :, :] = normalize_xyz(np.cross(p1, pp))

        # fill ghost on ew:
        _fill_halo_corners(ew1, 0.0, nhalo, tile_partitioner, rank)
        _fill_halo_corners(ew2, 0.0, nhalo, tile_partitioner, rank)

    else:
        ew1[:, :, 1] = 1.0
        ew2[:, :, 2] = 1.0

    return ew1[1:-1, :, :], ew2[1:-1, :, :]


def calc_unit_vector_south(
    xyz_dgrid,
    xyz_agrid,
    grid_type: int,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
    np,
):
    """
    Calculates the cartesian unit vectors at the top/bottom edges of each grid cell.

    Returns:
        vector1: the horizontal unit vector
        vector2: the vertical unit vector
    """
    es1 = np.zeros((xyz_agrid.shape[0], xyz_dgrid.shape[1], 3))
    es2 = np.zeros((xyz_agrid.shape[0], xyz_dgrid.shape[1], 3))
    if grid_type < 3:

        pp = xyz_midpoint(xyz_dgrid[:-1, 1:-1, :3], xyz_dgrid[1:, 1:-1, :3])
        p2 = np.cross(xyz_agrid[:, :-1, :3], xyz_agrid[:, 1:, :3])
        if tile_partitioner.on_tile_bottom(rank):
            p2[:, nhalo - 1] = np.cross(pp[:, nhalo - 1], xyz_agrid[:, nhalo, :3])
        if tile_partitioner.on_tile_top(rank):
            p2[:, -nhalo] = np.cross(xyz_agrid[:, -nhalo - 1, :3], pp[:, -nhalo])

        es2[:, 1:-1, :] = normalize_xyz(np.cross(p2, pp))

        p1 = np.cross(xyz_dgrid[:-1, 1:-1, :], xyz_dgrid[1:, 1:-1, :])
        es1[:, 1:-1, :] = normalize_xyz(np.cross(p1, pp))

        # fill ghost on es:
        _fill_halo_corners(es1, 0.0, nhalo, tile_partitioner, rank)
        _fill_halo_corners(es2, 0.0, nhalo, tile_partitioner, rank)
    else:
        es1[:, :, 1] = 1.0
        es2[:, :, 2] = 1.0

    return es1[:, 1:-1, :], es2[:, 1:-1, :]


def calculate_supergrid_cos_sin(
    xyz_dgrid,
    xyz_agrid,
    ec1,
    ec2,
    grid_type: int,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
    np,
):
    """
    Calculates the cosine and sine of the grid angles at each of the following points
    in a supergrid cell:
    9---4---8
    |       |
    1   5   3
    |       |
    6---2---7
    """
    big_number = 1.0e8
    tiny_number = 1.0e-8

    shape_a = xyz_agrid.shape
    cos_sg = np.zeros((shape_a[0], shape_a[1], 9)) + big_number
    sin_sg = np.zeros((shape_a[0], shape_a[1], 9)) + tiny_number

    if grid_type < 3:
        cos_sg[:, :, 5] = spherical_cos(
            xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, 1:, :], np
        )
        cos_sg[:, :, 6] = -1 * spherical_cos(
            xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, 1:, :], np
        )
        cos_sg[:, :, 7] = spherical_cos(
            xyz_dgrid[1:, 1:, :], xyz_dgrid[1:, :-1, :], xyz_dgrid[:-1, 1:, :], np
        )
        cos_sg[:, :, 8] = -1 * spherical_cos(
            xyz_dgrid[:-1, 1:, :], xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, 1:, :], np
        )

        midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, :], xyz_dgrid[:-1, 1:, :])
        cos_sg[:, :, 0] = spherical_cos(
            midpoint, xyz_agrid[:, :, :], xyz_dgrid[:-1, 1:, :], np
        )
        midpoint = xyz_midpoint(xyz_dgrid[:-1, :-1, :], xyz_dgrid[1:, :-1, :])
        cos_sg[:, :, 1] = spherical_cos(
            midpoint, xyz_dgrid[1:, :-1, :], xyz_agrid[:, :, :], np
        )
        midpoint = xyz_midpoint(xyz_dgrid[1:, :-1, :], xyz_dgrid[1:, 1:, :])
        cos_sg[:, :, 2] = spherical_cos(
            midpoint, xyz_agrid[:, :, :], xyz_dgrid[1:, :-1, :], np
        )
        midpoint = xyz_midpoint(xyz_dgrid[:-1, 1:, :], xyz_dgrid[1:, 1:, :])
        cos_sg[:, :, 3] = spherical_cos(
            midpoint, xyz_dgrid[:-1, 1:, :], xyz_agrid[:, :, :], np
        )

        cos_sg[:, :, 4] = np.sum(ec1 * ec2, axis=-1)

        cos_sg[abs(1.0 - cos_sg) < 1e-15] = 1.0

        sin_sg_tmp = 1.0 - cos_sg ** 2
        sin_sg_tmp[sin_sg_tmp < 0] = 0.0
        sin_sg = np.sqrt(sin_sg_tmp)
        sin_sg[sin_sg > 1.0] = 1.0

        # Adjust for corners:
        if tile_partitioner.on_tile_left(rank):
            if tile_partitioner.on_tile_bottom(rank):  # southwest corner
                sin_sg[nhalo - 1, :nhalo, 2] = sin_sg[:nhalo, nhalo, 1]
                sin_sg[:nhalo, nhalo - 1, 3] = sin_sg[nhalo, :nhalo, 0]
            if tile_partitioner.on_tile_top(rank):  # northwest corner
                sin_sg[nhalo - 1, -nhalo:, 2] = sin_sg[:nhalo, -nhalo - 1, 3][::-1]
                sin_sg[:nhalo, -nhalo, 1] = sin_sg[nhalo, -nhalo - 2 : -nhalo + 1, 0]
        if tile_partitioner.on_tile_right(rank):
            if tile_partitioner.on_tile_bottom(rank):  # southeast corner
                sin_sg[-nhalo, :nhalo, 0] = sin_sg[-nhalo:, nhalo, 1][::-1]
                sin_sg[-nhalo:, nhalo - 1, 3] = sin_sg[-nhalo - 1, :nhalo, 2][::-1]
            if tile_partitioner.on_tile_top(rank):  # northeast corner
                sin_sg[-nhalo, -nhalo:, 0] = sin_sg[-nhalo:, -nhalo - 1, 3]
                sin_sg[-nhalo:, -nhalo, 1] = sin_sg[-nhalo - 1, -nhalo:, 2]

    else:
        cos_sg[:] = 0.0
        sin_sg[:] = 1.0

    return cos_sg, sin_sg


def calculate_l2c_vu(dgrid, nhalo: int, np):
    # AAM correction

    point1v = dgrid[nhalo:-nhalo, nhalo : -nhalo - 1, :]
    point2v = dgrid[nhalo:-nhalo, nhalo + 1 : -nhalo, :]
    midpoint_y = np.array(
        lon_lat_midpoint(
            point1v[:, :, 0], point2v[:, :, 0], point1v[:, :, 1], point2v[:, :, 1], np
        )
    ).transpose([1, 2, 0])
    unit_dir_y = get_unit_vector_direction(point1v, point2v, np)
    exv, eyv = get_lonlat_vect(midpoint_y, np)
    l2c_v = np.cos(midpoint_y[:, :, 1]) * np.sum(unit_dir_y * exv, axis=-1)

    point1u = dgrid[nhalo : -nhalo - 1, nhalo:-nhalo, :]
    point2u = dgrid[nhalo + 1 : -nhalo, nhalo:-nhalo, :]
    midpoint_x = np.array(
        lon_lat_midpoint(
            point1u[:, :, 0], point2u[:, :, 0], point1u[:, :, 1], point2u[:, :, 1], np
        )
    ).transpose([1, 2, 0])
    unit_dir_x = get_unit_vector_direction(point1u, point2u, np)
    exu, eyu = get_lonlat_vect(midpoint_x, np)
    l2c_u = np.cos(midpoint_x[:, :, 1]) * np.sum(unit_dir_x * exu, axis=-1)

    return l2c_v, l2c_u


def calculate_xy_unit_vectors(
    xyz_dgrid, nhalo: int, tile_partitioner: TilePartitioner, rank: int, np
):
    """
    Calculates the cartesian unit vectors at the corners of each grid cell.
    vector1 is the horizontal unit vector, while
    vector2 is the vertical unit vector
    """
    cross_vect_x = np.cross(
        xyz_dgrid[nhalo - 1 : -nhalo - 1, nhalo:-nhalo, :],
        xyz_dgrid[nhalo + 1 : -nhalo + 1, nhalo:-nhalo, :],
    )
    if tile_partitioner.on_tile_left(rank):
        cross_vect_x[0, :] = np.cross(
            xyz_dgrid[nhalo, nhalo:-nhalo, :], xyz_dgrid[nhalo + 1, nhalo:-nhalo, :]
        )
    if tile_partitioner.on_tile_right(rank):
        cross_vect_x[-1, :] = np.cross(
            xyz_dgrid[-nhalo - 2, nhalo:-nhalo, :],
            xyz_dgrid[-nhalo - 1, nhalo:-nhalo, :],
        )
    unit_x_vector = normalize_xyz(
        np.cross(cross_vect_x, xyz_dgrid[nhalo:-nhalo, nhalo:-nhalo])
    )

    cross_vect_y = np.cross(
        xyz_dgrid[nhalo:-nhalo, nhalo - 1 : -nhalo - 1, :],
        xyz_dgrid[nhalo:-nhalo, nhalo + 1 : -nhalo + 1, :],
    )
    if tile_partitioner.on_tile_bottom(rank):
        cross_vect_y[:, 0] = np.cross(
            xyz_dgrid[nhalo:-nhalo, nhalo, :], xyz_dgrid[nhalo:-nhalo, nhalo + 1, :]
        )
    if tile_partitioner.on_tile_top(rank):
        cross_vect_y[:, -1] = np.cross(
            xyz_dgrid[nhalo:-nhalo, -nhalo - 2, :],
            xyz_dgrid[nhalo:-nhalo, -nhalo - 1, :],
        )
    unit_y_vector = normalize_xyz(
        np.cross(cross_vect_y, xyz_dgrid[nhalo:-nhalo, nhalo:-nhalo])
    )

    return unit_x_vector, unit_y_vector


def calculate_trig_uv(
    xyz_dgrid,
    cos_sg,
    sin_sg,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
    np,
):
    """
    Calculates more trig quantities
    """

    big_number = 1.0e8
    tiny_number = 1.0e-8

    dgrid_shape_2d = xyz_dgrid[:, :, 0].shape
    cosa = np.zeros(dgrid_shape_2d) + big_number
    sina = np.zeros(dgrid_shape_2d) + big_number
    cosa_u = np.zeros((dgrid_shape_2d[0], dgrid_shape_2d[1] - 1)) + big_number
    sina_u = np.zeros((dgrid_shape_2d[0], dgrid_shape_2d[1] - 1)) + big_number
    rsin_u = np.zeros((dgrid_shape_2d[0], dgrid_shape_2d[1] - 1)) + big_number
    cosa_v = np.zeros((dgrid_shape_2d[0] - 1, dgrid_shape_2d[1])) + big_number
    sina_v = np.zeros((dgrid_shape_2d[0] - 1, dgrid_shape_2d[1])) + big_number
    rsin_v = np.zeros((dgrid_shape_2d[0] - 1, dgrid_shape_2d[1])) + big_number

    cosa[nhalo:-nhalo, nhalo:-nhalo] = 0.5 * (
        cos_sg[nhalo - 1 : -nhalo, nhalo - 1 : -nhalo, 7]
        + cos_sg[nhalo : -nhalo + 1, nhalo : -nhalo + 1, 5]
    )
    sina[nhalo:-nhalo, nhalo:-nhalo] = 0.5 * (
        sin_sg[nhalo - 1 : -nhalo, nhalo - 1 : -nhalo, 7]
        + sin_sg[nhalo : -nhalo + 1, nhalo : -nhalo + 1, 5]
    )

    cosa_u[1:-1, :] = 0.5 * (cos_sg[:-1, :, 2] + cos_sg[1:, :, 0])
    sina_u[1:-1, :] = 0.5 * (sin_sg[:-1, :, 2] + sin_sg[1:, :, 0])
    sinu2 = sina_u[1:-1, :] ** 2
    sinu2[sinu2 < tiny_number] = tiny_number
    rsin_u[1:-1, :] = 1.0 / sinu2

    cosa_v[:, 1:-1] = 0.5 * (cos_sg[:, :-1, 3] + cos_sg[:, 1:, 1])
    sina_v[:, 1:-1] = 0.5 * (sin_sg[:, :-1, 3] + sin_sg[:, 1:, 1])
    sinv2 = sina_v[:, 1:-1] ** 2
    sinv2[sinv2 < tiny_number] = tiny_number
    rsin_v[:, 1:-1] = 1.0 / sinv2

    cosa_s = cos_sg[:, :, 4]
    sin2 = sin_sg[:, :, 4] ** 2
    sin2[sin2 < tiny_number] = tiny_number
    rsin2 = 1.0 / sin2

    # fill ghost on cosa_s:
    _fill_halo_corners(cosa_s, big_number, nhalo, tile_partitioner, rank)

    sina2 = sina[nhalo:-nhalo, nhalo:-nhalo] ** 2
    sina2[sina2 < tiny_number] = tiny_number
    rsina = 1.0 / sina2

    # Set special sin values at edges
    if tile_partitioner.on_tile_left(rank):
        rsina[0, :] = big_number
        sina_u_limit = sina_u[nhalo, :]
        sina_u_limit[abs(sina_u_limit) < tiny_number] = tiny_number * np.sign(
            sina_u_limit[abs(sina_u_limit) < tiny_number]
        )
        rsin_u[nhalo, :] = 1.0 / sina_u_limit
    if tile_partitioner.on_tile_right(rank):
        rsina[-1, :] = big_number
        sina_u_limit = sina_u[-nhalo - 1, :]
        sina_u_limit[abs(sina_u_limit) < tiny_number] = tiny_number * np.sign(
            sina_u_limit[abs(sina_u_limit) < tiny_number]
        )
        rsin_u[-nhalo - 1, :] = 1.0 / sina_u_limit
    if tile_partitioner.on_tile_bottom(rank):
        rsina[:, 0] = big_number
        sina_v_limit = sina_v[:, nhalo]
        sina_v_limit[abs(sina_v_limit) < tiny_number] = tiny_number * np.sign(
            sina_v_limit[abs(sina_v_limit) < tiny_number]
        )
        rsin_v[:, nhalo] = 1.0 / sina_v_limit
    if tile_partitioner.on_tile_top(rank):
        rsina[:, -1] = big_number
        sina_v_limit = sina_v[:, -nhalo - 1]
        sina_v_limit[abs(sina_v_limit) < tiny_number] = tiny_number * np.sign(
            sina_v_limit[abs(sina_v_limit) < tiny_number]
        )
        rsin_v[:, -nhalo - 1] = 1.0 / sina_v_limit

    return (
        cosa,
        sina,
        cosa_u,
        cosa_v,
        cosa_s,
        sina_u,
        sina_v,
        rsin_u,
        rsin_v,
        rsina,
        rsin2,
    )


def supergrid_corner_fix(
    cos_sg, sin_sg, nhalo: int, tile_partitioner: TilePartitioner, rank: int
):
    """
    filling the ghost cells overwrites some of the sin_sg
    values along the outward-facing edge of a tile in the corners, which is incorrect.
    This function resolves the issue by filling in the appropriate values
    after the _fill_single_halo_corner call
    """
    big_number = 1.0e8
    tiny_number = 1.0e-8

    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_single_halo_corner(sin_sg, tiny_number, nhalo, "sw")
            _fill_single_halo_corner(cos_sg, big_number, nhalo, "sw")
            _rotate_trig_sg_sw_counterclockwise(sin_sg[:, :, 1], sin_sg[:, :, 2], nhalo)
            _rotate_trig_sg_sw_counterclockwise(cos_sg[:, :, 1], cos_sg[:, :, 2], nhalo)
            _rotate_trig_sg_sw_clockwise(sin_sg[:, :, 0], sin_sg[:, :, 3], nhalo)
            _rotate_trig_sg_sw_clockwise(cos_sg[:, :, 0], cos_sg[:, :, 3], nhalo)
        if tile_partitioner.on_tile_top(rank):
            _fill_single_halo_corner(sin_sg, tiny_number, nhalo, "nw")
            _fill_single_halo_corner(cos_sg, big_number, nhalo, "nw")
            _rotate_trig_sg_nw_counterclockwise(sin_sg[:, :, 0], sin_sg[:, :, 1], nhalo)
            _rotate_trig_sg_nw_counterclockwise(cos_sg[:, :, 0], cos_sg[:, :, 1], nhalo)
            _rotate_trig_sg_nw_clockwise(sin_sg[:, :, 3], sin_sg[:, :, 2], nhalo)
            _rotate_trig_sg_nw_clockwise(cos_sg[:, :, 3], cos_sg[:, :, 2], nhalo)
    if tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):
            _fill_single_halo_corner(sin_sg, tiny_number, nhalo, "se")
            _fill_single_halo_corner(cos_sg, big_number, nhalo, "se")
            _rotate_trig_sg_se_clockwise(sin_sg[:, :, 1], sin_sg[:, :, 0], nhalo)
            _rotate_trig_sg_se_clockwise(cos_sg[:, :, 1], cos_sg[:, :, 0], nhalo)
            _rotate_trig_sg_se_counterclockwise(sin_sg[:, :, 2], sin_sg[:, :, 3], nhalo)
            _rotate_trig_sg_se_counterclockwise(cos_sg[:, :, 2], cos_sg[:, :, 3], nhalo)
        if tile_partitioner.on_tile_top(rank):
            _fill_single_halo_corner(sin_sg, tiny_number, nhalo, "ne")
            _fill_single_halo_corner(cos_sg, big_number, nhalo, "ne")
            _rotate_trig_sg_ne_counterclockwise(sin_sg[:, :, 3], sin_sg[:, :, 0], nhalo)
            _rotate_trig_sg_ne_counterclockwise(cos_sg[:, :, 3], cos_sg[:, :, 0], nhalo)
            _rotate_trig_sg_ne_clockwise(sin_sg[:, :, 2], sin_sg[:, :, 1], nhalo)
            _rotate_trig_sg_ne_clockwise(cos_sg[:, :, 2], cos_sg[:, :, 1], nhalo)


def _rotate_trig_sg_sw_counterclockwise(sg_field_in, sg_field_out, nhalo):
    sg_field_out[nhalo - 1, :nhalo] = sg_field_in[:nhalo, nhalo]


def _rotate_trig_sg_sw_clockwise(sg_field_in, sg_field_out, nhalo):
    sg_field_out[:nhalo, nhalo - 1] = sg_field_in[nhalo, :nhalo]


def _rotate_trig_sg_nw_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[:, ::-1], sg_field_out[:, ::-1], nhalo)


def _rotate_trig_sg_nw_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(
        sg_field_in[:, ::-1], sg_field_out[:, ::-1], nhalo
    )


def _rotate_trig_sg_se_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(sg_field_in[::-1, :], sg_field_out[::-1, :], nhalo)


def _rotate_trig_sg_se_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(
        sg_field_in[::-1, :], sg_field_out[::-1, :], nhalo
    )


def _rotate_trig_sg_ne_counterclockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_counterclockwise(
        sg_field_in[::-1, ::-1], sg_field_out[::-1, ::-1], nhalo
    )


def _rotate_trig_sg_ne_clockwise(sg_field_in, sg_field_out, nhalo):
    _rotate_trig_sg_sw_clockwise(
        sg_field_in[::-1, ::-1], sg_field_out[::-1, ::-1], nhalo
    )


def calculate_divg_del6(
    sin_sg,
    sina_u,
    sina_v,
    dx,
    dy,
    dxc,
    dyc,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
):

    divg_u = sina_v * dyc / dx
    del6_u = sina_v * dx / dyc
    divg_v = sina_u * dxc / dy
    del6_v = sina_u * dy / dxc

    if tile_partitioner.on_tile_bottom(rank):
        divg_u[:, nhalo] = (
            0.5
            * (sin_sg[:, nhalo, 1] + sin_sg[:, nhalo - 1, 3])
            * dyc[:, nhalo]
            / dx[:, nhalo]
        )
        del6_u[:, nhalo] = (
            0.5
            * (sin_sg[:, nhalo, 1] + sin_sg[:, nhalo - 1, 3])
            * dx[:, nhalo]
            / dyc[:, nhalo]
        )
    if tile_partitioner.on_tile_top(rank):
        divg_u[:, -nhalo - 1] = (
            0.5
            * (sin_sg[:, -nhalo, 1] + sin_sg[:, -nhalo - 1, 3])
            * dyc[:, -nhalo - 1]
            / dx[:, -nhalo - 1]
        )
        del6_u[:, -nhalo - 1] = (
            0.5
            * (sin_sg[:, -nhalo, 1] + sin_sg[:, -nhalo - 1, 3])
            * dx[:, -nhalo - 1]
            / dyc[:, -nhalo - 1]
        )
    if tile_partitioner.on_tile_left(rank):
        divg_v[nhalo, :] = (
            0.5
            * (sin_sg[nhalo, :, 0] + sin_sg[nhalo - 1, :, 2])
            * dxc[nhalo, :]
            / dy[nhalo, :]
        )
        del6_v[nhalo, :] = (
            0.5
            * (sin_sg[nhalo, :, 0] + sin_sg[nhalo - 1, :, 2])
            * dy[nhalo, :]
            / dxc[nhalo, :]
        )
    if tile_partitioner.on_tile_right(rank):
        divg_v[-nhalo - 1, :] = (
            0.5
            * (sin_sg[-nhalo, :, 0] + sin_sg[-nhalo - 1, :, 2])
            * dxc[-nhalo - 1, :]
            / dy[-nhalo - 1, :]
        )
        del6_v[-nhalo - 1, :] = (
            0.5
            * (sin_sg[-nhalo, :, 0] + sin_sg[-nhalo - 1, :, 2])
            * dy[-nhalo - 1, :]
            / dxc[-nhalo - 1, :]
        )

    return divg_u, divg_v, del6_u, del6_v


def calculate_grid_z(ec1, ec2, vlon, vlat, np):
    z11 = np.sum(ec1 * vlon, axis=-1)
    z12 = np.sum(ec1 * vlat, axis=-1)
    z21 = np.sum(ec2 * vlon, axis=-1)
    z22 = np.sum(ec2 * vlat, axis=-1)
    return z11, z12, z21, z22


def calculate_grid_a(z11, z12, z21, z22, sin_sg5):
    a11 = 0.5 * z22 / sin_sg5
    a12 = -0.5 * z12 / sin_sg5
    a21 = -0.5 * z21 / sin_sg5
    a22 = 0.5 * z11 / sin_sg5
    return a11, a12, a21, a22


def edge_factors(
    grid_quantity: Quantity,
    agrid,
    grid_type: int,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
    radius: float,
    np,
):
    """
    Creates interpolation factors from the A grid to the B grid on tile edges
    """
    grid = grid_quantity.data[:]
    big_number = 1.0e8
    i_range = grid[nhalo:-nhalo, nhalo:-nhalo].shape[0]
    j_range = grid[nhalo:-nhalo, nhalo:-nhalo].shape[1]
    edge_n = np.zeros(i_range) + big_number
    edge_s = np.zeros(i_range) + big_number
    edge_e = np.zeros(j_range) + big_number
    edge_w = np.zeros(j_range) + big_number
    npx, npy, ndims = tile_partitioner.global_extent(grid_quantity)
    slice_x, slice_y = tile_partitioner.subtile_slice(
        rank, grid_quantity.dims, (npx, npy)
    )
    global_is = nhalo + slice_x.start
    global_js = nhalo + slice_y.start
    global_ie = nhalo + slice_x.stop - 1
    global_je = nhalo + slice_y.stop - 1
    jstart = max(4, global_js) - global_js + nhalo
    jend = min(npy + nhalo - 1, global_je + 2) - global_js + nhalo
    istart = max(4, global_is) - global_is + nhalo
    iend = min(npx + nhalo - 1, global_ie + 2) - global_is + nhalo
    if grid_type < 3:
        if tile_partitioner.on_tile_left(rank):
            edge_w[jstart - nhalo : jend - nhalo] = set_west_edge_factor(
                grid, agrid, nhalo, radius, jstart, jend, np
            )
        if tile_partitioner.on_tile_right(rank):
            edge_e[jstart - nhalo : jend - nhalo] = set_east_edge_factor(
                grid, agrid, nhalo, radius, jstart, jend, np
            )
        if tile_partitioner.on_tile_bottom(rank):
            edge_s[istart - nhalo : iend - nhalo] = set_south_edge_factor(
                grid, agrid, nhalo, radius, istart, iend, np
            )
        if tile_partitioner.on_tile_top(rank):
            edge_n[istart - nhalo : iend - nhalo] = set_north_edge_factor(
                grid, agrid, nhalo, radius, istart, iend, np
            )

    return edge_w[np.newaxis, :], edge_e[np.newaxis, :], edge_s, edge_n


def set_west_edge_factor(grid, agrid, nhalo, radius, jstart, jend, np):
    py0, py1 = lon_lat_midpoint(
        agrid[nhalo - 1, jstart - 1 : jend, 0],
        agrid[nhalo, jstart - 1 : jend, 0],
        agrid[nhalo - 1, jstart - 1 : jend, 1],
        agrid[nhalo, jstart - 1 : jend, 1],
        np,
    )

    d1 = great_circle_distance_lon_lat(
        py0[:-1],
        grid[nhalo, jstart:jend, 0],
        py1[:-1],
        grid[nhalo, jstart:jend, 1],
        radius,
        np,
    )
    d2 = great_circle_distance_lon_lat(
        py0[1:],
        grid[nhalo, jstart:jend, 0],
        py1[1:],
        grid[nhalo, jstart:jend, 1],
        radius,
        np,
    )
    west_edge_factor = d2 / (d1 + d2)
    return west_edge_factor


def set_east_edge_factor(grid, agrid, nhalo, radius, jstart, jend, np):
    return set_west_edge_factor(
        grid[::-1, :, :], agrid[::-1, :, :], nhalo, radius, jstart, jend, np
    )


def set_south_edge_factor(grid, agrid, nhalo, radius, jstart, jend, np):
    return set_west_edge_factor(
        grid.transpose([1, 0, 2]),
        agrid.transpose([1, 0, 2]),
        nhalo,
        radius,
        jstart,
        jend,
        np,
    )


def set_north_edge_factor(grid, agrid, nhalo, radius, jstart, jend, np):
    return set_west_edge_factor(
        grid[:, ::-1, :].transpose([1, 0, 2]),
        agrid[:, ::-1, :].transpose([1, 0, 2]),
        nhalo,
        radius,
        jstart,
        jend,
        np,
    )


def efactor_a2c_v(
    grid_quantity: Quantity,
    agrid,
    grid_type: int,
    nhalo: int,
    tile_partitioner: TilePartitioner,
    rank: int,
    radius: float,
    np,
):
    """
    Creates interpolation factors at tile edges
    for interpolating vectors from A to C grids
    """
    big_number = 1.0e8
    grid = grid_quantity.data[:]
    npx, npy, ndims = tile_partitioner.global_extent(grid_quantity)
    slice_x, slice_y = tile_partitioner.subtile_slice(
        rank, grid_quantity.dims, (npx, npy)
    )
    global_is = nhalo + slice_x.start
    global_js = nhalo + slice_y.start

    if npx != npy:
        raise ValueError("npx must equal npy")
    if npx % 2 == 0:
        raise ValueError("npx must be odd")
    i_midpoint = int((npx - 1) / 2)
    j_midpoint = int((npy - 1) / 2)
    i_indices = np.arange(agrid.shape[0] - nhalo + 1) + global_is - nhalo
    j_indices = np.arange(agrid.shape[1] - nhalo + 1) + global_js - nhalo
    i_selection = i_indices[i_indices <= nhalo + i_midpoint]
    j_selection = j_indices[j_indices <= nhalo + j_midpoint]
    if len(i_selection) > 0:
        im2 = max(i_selection) - global_is
    else:
        im2 = len(i_selection)
    if len(i_selection) == len(i_indices):
        im2 = len(i_selection) - nhalo
    if len(j_selection) > 0:
        jm2 = max(j_selection) - global_js
    else:
        jm2 = len(j_selection)
    if len(j_selection) == len(j_indices):
        jm2 = len(j_selection) - nhalo
    im2 = max(im2, -1)
    jm2 = max(jm2, -1)

    edge_vect_s = np.zeros(grid.shape[0] - 1) + big_number
    edge_vect_n = np.zeros(grid.shape[0] - 1) + big_number
    edge_vect_e = np.zeros(grid.shape[1] - 1) + big_number
    edge_vect_w = np.zeros(grid.shape[1] - 1) + big_number
    if grid_type < 3:
        if tile_partitioner.on_tile_left(rank):
            edge_vect_w[2:-2] = calculate_west_edge_vectors(
                grid, agrid, jm2, nhalo, radius, np
            )
            if tile_partitioner.on_tile_bottom(rank):
                edge_vect_w[nhalo - 1] = edge_vect_w[nhalo]
            if tile_partitioner.on_tile_top(rank):
                edge_vect_w[-nhalo] = edge_vect_w[-nhalo - 1]
        if tile_partitioner.on_tile_right(rank):
            edge_vect_e[2:-2] = calculate_east_edge_vectors(
                grid, agrid, jm2, nhalo, radius, np
            )
            if tile_partitioner.on_tile_bottom(rank):
                edge_vect_e[nhalo - 1] = edge_vect_e[nhalo]
            if tile_partitioner.on_tile_top(rank):
                edge_vect_e[-nhalo] = edge_vect_e[-nhalo - 1]
        if tile_partitioner.on_tile_bottom(rank):
            edge_vect_s[2:-2] = calculate_south_edge_vectors(
                grid, agrid, im2, nhalo, radius, np
            )
            if tile_partitioner.on_tile_left(rank):
                edge_vect_s[nhalo - 1] = edge_vect_s[nhalo]
            if tile_partitioner.on_tile_right(rank):
                edge_vect_s[-nhalo] = edge_vect_s[-nhalo - 1]
        if tile_partitioner.on_tile_top(rank):
            edge_vect_n[2:-2] = calculate_north_edge_vectors(
                grid, agrid, im2, nhalo, radius, np
            )
            if tile_partitioner.on_tile_left(rank):
                edge_vect_n[nhalo - 1] = edge_vect_n[nhalo]
            if tile_partitioner.on_tile_right(rank):
                edge_vect_n[-nhalo] = edge_vect_n[-nhalo - 1]

    return edge_vect_w, edge_vect_e, edge_vect_s, edge_vect_n


def calculate_west_edge_vectors(grid, agrid, jm2, nhalo, radius, np):
    d2 = np.zeros(agrid.shape[0] - 2 * nhalo + 2)
    d1 = np.zeros(agrid.shape[0] - 2 * nhalo + 2)

    py0, py1 = lon_lat_midpoint(
        agrid[nhalo - 1, nhalo - 2 : -nhalo + 2, 0],
        agrid[nhalo, nhalo - 2 : -nhalo + 2, 0],
        agrid[nhalo - 1, nhalo - 2 : -nhalo + 2, 1],
        agrid[nhalo, nhalo - 2 : -nhalo + 2, 1],
        np,
    )

    p20, p21 = lon_lat_midpoint(
        grid[nhalo, nhalo - 2 : -nhalo + 1, 0],
        grid[nhalo, nhalo - 1 : -nhalo + 2, 0],
        grid[nhalo, nhalo - 2 : -nhalo + 1, 1],
        grid[nhalo, nhalo - 1 : -nhalo + 2, 1],
        np,
    )

    py = np.array([py0, py1]).transpose([1, 0])
    p2 = np.array([p20, p21]).transpose([1, 0])

    d1[: jm2 + 1] = great_circle_distance_lon_lat(
        py[1 : jm2 + 2, 0],
        p2[1 : jm2 + 2, 0],
        py[1 : jm2 + 2, 1],
        p2[1 : jm2 + 2, 1],
        radius,
        np,
    )
    d2[: jm2 + 1] = great_circle_distance_lon_lat(
        py[2 : jm2 + 3, 0],
        p2[1 : jm2 + 2, 0],
        py[2 : jm2 + 3, 1],
        p2[1 : jm2 + 2, 1],
        radius,
        np,
    )
    d1[jm2 + 1 :] = great_circle_distance_lon_lat(
        py[jm2 + 2 : -1, 0],
        p2[jm2 + 2 : -1, 0],
        py[jm2 + 2 : -1, 1],
        p2[jm2 + 2 : -1, 1],
        radius,
        np,
    )
    d2[jm2 + 1 :] = great_circle_distance_lon_lat(
        py[jm2 + 1 : -2, 0],
        p2[jm2 + 2 : -1, 0],
        py[jm2 + 1 : -2, 1],
        p2[jm2 + 2 : -1, 1],
        radius,
        np,
    )

    return d1 / (d2 + d1)


def calculate_east_edge_vectors(grid, agrid, jm2, nhalo, radius, np):
    return calculate_west_edge_vectors(
        grid[::-1, :, :], agrid[::-1, :, :], jm2, nhalo, radius, np
    )


def calculate_south_edge_vectors(grid, agrid, im2, nhalo, radius, np):
    return calculate_west_edge_vectors(
        grid.transpose([1, 0, 2]), agrid.transpose([1, 0, 2]), im2, nhalo, radius, np
    )


def calculate_north_edge_vectors(grid, agrid, jm2, nhalo, radius, np):
    return calculate_west_edge_vectors(
        grid[:, ::-1, :].transpose([1, 0, 2]),
        agrid[:, ::-1, :].transpose([1, 0, 2]),
        jm2,
        nhalo,
        radius,
        np,
    )


def unit_vector_lonlat(grid, np):
    """
    Calculates the cartesian unit vectors for each point on a lat/lon grid
    """

    sin_lon = np.sin(grid[:, :, 0])
    cos_lon = np.cos(grid[:, :, 0])
    sin_lat = np.sin(grid[:, :, 1])
    cos_lat = np.cos(grid[:, :, 1])

    unit_lon = np.array([-sin_lon, cos_lon, np.zeros(grid[:, :, 0].shape)]).transpose(
        [1, 2, 0]
    )
    unit_lat = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat]).transpose(
        [1, 2, 0]
    )

    return unit_lon, unit_lat


def _fill_halo_corners(field, value: float, nhalo: int, tile_partitioner, rank):
    """
    Fills a tile halo corners (ghost cells) of a field
    with a set value along the first 2 axes
    """
    if tile_partitioner.on_tile_left(rank):
        if tile_partitioner.on_tile_bottom(rank):  # SW corner
            field[:nhalo, :nhalo] = value
        if tile_partitioner.on_tile_top(rank):  # NW corner
            field[:nhalo, -nhalo:] = value
    if tile_partitioner.on_tile_right(rank):
        if tile_partitioner.on_tile_bottom(rank):  # SE corner
            field[-nhalo:, :nhalo] = value
        if tile_partitioner.on_tile_top(rank):
            field[-nhalo:, -nhalo:] = value  # NE corner


def _fill_single_halo_corner(field, value: float, nhalo: int, corner: str):
    """
    Fills a tile halo corner (ghost cells) of a field
    with a set value along the first 2 axes
    Args:
        field: the field to fill in, assumed to have x and y as the first 2 dimensions
        value: the value to fill
        nhalo: the number of halo points in the field
        corner: which corner to fill
    """
    if (corner == "sw") or (corner == "southwest"):
        field[:nhalo, :nhalo] = value
    elif (corner == "nw") or (corner == "northwest"):
        field[:nhalo, -nhalo:] = value
    elif (corner == "se") or (corner == "southeast"):
        field[-nhalo:, :nhalo] = value
    elif (corner == "ne") or (corner == "northeast"):
        field[-nhalo:, -nhalo:] = value
    else:
        raise ValueError("fill ghost requires a corner to be one of: sw, se, nw, ne")
