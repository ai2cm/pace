from pace.util.constants import PI, RADIUS


__all__ = ["mirror_grid"]

RIGHT_HAND_GRID = False


def mirror_grid(
    mirror_data,
    tile_index,
    npx,
    npy,
    x_subtile_width,
    y_subtile_width,
    global_is,
    global_js,
    ng,
    np,
    right_hand_grid,
):
    istart = ng
    iend = ng + x_subtile_width
    jstart = ng
    jend = ng + y_subtile_width
    x_center_tile = (
        global_is <= ng + (npx - 1) / 2
        and global_is + x_subtile_width > ng + (npx - 1) / 2
    )
    y_center_tile = (
        global_js <= ng + (npy - 1) / 2
        and global_js + y_subtile_width > ng + (npy - 1) / 2
    )

    i_mid = npx // 2 - global_is + istart
    j_mid = npy // 2 - global_js + jstart

    # first fix base region
    for j in range(jstart, jend + 1):
        for i in range(istart, iend + 1):

            iend_domain = iend - 1 + ng
            jend_domain = jend - 1 + ng
            x1 = np.multiply(
                0.25,
                np.abs(mirror_data["local"][i, j, 0])
                + np.abs(mirror_data["east-west"][iend_domain - i, j, 0])
                + np.abs(mirror_data["north-south"][i, jend_domain - j, 0])
                + np.abs(mirror_data["diagonal"][iend_domain - i, jend_domain - j, 0]),
            )
            mirror_data["local"][i, j, 0] = np.copysign(
                x1, mirror_data["local"][i, j, 0]
            )

            y1 = np.multiply(
                0.25,
                np.abs(mirror_data["local"][i, j, 1])
                + np.abs(mirror_data["east-west"][iend_domain - i, j, 1])
                + np.abs(mirror_data["north-south"][i, jend_domain - j, 1])
                + np.abs(mirror_data["diagonal"][iend_domain - i, jend_domain - j, 1]),
            )

            mirror_data["local"][i, j, 1] = np.copysign(
                y1, mirror_data["local"][i, j, 1]
            )

            # force dateline/greenwich-meridion consistency
            if npx % 2 != 0:
                if x_center_tile and i == ng + i_mid:
                    mirror_data["local"][i, j, 0] = 0.0
                    mirror_data["north-south"][i, -(j + 1), 0] = 0

    if tile_index > 0:

        for j in range(jstart, jend + 1):
            x1 = mirror_data["local"][istart : iend + 1, j, 0]
            y1 = mirror_data["local"][istart : iend + 1, j, 1]
            z1 = np.add(RADIUS, np.multiply(0.0, x1))

            if tile_index == 1:
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
            elif tile_index == 2:
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
                    if (
                        j == ng + j_mid
                        and x_center_tile
                        and y_center_tile
                        and i_mid == j_mid
                    ):
                        x2[i_mid] = 0.0
                        y2[i_mid] = PI / 2.0
                    if j == ng + j_mid and y_center_tile:
                        if x_center_tile:
                            x2[: i_mid + 1] = 0.0
                            x2[i_mid + 1 :] = PI
                        elif global_is + i_mid < ng + (npx - 1) / 2:
                            x2[:] = 0.0
                        elif global_is + i_mid > ng + (npx - 1) / 2:
                            x2[:] = PI
            elif tile_index == 3:
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
                    if j == ng + j_mid and y_center_tile:
                        x2[:] = PI
            elif tile_index == 4:
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
            elif tile_index == 5:
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
                    if (
                        i == ng + i_mid
                        and x_center_tile
                        and y_center_tile
                        and i_mid == j_mid
                    ):
                        x2[i_mid] = 0.0
                        y2[i_mid] = -PI / 2.0
                    if global_js + j_mid > ng + (npy - 1) / 2 and x_center_tile:
                        x2[i_mid] = 0.0
                    elif global_js + j_mid < ng + (npy - 1) / 2 and x_center_tile:
                        x2[i_mid] = PI

            mirror_data["local"][istart : iend + 1, j, 0] = x2
            mirror_data["local"][istart : iend + 1, j, 1] = y2


def _rot_3d(axis, p, angle, np, right_hand_grid, degrees=False, convert=False):

    if convert:
        p1 = _spherical_to_cartesian(p, np, right_hand_grid)
    else:
        p1 = p

    if degrees:
        angle = np.deg2rad(angle)

    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 1:
        x2 = p1[0]
        y2 = c * p1[1] + s * p1[2]
        z2 = -s * p1[1] + c * p1[2]
    elif axis == 2:
        x2 = c * p1[0] - s * p1[2]
        y2 = p1[1]
        z2 = s * p1[0] + c * p1[2]
    elif axis == 3:
        x2 = c * p1[0] + s * p1[1]
        y2 = -s * p1[0] + c * p1[1]
        z2 = p1[2]
    else:
        assert False, "axis must be in [1,2,3]"

    if convert:
        p2 = _cartesian_to_spherical([x2, y2, z2], np, right_hand_grid)
    else:
        p2 = [x2, y2, z2]

    return p2


def _spherical_to_cartesian(p, np, right_hand_grid):
    lon, lat, r = p
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    if right_hand_grid:
        z = r * np.sin(lat)
    else:
        z = -r * np.sin(lat)
    return [x, y, z]


def _cartesian_to_spherical(p, np, right_hand_grid):
    x, y, z = p
    r = np.sqrt(x * x + y * y + z * z)
    lon = np.where(np.abs(x) + np.abs(y) < 1.0e-10, 0.0, np.arctan2(y, x))
    if right_hand_grid:
        lat = np.arcsin(z / r)
    else:
        lat = np.arccos(z / r) - PI / 2.0
    return [lon, lat, r]


def set_halo_nan(grid, ng: int, np):
    grid[:ng, :, :] = np.nan  # west edge
    grid[:, :ng, :] = np.nan  # south edge
    grid[-ng:, :, :] = np.nan  # east edge
    grid[:, -ng:, :] = np.nan  # north edge
    return grid
