#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import numpy as np
import gt4py.gtscript as gtscript
from gt4py.gtscript import computation, interval, PARALLEL
import fv3._config as spec
import fv3.stencils.copy_stencil as cp

# comact 4-pt cubic interpolation
c1 = 2.0 / 3.0
c2 = -1.0 / 6.0
d1 = 0.375
d2 = -1.0 / 24.0
# PPM volume mean form
b1 = 7.0 / 12.0
b2 = -1.0 / 12.0
# 4-pt Lagrange interpolation
a1 = 9.0 / 16.0
a2 = -1.0 / 16.0
sd = utils.sd


def grid():
    return spec.grid


@utils.stencil()
def ppm_volume_mean_x(qin: sd, qx: sd):
    with computation(PARALLEL), interval(...):
        qx[0, 0, 0] = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)


@utils.stencil()
def ppm_volume_mean_y(qin: sd, qy: sd):
    with computation(PARALLEL), interval(...):
        qy[0, 0, 0] = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)


@utils.stencil()
def lagrange_interpolation_y(qx: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout[0, 0, 0] = a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@utils.stencil()
def lagrange_interpolation_x(qy: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout[0, 0, 0] = a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


@utils.stencil()
def cubic_interpolation_south(qx: sd, qout: sd, qxx: sd):
    with computation(PARALLEL), interval(...):
        qxx[0, 0, 0] = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx[0, 1, 0])


@utils.stencil()
def cubic_interpolation_north(qx: sd, qout: sd, qxx: sd):
    with computation(PARALLEL), interval(...):
        qxx[0, 0, 0] = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx[0, -1, 0])


@utils.stencil()
def cubic_interpolation_west(qy: sd, qout: sd, qyy: sd):
    with computation(PARALLEL), interval(...):
        qyy[0, 0, 0] = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy[1, 0, 0])


@utils.stencil()
def cubic_interpolation_east(qy: sd, qout: sd, qyy: sd):
    with computation(PARALLEL), interval(...):
        qyy[0, 0, 0] = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy[-1, 0, 0])


@utils.stencil()
def qout_avg(qxx: sd, qyy: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout[0, 0, 0] = 0.5 * (qxx + qyy)


@utils.stencil()
def vort_adjust(qxx: sd, qyy: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout[0, 0, 0] = 0.5 * (qxx + qyy)


# @utils.stencil()
# def x_edge_q2_west(qin: sd, dxa: sd, q2: sd):
#    with computation(PARALLEL), interval(...):
#        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0, 0]) / (dxa[-1, 0, 0] + dxa)

# @utils.stencil()
# def x_edge_qout_west_q2(edge_w: sd, q2: sd, qout: sd):
#    with computation(PARALLEL), interval(...):
#        qout = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2
@utils.stencil()
def qout_x_edge(qin: sd, dxa: sd, edge_w: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0, 0]) / (dxa[-1, 0, 0] + dxa)
        qout[0, 0, 0] = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


@utils.stencil()
def qout_y_edge(qin: sd, dya: sd, edge_s: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1, 0]) / (dya[0, -1, 0] + dya)
        qout[0, 0, 0] = edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


@utils.stencil()
def qx_edge_west(qin: sd, dxa: sd, qx: sd):
    with computation(PARALLEL), interval(...):
        g_in = dxa[1, 0, 0] / dxa
        g_ou = dxa[-2, 0, 0] / dxa[-1, 0, 0]
        qx[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
        )
        # This does not work, due to access of qx that is changing above
        # qx[1, 0, 0] = (3.0 * (g_in * qin + qin[1, 0, 0]) - (g_in * qx + qx[2, 0, 0])) / (2.0 + 2.0 * g_in)


@utils.stencil()
def qx_edge_west2(qin: sd, dxa: sd, qx: sd):
    with computation(PARALLEL), interval(...):
        g_in = dxa / dxa[-1, 0, 0]
        qx[0, 0, 0] = (
            3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx[-1, 0, 0] + qx[1, 0, 0])
        ) / (2.0 + 2.0 * g_in)


@utils.stencil()
def qx_edge_east(qin: sd, dxa: sd, qx: sd):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-2, 0, 0] / dxa[-1, 0, 0]
        g_ou = dxa[1, 0, 0] / dxa
        qx[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
        )


@utils.stencil()
def qx_edge_east2(qin: sd, dxa: sd, qx: sd):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-1, 0, 0] / dxa
        qx[0, 0, 0] = (
            3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx[1, 0, 0] + qx[-1, 0, 0])
        ) / (2.0 + 2.0 * g_in)


@utils.stencil()
def qy_edge_south(qin: sd, dya: sd, qy: sd):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, 1, 0] / dya
        g_ou = dya[0, -2, 0] / dya[0, -1, 0]
        qy[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
        )


@utils.stencil()
def qy_edge_south2(qin: sd, dya: sd, qy: sd):
    with computation(PARALLEL), interval(...):
        g_in = dya / dya[0, -1, 0]
        qy[0, 0, 0] = (
            3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy[0, -1, 0] + qy[0, 1, 0])
        ) / (2.0 + 2.0 * g_in)


@utils.stencil()
def qy_edge_north(qin: sd, dya: sd, qy: sd):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -2, 0] / dya[0, -1, 0]
        g_ou = dya[0, 1, 0] / dya
        qy[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
        )


@utils.stencil()
def qy_edge_north2(qin: sd, dya: sd, qy: sd):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -1, 0] / dya
        qy[0, 0, 0] = (
            3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy[0, 1, 0] + qy[0, -1, 0])
        ) / (2.0 + 2.0 * g_in)


# # TODO: all of these offsets are either 0,1 or -1, -2. Totally should be able to consolidate
# implemented below in a re-definition, can this be deleted?
# def ec1_offsets_dir(corner, lower_direction):
#     if lower_direction in corner:
#         a = 0
#         b = 1
#     else:
#         a = -1
#         b = -2
#     return i1a, i1b


def ec1_offsets(corner):
    i1a, i1b = ec1_offsets_dir(corner, "w")
    j1a, j1b = ec1_offsets_dir(corner, "s")
    return i1a, i1b, j1a, j1b


def ec1_offsets_dir(corner, lower_direction):
    if lower_direction in corner:
        a = 0
        b = 1
    else:
        a = -1
        b = -2
    return (
        a,
        b,
    )


def ec2_offsets_dirs(corner, lower_direction, other_direction):
    if lower_direction in corner or other_direction in corner:
        a = -1
        b = -2
    else:
        a = 0
        b = 1
    return a, b


def ec2_offsets(corner):
    i2a, i2b = ec2_offsets_dirs(corner, "s", "w")
    j2a, j2b = ec2_offsets_dirs(corner, "e", "n")
    return i2a, i2b, j2a, j2b


def ec3_offsets_dirs(corner, lower_direction, other_direction):
    if lower_direction in corner or other_direction in corner:
        a = 0
        b = 1
    else:
        a = -1
        b = -2
    return a, b


def ec3_offsets(corner):
    i3a, i3b = ec3_offsets_dirs(corner, "s", "w")
    j3a, j3b = ec3_offsets_dirs(corner, "e", "n")
    return i3a, i3b, j3a, j3b


# TODO: put into stencil?
def extrapolate_corner_qout(qin, qout, i, j, corner):
    if not getattr(grid(), corner + "_corner"):
        return
    bgrid = np.stack((grid().bgrid1[:, :, 0], grid().bgrid2[:, :, 0]), axis=2)
    agrid = np.stack((grid().agrid1[:, :, 0], grid().agrid2[:, :, 0]), axis=2)
    p0 = bgrid[i, j, :]
    # TODO: - please simplify
    i1a, i1b, j1a, j1b = ec1_offsets(corner)
    i2a, i2b, j2a, j2b = ec2_offsets(corner)
    i3a, i3b, j3a, j3b = ec3_offsets(corner)
    ec1 = utils.extrap_corner(
        p0,
        agrid[i + i1a, j + j1a, :],
        agrid[i + i1b, j + j1b, :],
        qin[i + i1a, j + j1a, :],
        qin[i + i1b, j + j1b, :],
    )
    ec2 = utils.extrap_corner(
        p0,
        agrid[i + i2a, j + j2a, :],
        agrid[i + i2b, j + j2b, :],
        qin[i + i2a, j + j2a, :],
        qin[i + i2b, j + j2b, :],
    )
    ec3 = utils.extrap_corner(
        p0,
        agrid[i + i3a, j + j3a, :],
        agrid[i + i3b, j + j3b, :],
        qin[i + i3a, j + j3a, :],
        qin[i + i3b, j + j3b, :],
    )
    r3 = 1.0 / 3.0
    qout[i, j, :] = (ec1 + ec2 + ec3) * r3


def extrapolate_corners(qin, qout):
    # qout corners, 3 way extrapolation
    extrapolate_corner_qout(qin, qout, grid().is_, grid().js, "sw")
    extrapolate_corner_qout(qin, qout, grid().ie + 1, grid().js, "se")
    extrapolate_corner_qout(qin, qout, grid().ie + 1, grid().je + 1, "ne")
    extrapolate_corner_qout(qin, qout, grid().is_, grid().je + 1, "nw")


def compute_qout_edges(qin, qout):
    compute_qout_x_edges(qin, qout)
    compute_qout_y_edges(qin, qout)


def compute_qout_x_edges(qin, qout):
    # qout bounds
    # avoid running west/east computation on south/north tile edges, since they'll be overwritten.
    js2 = grid().js + 1 if grid().south_edge else grid().js
    je1 = grid().je if grid().north_edge else grid().je + 1
    dj2 = je1 - js2 + 1
    if grid().west_edge:
        qout_x_edge(
            qin,
            grid().dxa,
            grid().edge_w,
            qout,
            origin=(grid().is_, js2, 0),
            domain=(1, dj2, grid().npz),
        )
    if grid().east_edge:
        qout_x_edge(
            qin,
            grid().dxa,
            grid().edge_e,
            qout,
            origin=(grid().ie + 1, js2, 0),
            domain=(1, dj2, grid().npz),
        )


def compute_qout_y_edges(qin, qout):
    # avoid running south/north computation on west/east tile edges, since they'll be overwritten.
    is2 = grid().is_ + 1 if grid().west_edge else grid().is_
    ie1 = grid().ie if grid().east_edge else grid().ie + 1
    di2 = ie1 - is2 + 1
    if grid().south_edge:
        qout_y_edge(
            qin,
            grid().dya,
            grid().edge_s,
            qout,
            origin=(is2, grid().js, 0),
            domain=(di2, 1, grid().npz),
        )
    if grid().north_edge:
        qout_y_edge(
            qin,
            grid().dya,
            grid().edge_n,
            qout,
            origin=(is2, grid().je + 1, 0),
            domain=(di2, 1, grid().npz),
        )


def compute_qx(qin, qout):
    qx = utils.make_storage_from_shape(qin.shape, origin=(grid().is_, grid().jsd, 0))
    # qx bounds
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js if grid().south_edge else grid().js - 2
    je = grid().je if grid().north_edge else grid().je + 2
    is_ = grid().is_ + 2 if grid().west_edge else grid().is_
    ie = grid().ie - 1 if grid().east_edge else grid().ie + 1
    dj = je - js + 1
    # qx interior
    ppm_volume_mean_x(
        qin, qx, origin=(is_, js, 0), domain=(ie - is_ + 1, dj, grid().npz)
    )

    # qx edges
    if grid().west_edge:
        qx_edge_west(
            qin, grid().dxa, qx, origin=(grid().is_, js, 0), domain=(1, dj, grid().npz)
        )
        qx_edge_west2(
            qin,
            grid().dxa,
            qx,
            origin=(grid().is_ + 1, js, 0),
            domain=(1, dj, grid().npz),
        )
    if grid().east_edge:
        qx_edge_east(
            qin,
            grid().dxa,
            qx,
            origin=(grid().ie + 1, js, 0),
            domain=(1, dj, grid().npz),
        )
        qx_edge_east2(
            qin, grid().dxa, qx, origin=(grid().ie, js, 0), domain=(1, dj, grid().npz)
        )
    return qx


def compute_qy(qin, qout):
    qy = utils.make_storage_from_shape(qin.shape, origin=(grid().isd, grid().js, 0))
    # qy bounds
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 2 if grid().south_edge else grid().js
    je = grid().je - 1 if grid().north_edge else grid().je + 1
    is_ = grid().is_ if grid().west_edge else grid().is_ - 2
    ie = grid().ie if grid().east_edge else grid().ie + 2
    di = ie - is_ + 1
    # qy interior
    ppm_volume_mean_y(
        qin, qy, origin=(is_, js, 0), domain=(di, je - js + 1, grid().npz)
    )
    # qy edges
    if grid().south_edge:
        qy_edge_south(
            qin, grid().dya, qy, origin=(is_, grid().js, 0), domain=(di, 1, grid().npz)
        )
        qy_edge_south2(
            qin,
            grid().dya,
            qy,
            origin=(is_, grid().js + 1, 0),
            domain=(di, 1, grid().npz),
        )
    if grid().north_edge:
        qy_edge_north(
            qin,
            grid().dya,
            qy,
            origin=(is_, grid().je + 1, 0),
            domain=(di, 1, grid().npz),
        )
        qy_edge_north2(
            qin, grid().dya, qy, origin=(is_, grid().je, 0), domain=(di, 1, grid().npz)
        )
    return qy


def compute_qxx(qx, qout):
    qxx = utils.make_storage_from_shape(qx.shape, origin=grid().default_origin())
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 2 if grid().south_edge else grid().js
    je = grid().je - 1 if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 1 if grid().west_edge else grid().is_
    ie = grid().ie if grid().east_edge else grid().ie + 1
    di = ie - is_ + 1
    lagrange_interpolation_y(
        qx, qxx, origin=(is_, js, 0), domain=(di, je - js + 1, grid().npz)
    )
    if grid().south_edge:
        cubic_interpolation_south(
            qx, qout, qxx, origin=(is_, grid().js + 1, 0), domain=(di, 1, grid().npz)
        )
    if grid().north_edge:
        cubic_interpolation_north(
            qx, qout, qxx, origin=(is_, grid().je, 0), domain=(di, 1, grid().npz)
        )
    return qxx


def compute_qyy(qy, qout):
    qyy = utils.make_storage_from_shape(qy.shape, origin=grid().default_origin())
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 1 if grid().south_edge else grid().js
    je = grid().je if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 2 if grid().west_edge else grid().is_
    ie = grid().ie - 1 if grid().east_edge else grid().ie + 1
    dj = je - js + 1
    lagrange_interpolation_x(
        qy, qyy, origin=(is_, js, 0), domain=(ie - is_ + 1, dj, grid().npz)
    )
    if grid().west_edge:
        cubic_interpolation_west(
            qy, qout, qyy, origin=(grid().is_ + 1, js, 0), domain=(1, dj, grid().npz)
        )
    if grid().east_edge:
        cubic_interpolation_east(
            qy, qout, qyy, origin=(grid().ie, js, 0), domain=(1, dj, grid().npz)
        )
    return qyy


def compute_qout(qxx, qyy, qout):
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 1 if grid().south_edge else grid().js
    je = grid().je if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 1 if grid().west_edge else grid().is_
    ie = grid().ie if grid().east_edge else grid().ie + 1
    qout_avg(
        qxx,
        qyy,
        qout,
        origin=(is_, js, 0),
        domain=(ie - is_ + 1, je - js + 1, grid().npz),
    )


def compute(qin, qout, replace=False):
    extrapolate_corners(qin, qout)
    if spec.namelist["grid_type"] < 3:
        compute_qout_edges(qin, qout)
        qx = compute_qx(qin, qout)
        qy = compute_qy(qin, qout)
        qxx = compute_qxx(qx, qout)
        qyy = compute_qyy(qy, qout)
        compute_qout(qxx, qyy, qout)
        if replace:
            cp.copy_stencil(
                qout,
                qin,
                origin=grid().compute_origin(),
                domain=grid().domain_shape_compute_buffer_2d(),
            )
    else:
        raise Exception("grid_type >= 3 is not implemented")
