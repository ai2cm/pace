#!/usr/bin/env python3
import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.basic_operations import copy_stencil


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


@gtscript.function
def lagrange_y_func(qx):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@utils.stencil()
def lagrange_interpolation_y(qx: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func(qx)


@gtscript.function
def lagrange_x_func(qy):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


@utils.stencil()
def lagrange_interpolation_x(qy: sd, qout: sd):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func(qy)


@utils.stencil()
def cubic_interpolation_south(qx: sd, qout: sd, qxx: sd):
    with computation(PARALLEL), interval(...):
        qxx0 = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx0[0, 1, 0])


@utils.stencil()
def cubic_interpolation_north(qx: sd, qout: sd, qxx: sd):
    with computation(PARALLEL), interval(...):
        qxx0 = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx0[0, -1, 0])


@utils.stencil()
def cubic_interpolation_west(qy: sd, qout: sd, qyy: sd):
    with computation(PARALLEL), interval(...):
        qyy0 = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy0[1, 0, 0])


@utils.stencil()
def cubic_interpolation_east(qy: sd, qout: sd, qyy: sd):
    with computation(PARALLEL), interval(...):
        qyy0 = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy0[-1, 0, 0])


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
        qx0 = qx
        qx = (
            3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx0[-1, 0, 0] + qx0[1, 0, 0])
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
        qx0 = qx
        qx = (
            3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx0[1, 0, 0] + qx0[-1, 0, 0])
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
        qy0 = qy
        qy = (
            3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy0[0, -1, 0] + qy0[0, 1, 0])
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
        qy0 = qy
        qy = (
            3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy0[0, 1, 0] + qy0[0, -1, 0])
        ) / (2.0 + 2.0 * g_in)


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
    return (a, b)


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
def extrapolate_corner_qout(qin, qout, i, j, kstart, nk, corner):
    if not getattr(grid(), corner + "_corner"):
        return
    kslice = slice(kstart, kstart + nk)
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
        qin[i + i1a, j + j1a, kslice],
        qin[i + i1b, j + j1b, kslice],
    )
    ec2 = utils.extrap_corner(
        p0,
        agrid[i + i2a, j + j2a, :],
        agrid[i + i2b, j + j2b, :],
        qin[i + i2a, j + j2a, kslice],
        qin[i + i2b, j + j2b, kslice],
    )
    ec3 = utils.extrap_corner(
        p0,
        agrid[i + i3a, j + j3a, :],
        agrid[i + i3b, j + j3b, :],
        qin[i + i3a, j + j3a, kslice],
        qin[i + i3b, j + j3b, kslice],
    )
    r3 = 1.0 / 3.0
    qout[i, j, kslice] = (ec1 + ec2 + ec3) * r3


def extrapolate_corners(qin, qout, kstart, nk):
    # qout corners, 3 way extrapolation
    extrapolate_corner_qout(qin, qout, grid().is_, grid().js, kstart, nk, "sw")
    extrapolate_corner_qout(qin, qout, grid().ie + 1, grid().js, kstart, nk, "se")
    extrapolate_corner_qout(qin, qout, grid().ie + 1, grid().je + 1, kstart, nk, "ne")
    extrapolate_corner_qout(qin, qout, grid().is_, grid().je + 1, kstart, nk, "nw")


def compute_qout_edges(qin, qout, kstart, nk):
    compute_qout_x_edges(qin, qout, kstart, nk)
    compute_qout_y_edges(qin, qout, kstart, nk)


def compute_qout_x_edges(qin, qout, kstart, nk):
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
            origin=(grid().is_, js2, kstart),
            domain=(1, dj2, nk),
        )
    if grid().east_edge:
        qout_x_edge(
            qin,
            grid().dxa,
            grid().edge_e,
            qout,
            origin=(grid().ie + 1, js2, kstart),
            domain=(1, dj2, nk),
        )


def compute_qout_y_edges(qin, qout, kstart, nk):
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
            origin=(is2, grid().js, kstart),
            domain=(di2, 1, nk),
        )
    if grid().north_edge:
        qout_y_edge(
            qin,
            grid().dya,
            grid().edge_n,
            qout,
            origin=(is2, grid().je + 1, kstart),
            domain=(di2, 1, nk),
        )


def compute_qx(qin, qout, kstart, nk):
    qx = utils.make_storage_from_shape(
        qin.shape, origin=(grid().is_, grid().jsd, kstart)
    )
    # qx bounds
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js if grid().south_edge else grid().js - 2
    je = grid().je if grid().north_edge else grid().je + 2
    is_ = grid().is_ + 2 if grid().west_edge else grid().is_
    ie = grid().ie - 1 if grid().east_edge else grid().ie + 1
    dj = je - js + 1
    # qx interior
    ppm_volume_mean_x(qin, qx, origin=(is_, js, kstart), domain=(ie - is_ + 1, dj, nk))

    # qx edges
    if grid().west_edge:
        qx_edge_west(
            qin, grid().dxa, qx, origin=(grid().is_, js, kstart), domain=(1, dj, nk)
        )
        qx_edge_west2(
            qin, grid().dxa, qx, origin=(grid().is_ + 1, js, kstart), domain=(1, dj, nk)
        )
    if grid().east_edge:
        qx_edge_east(
            qin, grid().dxa, qx, origin=(grid().ie + 1, js, kstart), domain=(1, dj, nk)
        )
        qx_edge_east2(
            qin, grid().dxa, qx, origin=(grid().ie, js, kstart), domain=(1, dj, nk)
        )
    return qx


def compute_qy(qin, qout, kstart, nk):
    qy = utils.make_storage_from_shape(
        qin.shape, origin=(grid().isd, grid().js, kstart)
    )
    # qy bounds
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 2 if grid().south_edge else grid().js
    je = grid().je - 1 if grid().north_edge else grid().je + 1
    is_ = grid().is_ if grid().west_edge else grid().is_ - 2
    ie = grid().ie if grid().east_edge else grid().ie + 2
    di = ie - is_ + 1
    # qy interior
    ppm_volume_mean_y(qin, qy, origin=(is_, js, kstart), domain=(di, je - js + 1, nk))
    # qy edges
    if grid().south_edge:
        qy_edge_south(
            qin, grid().dya, qy, origin=(is_, grid().js, kstart), domain=(di, 1, nk)
        )
        qy_edge_south2(
            qin, grid().dya, qy, origin=(is_, grid().js + 1, kstart), domain=(di, 1, nk)
        )
    if grid().north_edge:
        qy_edge_north(
            qin, grid().dya, qy, origin=(is_, grid().je + 1, kstart), domain=(di, 1, nk)
        )
        qy_edge_north2(
            qin, grid().dya, qy, origin=(is_, grid().je, kstart), domain=(di, 1, nk)
        )
    return qy


def compute_qxx(qx, qout, kstart, nk):
    qxx = utils.make_storage_from_shape(qx.shape, origin=grid().default_origin())
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 2 if grid().south_edge else grid().js
    je = grid().je - 1 if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 1 if grid().west_edge else grid().is_
    ie = grid().ie if grid().east_edge else grid().ie + 1
    di = ie - is_ + 1
    lagrange_interpolation_y(
        qx, qxx, origin=(is_, js, kstart), domain=(di, je - js + 1, nk)
    )
    if grid().south_edge:
        cubic_interpolation_south(
            qx, qout, qxx, origin=(is_, grid().js + 1, kstart), domain=(di, 1, nk)
        )
    if grid().north_edge:
        cubic_interpolation_north(
            qx, qout, qxx, origin=(is_, grid().je, kstart), domain=(di, 1, nk)
        )
    return qxx


def compute_qyy(qy, qout, kstart, nk):
    qyy = utils.make_storage_from_shape(qy.shape, origin=grid().default_origin())
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 1 if grid().south_edge else grid().js
    je = grid().je if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 2 if grid().west_edge else grid().is_
    ie = grid().ie - 1 if grid().east_edge else grid().ie + 1
    dj = je - js + 1
    lagrange_interpolation_x(
        qy, qyy, origin=(is_, js, kstart), domain=(ie - is_ + 1, dj, nk)
    )
    if grid().west_edge:
        cubic_interpolation_west(
            qy, qout, qyy, origin=(grid().is_ + 1, js, kstart), domain=(1, dj, nk)
        )
    if grid().east_edge:
        cubic_interpolation_east(
            qy, qout, qyy, origin=(grid().ie, js, kstart), domain=(1, dj, nk)
        )
    return qyy


def compute_qout(qxx, qyy, qout, kstart, nk):
    # avoid running center-domain computation on tile edges, since they'll be overwritten.
    js = grid().js + 1 if grid().south_edge else grid().js
    je = grid().je if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 1 if grid().west_edge else grid().is_
    ie = grid().ie if grid().east_edge else grid().ie + 1
    qout_avg(
        qxx, qyy, qout, origin=(is_, js, kstart), domain=(ie - is_ + 1, je - js + 1, nk)
    )


def compute(qin, qout, kstart=0, nk=None, replace=False):
    if nk == None:
        nk = grid().npz - kstart
    extrapolate_corners(qin, qout, kstart, nk)
    if spec.namelist.grid_type < 3:
        compute_qout_edges(qin, qout, kstart, nk)
        qx = compute_qx(qin, qout, kstart, nk)
        qy = compute_qy(qin, qout, kstart, nk)
        qxx = compute_qxx(qx, qout, kstart, nk)
        qyy = compute_qyy(qy, qout, kstart, nk)
        compute_qout(qxx, qyy, qout, kstart, nk)
        if replace:
            copy_stencil(
                qout,
                qin,
                origin=(grid().is_, grid().js, kstart),
                domain=(grid().ie - grid().is_ + 2, grid().je - grid().js + 2, nk),
            )
    else:
        raise Exception("grid_type >= 3 is not implemented")
