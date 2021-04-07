import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, asin, computation, cos, interval, sin, sqrt

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.utils.typing import FloatField, FloatFieldI, FloatFieldIJ


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


def grid():
    return spec.grid


@gtscript.function
def great_circle_dist(p1a, p1b, p2a, p2b):
    tb = sin((p1b - p2b) / 2.0) ** 2.0
    ta = sin((p1a - p2a) / 2.0) ** 2.0
    return asin(sqrt(tb + cos(p1b) * cos(p2b) * ta)) * 2.0


@gtscript.function
def extrap_corner(
    p0a,
    p0b,
    p1a,
    p1b,
    p2a,
    p2b,
    qa,
    qb,
):
    x1 = great_circle_dist(p1a, p1b, p0a, p0b)
    x2 = great_circle_dist(p2a, p2b, p0a, p0b)
    return qa + x1 / (x2 - x1) * (qa - qb)


@gtstencil()
def _sw_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):

    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )

        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def _nw_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def _ne_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, 0],
            agrid2[-1, 0],
            agrid1[-2, 1],
            agrid2[-2, 1],
            qin[-1, 0, 0],
            qin[-2, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def _se_corner(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(...):
        ec1 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, -1],
            agrid2[0, -1],
            agrid1[1, -2],
            agrid2[1, -2],
            qin[0, -1, 0],
            qin[1, -2, 0],
        )
        ec2 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[-1, -1],
            agrid2[-1, -1],
            agrid1[-2, -2],
            agrid2[-2, -2],
            qin[-1, -1, 0],
            qin[-2, -2, 0],
        )
        ec3 = extrap_corner(
            bgrid1[0, 0],
            bgrid2[0, 0],
            agrid1[0, 0],
            agrid2[0, 0],
            agrid1[1, 1],
            agrid2[1, 1],
            qin[0, 0, 0],
            qin[1, 1, 0],
        )
        qout = (ec1 + ec2 + ec3) * (1.0 / 3.0)


@gtstencil()
def ppm_volume_mean_x(qin: FloatField, qx: FloatField):
    with computation(PARALLEL), interval(...):
        qx[0, 0, 0] = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)


@gtstencil()
def ppm_volume_mean_y(qin: FloatField, qy: FloatField):
    with computation(PARALLEL), interval(...):
        qy[0, 0, 0] = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)


@gtscript.function
def lagrange_y_func(qx):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtstencil()
def lagrange_interpolation_y(qx: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_y_func(qx)


@gtscript.function
def lagrange_x_func(qy):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


@gtstencil()
def lagrange_interpolation_x(qy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout = lagrange_x_func(qy)


@gtstencil()
def cubic_interpolation_south(qx: FloatField, qout: FloatField, qxx: FloatField):
    with computation(PARALLEL), interval(...):
        qxx0 = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx0[0, 1, 0])


@gtstencil()
def cubic_interpolation_north(qx: FloatField, qout: FloatField, qxx: FloatField):
    with computation(PARALLEL), interval(...):
        qxx0 = qxx
        qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx0[0, -1, 0])


@gtstencil()
def cubic_interpolation_west(qy: FloatField, qout: FloatField, qyy: FloatField):
    with computation(PARALLEL), interval(...):
        qyy0 = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy0[1, 0, 0])


@gtstencil()
def cubic_interpolation_east(qy: FloatField, qout: FloatField, qyy: FloatField):
    with computation(PARALLEL), interval(...):
        qyy0 = qyy
        qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy0[-1, 0, 0])


@gtstencil()
def qout_avg(qxx: FloatField, qyy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout[0, 0, 0] = 0.5 * (qxx + qyy)


@gtstencil()
def vort_adjust(qxx: FloatField, qyy: FloatField, qout: FloatField):
    with computation(PARALLEL), interval(...):
        qout[0, 0, 0] = 0.5 * (qxx + qyy)


# @gtstencil()
# def x_edge_q2_west(qin: FloatField, dxa: FloatField, q2: FloatField):
#    with computation(PARALLEL), interval(...):
#        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0, 0]) / (dxa[-1, 0, 0] + dxa)

# @gtstencil()
# def x_edge_qout_west_q2(edge_w: FloatField, q2: FloatField, qout: FloatField):
#    with computation(PARALLEL), interval(...):
#        qout = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2
@gtstencil()
def qout_x_edge(
    qin: FloatField, dxa: FloatFieldIJ, edge_w: FloatFieldIJ, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)
        qout[0, 0, 0] = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


@gtstencil()
def qout_y_edge(
    qin: FloatField, dya: FloatFieldIJ, edge_s: FloatFieldI, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
        qout[0, 0, 0] = edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


@gtstencil()
def qx_edge_west(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dxa[1, 0] / dxa
        g_ou = dxa[-2, 0] / dxa[-1, 0]
        qx[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
        )
        # This does not work, due to access of qx that is changing above

        # qx[1, 0, 0] = (3.0 * (g_in * qin + qin[1, 0, 0])
        #     - (g_in * qx + qx[2, 0, 0])) / (2.0 + 2.0 * g_in)


@gtstencil()
def qx_edge_west2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dxa / dxa[-1, 0]
        qx0 = qx
        qx = (
            3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx0[-1, 0, 0] + qx0[1, 0, 0])
        ) / (2.0 + 2.0 * g_in)


@gtstencil()
def qx_edge_east(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-2, 0] / dxa[-1, 0]
        g_ou = dxa[1, 0] / dxa
        qx[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
        )


@gtstencil()
def qx_edge_east2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dxa[-1, 0] / dxa
        qx0 = qx
        qx = (
            3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx0[1, 0, 0] + qx0[-1, 0, 0])
        ) / (2.0 + 2.0 * g_in)


@gtstencil()
def qy_edge_south(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, 1] / dya
        g_ou = dya[0, -2] / dya[0, -1]
        qy[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
        )


@gtstencil()
def qy_edge_south2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya / dya[0, -1]
        qy0 = qy
        qy = (
            3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy0[0, -1, 0] + qy0[0, 1, 0])
        ) / (2.0 + 2.0 * g_in)


@gtstencil()
def qy_edge_north(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -2] / dya[0, -1]
        g_ou = dya[0, 1] / dya
        qy[0, 0, 0] = 0.5 * (
            ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
            + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
        )


@gtstencil()
def qy_edge_north2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    with computation(PARALLEL), interval(...):
        g_in = dya[0, -1] / dya
        qy0 = qy
        qy = (
            3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy0[0, 1, 0] + qy0[0, -1, 0])
        ) / (2.0 + 2.0 * g_in)


def compute_qout_edges(qin, qout, kstart, nk):
    compute_qout_x_edges(qin, qout, kstart, nk)
    compute_qout_y_edges(qin, qout, kstart, nk)


def compute_qout_x_edges(qin, qout, kstart, nk):
    # qout bounds
    # avoid running west/east computation on south/north tile edges, since
    # they'll be overwritten.
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
    # avoid running south/north computation on west/east tile edges, since
    # they'll be overwritten.
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
        qin.shape, origin=(grid().is_, grid().jsd, kstart), cache_key="a2b_ord4_qx"
    )
    # qx bounds
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
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
        qin.shape, origin=(grid().isd, grid().js, kstart), cache_key="a2b_ord4_qy"
    )
    # qy bounds
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
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
    qxx = utils.make_storage_from_shape(
        qx.shape, origin=grid().full_origin(), cache_key="a2b_ord4_qxx"
    )
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
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
    qyy = utils.make_storage_from_shape(
        qy.shape, origin=grid().full_origin(), cache_key="a2b_ord4_qyy"
    )
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
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
    # avoid running center-domain computation on tile edges, since they'll be
    # overwritten.
    js = grid().js + 1 if grid().south_edge else grid().js
    je = grid().je if grid().north_edge else grid().je + 1
    is_ = grid().is_ + 1 if grid().west_edge else grid().is_
    ie = grid().ie if grid().east_edge else grid().ie + 1
    qout_avg(
        qxx, qyy, qout, origin=(is_, js, kstart), domain=(ie - is_ + 1, je - js + 1, nk)
    )


def compute(qin, qout, kstart=0, nk=None, replace=False):
    if nk is None:
        nk = grid().npz - kstart
    corner_domain = (1, 1, nk)
    _sw_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().is_, grid().js, kstart),
        domain=corner_domain,
    )

    _nw_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().ie + 1, grid().js, kstart),
        domain=corner_domain,
    )
    _ne_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().ie + 1, grid().je + 1, kstart),
        domain=corner_domain,
    )
    _se_corner(
        qin,
        qout,
        grid().agrid1,
        grid().agrid2,
        grid().bgrid1,
        grid().bgrid2,
        origin=(grid().is_, grid().je + 1, kstart),
        domain=corner_domain,
    )
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
