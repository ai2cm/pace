from typing import Optional

import gt4py
import gt4py.gtscript as gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    asin,
    computation,
    cos,
    horizontal,
    interval,
    region,
    sin,
    sqrt,
)

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.utils import global_config
from fv3core.utils.typing import Float, FloatField, FloatFieldIJ


# compact 4-pt cubic interpolation
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


@gtscript.function
def ppm_volume_mean_x(qin: FloatField):
    return b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)


@gtscript.function
def ppm_volume_mean_y(qin: FloatField):
    return b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)


@gtscript.function
def lagrange_y(qx: FloatField):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtscript.function
def lagrange_x(qy: FloatField):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


@gtscript.function
def cubic_interpolation_south(qx: FloatField, qout: FloatField, qxx: FloatField):
    return c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx[0, 1, 0])


@gtscript.function
def cubic_interpolation_north(qx: FloatField, qout: FloatField, qxx: FloatField):
    return c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx[0, -1, 0])


@gtscript.function
def cubic_interpolation_west(qy: FloatField, qout: FloatField, qyy: FloatField):
    return c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy[1, 0, 0])


@gtscript.function
def cubic_interpolation_east(qy: FloatField, qout: FloatField, qyy: FloatField):
    return c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy[-1, 0, 0])


@gtscript.function
def qout_x_edge(edge_w: FloatFieldIJ, q2: FloatField):
    return edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


@gtscript.function
def qout_y_edge(edge_s: FloatFieldIJ, q1: FloatField):
    return edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


@gtscript.function
def qx_edge_west(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[1, 0] / dxa
    g_ou = dxa[-2, 0] / dxa[-1, 0]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qx_edge_west2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatFieldIJ):
    g_in = dxa / dxa[-1, 0]
    return (
        3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qx[-1, 0, 0] + qx[1, 0, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qx_edge_east(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[-2, 0] / dxa[-1, 0]
    g_ou = dxa[1, 0] / dxa
    return 0.5 * (
        ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qx_edge_east2(qin: FloatField, dxa: FloatFieldIJ, qx: FloatField):
    g_in = dxa[-1, 0] / dxa
    return (
        3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qx[1, 0, 0] + qx[-1, 0, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qy_edge_south(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, 1] / dya
    g_ou = dya[0, -2] / dya[0, -1]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_south2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya / dya[0, -1]
    return (
        3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy[0, -1, 0] + qy[0, 1, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def qy_edge_north(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, -2] / dya[0, -1]
    g_ou = dya[0, 1] / dya
    return 0.5 * (
        ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_north2(qin: FloatField, dya: FloatFieldIJ, qy: FloatField):
    g_in = dya[0, -1] / dya
    return (
        3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy[0, 1, 0] + qy[0, -1, 0])
    ) / (2.0 + 2.0 * g_in)


@gtscript.function
def great_circle_dist_noradius(p1a: Float, p1b: Float, p2a: Float, p2b: Float):
    tb = sin((p1b - p2b) / 2.0) ** 2
    ta = sin((p1a - p2a) / 2.0) ** 2
    return asin(sqrt(tb + cos(p1b) * cos(p2b) * ta)) * 2.0


@gtscript.function
def extrap_corner(
    p0a: Float,
    p0b: Float,
    p1a: Float,
    p1b: Float,
    p2a: Float,
    p2b: Float,
    qa: Float,
    qb: Float,
):
    x1 = great_circle_dist_noradius(p1a, p1b, p0a, p0b)
    x2 = great_circle_dist_noradius(p2a, p2b, p0a, p0b)
    return qa + x1 / (x2 - x1) * (qa - qb)


def _a2b_ord4_stencil(
    qin: FloatField,
    qout: FloatField,
    agrid1: FloatFieldIJ,
    agrid2: FloatFieldIJ,
    bgrid1: FloatFieldIJ,
    bgrid2: FloatFieldIJ,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    edge_n: FloatFieldIJ,
    edge_s: FloatFieldIJ,
    edge_e: FloatFieldIJ,
    edge_w: FloatFieldIJ,
):
    from __externals__ import REPLACE, i_end, i_start, j_end, j_start, namelist

    with computation(PARALLEL), interval(...):
        with horizontal(region[i_start, j_start]):
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

        with horizontal(region[i_end + 1, j_start]):
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

        with horizontal(region[i_end + 1, j_end + 1]):
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

        with horizontal(region[i_start, j_end + 1]):
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

        assert __INLINED(namelist.grid_type < 3)
        # {

        with horizontal(
            region[i_start - 1 : i_start + 1, :], region[i_end : i_end + 2, :]
        ):
            q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)
        with horizontal(region[i_start, j_start + 1 : j_end + 1]):
            qout = qout_x_edge(edge_w, q2)
        with horizontal(region[i_end + 1, j_start + 1 : j_end + 1]):
            qout = qout_x_edge(edge_e, q2)

        with horizontal(
            region[:, j_start - 1 : j_start + 1], region[:, j_end : j_end + 2]
        ):
            q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
        with horizontal(region[i_start + 1 : i_end + 1, j_start]):
            qout = qout_y_edge(edge_s, q1)
        with horizontal(region[i_start + 1 : i_end + 1, j_end + 1]):
            qout = qout_y_edge(edge_n, q1)

        # compute_qx
        qx = ppm_volume_mean_x(qin)
        with horizontal(region[i_start, :]):
            qx = qx_edge_west(qin, dxa)
        with horizontal(region[i_start + 1, :]):
            qx = qx_edge_west2(qin, dxa, qx)

        with horizontal(region[i_end + 1, :]):
            qx = qx_edge_east(qin, dxa)
        with horizontal(region[i_end, :]):
            qx = qx_edge_east2(qin, dxa, qx)

        # compute_qy
        qy = ppm_volume_mean_y(qin)
        with horizontal(region[:, j_start]):
            qy = qy_edge_south(qin, dya)
        with horizontal(region[:, j_start + 1]):
            qy = qy_edge_south2(qin, dya, qy)

        with horizontal(region[:, j_end + 1]):
            qy = qy_edge_north(qin, dya)
        with horizontal(region[:, j_end]):
            qy = qy_edge_north2(qin, dya, qy)

        # compute_qxx
        qxx = lagrange_y(qx)
        with horizontal(region[:, j_start + 1]):
            qxx = cubic_interpolation_south(qx, qout, qxx)
        with horizontal(region[:, j_end]):
            qxx = cubic_interpolation_north(qx, qout, qxx)

        # compute_qyy
        qyy = lagrange_x(qy)
        with horizontal(region[i_start + 1, :]):
            qyy = cubic_interpolation_west(qy, qout, qyy)
        with horizontal(region[i_end, :]):
            qyy = cubic_interpolation_east(qy, qout, qyy)

        with horizontal(region[i_start + 1 : i_end + 1, j_start + 1 : j_end + 1]):
            qout = 0.5 * (qxx + qyy)
        # }

        if __INLINED(REPLACE):
            qin = qout


def _make_grid_storage_2d(grid_array: gt4py.storage.Storage, index: int = 0):
    grid = spec.grid
    return gt4py.storage.from_array(
        grid_array[:, :, index],
        backend=global_config.get_backend(),
        default_origin=grid.compute_origin()[:-1],
        shape=grid_array[:, :, index].shape,
        dtype=grid_array.dtype,
        mask=(True, True, False),
    )


def compute(
    qin: FloatField,
    qout: FloatField,
    kstart: int = 0,
    nk: Optional[int] = None,
    replace: bool = False,
):
    """
    Transfers qin from A-grid to B-grid.

    Args:
        qin: Input on A-grid (in)
        qout: Output on B-grid (out)
        kstart: Starting level
        nk: Number of levels
        replace: If True, sets `qout = qin` as the last step
    """
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    agrid1 = _make_grid_storage_2d(grid.agrid1)
    agrid2 = _make_grid_storage_2d(grid.agrid2)
    bgrid1 = _make_grid_storage_2d(grid.bgrid1)
    bgrid2 = _make_grid_storage_2d(grid.bgrid2)
    dxa = _make_grid_storage_2d(grid.dxa)
    dya = _make_grid_storage_2d(grid.dya)
    edge_n = _make_grid_storage_2d(grid.edge_n)
    edge_s = _make_grid_storage_2d(grid.edge_s)
    edge_e = _make_grid_storage_2d(grid.edge_e)
    edge_w = _make_grid_storage_2d(grid.edge_w)

    stencil = gtstencil(definition=_a2b_ord4_stencil, externals={"REPLACE": replace})

    stencil(
        qin,
        qout,
        agrid1,
        agrid2,
        bgrid1,
        bgrid2,
        dxa,
        dya,
        edge_n,
        edge_s,
        edge_e,
        edge_w,
        origin=(grid.is_, grid.js, kstart),
        domain=(grid.nic + 1, grid.njc + 1, nk),
    )
