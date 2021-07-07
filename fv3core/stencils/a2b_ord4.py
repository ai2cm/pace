import gt4py.gtscript as gtscript
from gt4py.gtscript import (
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

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.stencils.basic_operations import copy_defn
from fv3core.utils import axis_offsets
from fv3core.utils.grid import GridIndexing
from fv3core.utils.typing import FloatField, FloatFieldI, FloatFieldIJ
from fv3gfs.util import X_DIM, X_INTERFACE_DIM, Y_DIM, Y_INTERFACE_DIM, Z_DIM


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


@gtscript.function
def lagrange_y_func(qx):
    return a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)


@gtscript.function
def lagrange_x_func(qy):
    return a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)


def ppm_volume_mean_x(
    qin: FloatField,
    qx: FloatField,
    dxa: FloatFieldIJ,
):
    from __externals__ import i_end, i_start

    with computation(PARALLEL), interval(...):
        qx = b2 * (qin[-2, 0, 0] + qin[1, 0, 0]) + b1 * (qin[-1, 0, 0] + qin)
        with horizontal(region[i_start, :]):
            qx = qx_edge_west(qin, dxa)
        with horizontal(region[i_start + 1, :]):
            qx = qx_edge_west2(qin, dxa)
        with horizontal(region[i_end + 1, :]):
            qx = qx_edge_east(qin, dxa)
        with horizontal(region[i_end, :]):
            qx = qx_edge_east2(qin, dxa)


def ppm_volume_mean_y(
    qin: FloatField,
    qy: FloatField,
    dya: FloatFieldIJ,
):
    from __externals__ import j_end, j_start

    with computation(PARALLEL), interval(...):
        qy = b2 * (qin[0, -2, 0] + qin[0, 1, 0]) + b1 * (qin[0, -1, 0] + qin)
        with horizontal(region[:, j_start]):
            qy = qy_edge_south(qin, dya)
        with horizontal(region[:, j_start + 1]):
            qy = qy_edge_south2(qin, dya)
        with horizontal(region[:, j_end + 1]):
            qy = qy_edge_north(qin, dya)
        with horizontal(region[:, j_end]):
            qy = qy_edge_north2(qin, dya)


@gtscript.function
def a2b_interpolation(
    qout: FloatField,
    qx: FloatField,
    qy: FloatField,
    qxx: FloatField,
    qyy: FloatField,
):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(...):
        qxx = a2 * (qx[0, -2, 0] + qx[0, 1, 0]) + a1 * (qx[0, -1, 0] + qx)
        qyy = a2 * (qy[-2, 0, 0] + qy[1, 0, 0]) + a1 * (qy[-1, 0, 0] + qy)
        # TODO(rheag) use a function with an offset when that works consistently
        with horizontal(region[:, j_start + 1]):
            qxx_upper = a2 * (qx[0, -1, 0] + qx[0, 2, 0]) + a1 * (qx + qx[0, 1, 0])
            qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, -1, 0] + qxx_upper)
        with horizontal(region[:, j_end]):
            qxx_lower = a2 * (qx[0, -3, 0] + qx) + a1 * (qx[0, -2, 0] + qx[0, -1, 0])
            qxx = c1 * (qx[0, -1, 0] + qx) + c2 * (qout[0, 1, 0] + qxx_lower)
        with horizontal(region[i_start + 1, :]):
            qyy_right = a2 * (qy[-1, 0, 0] + qy[2, 0, 0]) + a1 * (qy + qy[1, 0, 0])
            qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[-1, 0, 0] + qyy_right)
        with horizontal(region[i_end, :]):
            qyy_left = a2 * (qy[-3, 0, 0] + qy) + a1 * (qy[-2, 0, 0] + qy[-1, 0, 0])
            qyy = c1 * (qy[-1, 0, 0] + qy) + c2 * (qout[1, 0, 0] + qyy_left)
        qout = 0.5 * (qxx + qyy)


def qout_x_edge(
    qin: FloatField, dxa: FloatFieldIJ, edge_w: FloatFieldIJ, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q2 = (qin[-1, 0, 0] * dxa + qin * dxa[-1, 0]) / (dxa[-1, 0] + dxa)
        qout[0, 0, 0] = edge_w * q2[0, -1, 0] + (1.0 - edge_w) * q2


def qout_y_edge(
    qin: FloatField, dya: FloatFieldIJ, edge_s: FloatFieldI, qout: FloatField
):
    with computation(PARALLEL), interval(...):
        q1 = (qin[0, -1, 0] * dya + qin * dya[0, -1]) / (dya[0, -1] + dya)
        qout[0, 0, 0] = edge_s * q1[-1, 0, 0] + (1.0 - edge_s) * q1


@gtscript.function
def qx_edge_west(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[1, 0] / dxa
    g_ou = dxa[-2, 0] / dxa[-1, 0]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[1, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_ou)
    )
    # This does not work, due to access of qx that is changing above

    # qx[1, 0, 0] = (3.0 * (g_in * qin + qin[1, 0, 0])
    #     - (g_in * qx + qx[2, 0, 0])) / (2.0 + 2.0 * g_in)


@gtscript.function
def qx_edge_west2(qin: FloatField, dxa: FloatFieldIJ):
    # TODO: should be able to use 2d variable with offset:
    # qxleft = qx_edge_west(qin[-1, 0, 0], dxa[-1, 0])
    # TODO this seemed to work for a bit, and then stopped
    # qxright = ppm_volume_mean_x_main(qin[1, 0, 0])
    g_in = dxa / dxa[-1, 0]
    g_ou = dxa[-3, 0] / dxa[-2, 0]
    qxleft = 0.5 * (
        ((2.0 + g_in) * qin[-1, 0, 0] - qin) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[-2, 0, 0] - qin[-3, 0, 0]) / (1.0 + g_ou)
    )
    qxright = b2 * (qin[-1, 0, 0] + qin[2, 0, 0]) + b1 * (qin + qin[1, 0, 0])
    return (3.0 * (g_in * qin[-1, 0, 0] + qin) - (g_in * qxleft + qxright)) / (
        2.0 + 2.0 * g_in
    )


@gtscript.function
def qx_edge_east(qin: FloatField, dxa: FloatFieldIJ):
    g_in = dxa[-2, 0] / dxa[-1, 0]
    g_ou = dxa[1, 0] / dxa
    return 0.5 * (
        ((2.0 + g_in) * qin[-1, 0, 0] - qin[-2, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[1, 0, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qx_edge_east2(qin: FloatField, dxa: FloatFieldIJ):
    # TODO(rheag) use a function with an offset when that works consistently
    # qxright = qx_edge_east(qin[1, 0, 0], dxa[1, 0])
    # qxleft = ppm_volume_mean_x_main(qin[-1, 0, 0])
    g_in = dxa[-1, 0] / dxa
    g_ou = dxa[2, 0] / dxa[1, 0]
    qxright = 0.5 * (
        ((2.0 + g_in) * qin - qin[-1, 0, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[1, 0, 0] - qin[2, 0, 0]) / (1.0 + g_ou)
    )
    qxleft = b2 * (qin[-3, 0, 0] + qin) + b1 * (qin[-2, 0, 0] + qin[-1, 0, 0])
    return (3.0 * (qin[-1, 0, 0] + g_in * qin) - (g_in * qxright + qxleft)) / (
        2.0 + 2.0 * g_in
    )


@gtscript.function
def qy_edge_south(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, 1] / dya
    g_ou = dya[0, -2] / dya[0, -1]
    return 0.5 * (
        ((2.0 + g_in) * qin - qin[0, 1, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_south2(qin: FloatField, dya: FloatFieldIJ):
    # TODO(rheag) use a function with an offset when that works consistently
    # qy_lower = qy_edge_south(qin[0, -1, 0], dya[0, -1])
    # qy_upper = ppm_volume_mean_y_main(qin[0, 1, 0])
    g_in = dya / dya[0, -1]
    g_ou = dya[0, -3] / dya[0, -2]
    qy_lower = 0.5 * (
        ((2.0 + g_in) * qin[0, -1, 0] - qin) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[0, -2, 0] - qin[0, -3, 0]) / (1.0 + g_ou)
    )
    qy_upper = b2 * (qin[0, -1, 0] + qin[0, 2, 0]) + b1 * (qin + qin[0, 1, 0])
    return (3.0 * (g_in * qin[0, -1, 0] + qin) - (g_in * qy_lower + qy_upper)) / (
        2.0 + 2.0 * g_in
    )


@gtscript.function
def qy_edge_north(qin: FloatField, dya: FloatFieldIJ):
    g_in = dya[0, -2] / dya[0, -1]
    g_ou = dya[0, 1] / dya
    return 0.5 * (
        ((2.0 + g_in) * qin[0, -1, 0] - qin[0, -2, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin - qin[0, 1, 0]) / (1.0 + g_ou)
    )


@gtscript.function
def qy_edge_north2(qin: FloatField, dya: FloatFieldIJ):
    # TODO(rheag) use a function with an offset when that works consistently
    # qy_lower = ppm_volume_mean_y_main(qin[0, -1, 0])
    # qy_upper = qy_edge_north(qin[0, 1, 0], dya[0, 1])
    g_in = dya[0, -1] / dya
    g_ou = dya[0, 2] / dya[0, 1]
    qy_lower = b2 * (qin[0, -3, 0] + qin[0, 0, 0]) + b1 * (
        qin[0, -2, 0] + qin[0, -1, 0]
    )
    qy_upper = 0.5 * (
        ((2.0 + g_in) * qin - qin[0, -1, 0]) / (1.0 + g_in)
        + ((2.0 + g_ou) * qin[0, 1, 0] - qin[0, 2, 0]) / (1.0 + g_ou)
    )
    return (3.0 * (qin[0, -1, 0] + g_in * qin) - (g_in * qy_upper + qy_lower)) / (
        2.0 + 2.0 * g_in
    )


class AGrid2BGridFourthOrder:
    """
    Fortran name is a2b_ord4, test module is A2B_Ord4
    """

    def __init__(
        self,
        grid_indexing: GridIndexing,
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
        grid_type,
        z_dim=Z_DIM,
        replace: bool = False,
    ):
        """
        Args:
            grid_indexing: defines indices over which to perform conversion
            grid_type: integer representing the type of grid
            z_dim: defines whether vertical dimension is centered or staggered
            replace: boolean, update qin to the B grid as well
        """
        assert grid_type < 3
        self._idx: GridIndexing = grid_indexing
        self._agrid1 = agrid1
        self._agrid2 = agrid2
        self._bgrid1 = bgrid1
        self._bgrid2 = bgrid2
        self._dxa = dxa
        self._dya = dya
        self._edge_n = edge_n
        self._edge_s = edge_s
        self._edge_e = edge_e
        self._edge_w = edge_w

        self.replace = replace

        self._tmp_qx = utils.make_storage_from_shape(self._idx.max_shape)
        self._tmp_qy = utils.make_storage_from_shape(self._idx.max_shape)
        self._tmp_qxx = utils.make_storage_from_shape(self._idx.max_shape)
        self._tmp_qyy = utils.make_storage_from_shape(self._idx.max_shape)

        _, (z_domain,) = self._idx.get_origin_domain([z_dim])
        corner_domain = (1, 1, z_domain)

        self._sw_corner_stencil = FrozenStencil(
            _sw_corner,
            origin=self._idx.origin_compute(),
            domain=corner_domain,
        )
        self._nw_corner_stencil = FrozenStencil(
            _nw_corner,
            origin=(self._idx.iec + 1, self._idx.jsc, self._idx.origin[2]),
            domain=corner_domain,
        )
        self._ne_corner_stencil = FrozenStencil(
            _ne_corner,
            origin=(self._idx.iec + 1, self._idx.jec + 1, self._idx.origin[2]),
            domain=corner_domain,
        )
        self._se_corner_stencil = FrozenStencil(
            _se_corner,
            origin=(self._idx.isc, self._idx.jec + 1, self._idx.origin[2]),
            domain=corner_domain,
        )
        js2 = self._idx.jsc + 1 if self._idx.south_edge else self._idx.jsc
        je1 = self._idx.jec if self._idx.north_edge else self._idx.jec + 1
        dj2 = je1 - js2 + 1

        # edge_w is singleton in the I-dimension to work around gt4py not yet
        # supporting J-fields. As a result, the origin has to be zero for
        # edge_w, anything higher is outside its index range
        self._qout_x_edge_west = FrozenStencil(
            qout_x_edge,
            origin={
                "_all_": (self._idx.isc, js2, self._idx.origin[2]),
                "edge_w": (0, js2),
            },
            domain=(1, dj2, z_domain),
        )
        self._qout_x_edge_east = FrozenStencil(
            qout_x_edge,
            origin={
                "_all_": (self._idx.iec + 1, js2, self._idx.origin[2]),
                "edge_w": (0, js2),
            },
            domain=(1, dj2, z_domain),
        )

        is2 = self._idx.isc + 1 if self._idx.west_edge else self._idx.isc
        ie1 = self._idx.iec if self._idx.east_edge else self._idx.iec + 1
        di2 = ie1 - is2 + 1
        self._qout_y_edge_south = FrozenStencil(
            qout_y_edge,
            origin=(is2, self._idx.jsc, self._idx.origin[2]),
            domain=(di2, 1, z_domain),
        )
        self._qout_y_edge_north = FrozenStencil(
            qout_y_edge,
            origin=(is2, self._idx.jec + 1, self._idx.origin[2]),
            domain=(di2, 1, z_domain),
        )

        origin_x, domain_x = self._idx.get_origin_domain(
            dims=[X_INTERFACE_DIM, Y_DIM, z_dim], halos=(0, 2)
        )

        ax_offsets_x = axis_offsets(
            self._idx,
            origin_x,
            domain_x,
        )
        self._ppm_volume_mean_x_stencil = FrozenStencil(
            ppm_volume_mean_x, externals=ax_offsets_x, origin=origin_x, domain=domain_x
        )
        origin_y, domain_y = self._idx.get_origin_domain(
            dims=[X_DIM, Y_INTERFACE_DIM, z_dim], halos=(2, 0)
        )
        ax_offsets_y = axis_offsets(
            self._idx,
            origin_y,
            domain_y,
        )
        self._ppm_volume_mean_y_stencil = FrozenStencil(
            ppm_volume_mean_y, externals=ax_offsets_y, origin=origin_y, domain=domain_y
        )

        origin, domain = self._idx.get_origin_domain(
            dims=(X_INTERFACE_DIM, Y_INTERFACE_DIM, z_dim),
        )
        origin, domain = self._exclude_tile_edges(origin, domain)

        ax_offsets = axis_offsets(
            self._idx,
            origin,
            domain,
        )
        self._a2b_interpolation_stencil = FrozenStencil(
            a2b_interpolation, externals=ax_offsets, origin=origin, domain=domain
        )
        origin, domain = self._idx.get_origin_domain(
            dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, z_dim]
        )
        self._copy_stencil = FrozenStencil(
            copy_defn,
            origin=origin,
            domain=domain,
        )

    def _exclude_tile_edges(self, origin, domain, dims=("x", "y")):
        """
        Args:
            origin: origin for which to exclude tile edges
            domain: domain for which to exclude tile edges
            dims: dimensions on which to exclude tile edges,
                can include "x" or "y" and defaults to both
        """
        origin, domain = list(origin), list(domain)
        # don't compute last point in tile domain on each edge
        if self._idx.south_edge and "y" in dims:
            origin[1] += 1
            domain[1] -= 1  # must adjust domain to keep endpoint the same
        if self._idx.north_edge and "y" in dims:
            domain[1] -= 1
        if self._idx.west_edge and "x" in dims:
            origin[0] += 1
            domain[0] -= 1
        if self._idx.east_edge and "x" in dims:
            domain[0] -= 1
        return tuple(origin), tuple(domain)

    def __call__(self, qin: FloatField, qout: FloatField):
        """Converts qin from A-grid to B-grid in qout.
        In some cases, qin is also updated to the B grid.
        Args:
        qin: Input on A-grid (inout)
        qout: Output on B-grid (inout)
        """

        self._sw_corner_stencil(
            qin,
            qout,
            self._agrid1,
            self._agrid2,
            self._bgrid1,
            self._bgrid2,
        )

        self._nw_corner_stencil(
            qin,
            qout,
            self._agrid1,
            self._agrid2,
            self._bgrid1,
            self._bgrid2,
        )
        self._ne_corner_stencil(
            qin,
            qout,
            self._agrid1,
            self._agrid2,
            self._bgrid1,
            self._bgrid2,
        )
        self._se_corner_stencil(
            qin,
            qout,
            self._agrid1,
            self._agrid2,
            self._bgrid1,
            self._bgrid2,
        )

        self._compute_qout_edges(qin, qout)
        self._ppm_volume_mean_x_stencil(
            qin,
            self._tmp_qx,
            self._dxa,
        )
        self._ppm_volume_mean_y_stencil(
            qin,
            self._tmp_qy,
            self._dya,
        )
        self._a2b_interpolation_stencil(
            qout,
            self._tmp_qx,
            self._tmp_qy,
            self._tmp_qxx,
            self._tmp_qyy,
        )
        if self.replace:
            self._copy_stencil(
                qout,
                qin,
            )

    def _compute_qout_edges(self, qin: FloatField, qout: FloatField):
        if self._idx.west_edge:
            self._qout_x_edge_west(
                qin,
                self._dxa,
                self._edge_w,
                qout,
            )
        if self._idx.east_edge:
            self._qout_x_edge_east(
                qin,
                self._dxa,
                self._edge_e,
                qout,
            )

        if self._idx.south_edge:
            self._qout_y_edge_south(
                qin,
                self._dya,
                self._edge_s,
                qout,
            )
        if self._idx.north_edge:
            self._qout_y_edge_north(
                qin,
                self._dya,
                self._edge_n,
                qout,
            )
