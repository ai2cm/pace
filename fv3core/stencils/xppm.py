from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    compile_assert,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
from fv3core.decorators import FrozenStencil
from fv3core.stencils import yppm
from fv3core.stencils.basic_operations import sign
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def final_flux(courant, q, fx1, tmp):
    return q[-1, 0, 0] + fx1 * tmp if courant > 0.0 else q + fx1 * tmp


@gtscript.function
def fx1_fn(courant, br, b0, bl):
    if courant > 0.0:
        ret = (1.0 - courant) * (br[-1, 0, 0] - courant * b0[-1, 0, 0])
    else:
        ret = (1.0 + courant) * (bl + courant * b0)
    return ret


@gtscript.function
def get_tmp(bl, b0, br):
    from __externals__ import mord

    if __INLINED(mord == 5):
        smt5 = bl * br < 0
    else:
        smt5 = (3.0 * abs(b0)) < abs(bl - br)

    if smt5[-1, 0, 0] or smt5[0, 0, 0]:
        tmp = 1.0
    else:
        tmp = 0.0

    return tmp


@gtscript.function
def get_flux(q: FloatField, courant: FloatField, al: FloatField):
    bl = al[0, 0, 0] - q[0, 0, 0]
    br = al[1, 0, 0] - q[0, 0, 0]
    b0 = bl + br

    tmp = get_tmp(bl, b0, br)
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, tmp)  # noqa


@gtscript.function
def get_flux_ord8plus(
    q: FloatField, courant: FloatField, bl: FloatField, br: FloatField
):
    b0 = bl + br
    fx1 = fx1_fn(courant, br, b0, bl)
    return final_flux(courant, q, fx1, 1.0)


@gtscript.function
def dm_iord8plus(q: FloatField):
    xt = 0.25 * (q[1, 0, 0] - q[-1, 0, 0])
    dqr = max(max(q, q[-1, 0, 0]), q[1, 0, 0]) - q
    dql = q - min(min(q, q[-1, 0, 0]), q[1, 0, 0])
    return sign(min(min(abs(xt), dqr), dql), xt)


@gtscript.function
def al_iord8plus(q: FloatField, dm: FloatField):
    return 0.5 * (q[-1, 0, 0] + q) + 1.0 / 3.0 * (dm[-1, 0, 0] - dm)


@gtscript.function
def blbr_iord8(q: FloatField, al: FloatField, dm: FloatField):
    xt = 2.0 * dm
    bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
    br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)
    return bl, br


@gtscript.function
def xt_dxa_edge_0_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa + dxa[-1, 0]) * q - dxa * q[-1, 0, 0]) / (dxa[-1, 0] + dxa)
        + ((2.0 * dxa[1, 0] + dxa[2, 0]) * q[1, 0, 0] - dxa[1, 0] * q[2, 0, 0])
        / (dxa[1, 0] + dxa[2, 0])
    )


@gtscript.function
def xt_dxa_edge_1_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0] - dxa[-1, 0] * q[-2, 0, 0])
        / (dxa[-2, 0] + dxa[-1, 0])
        + ((2.0 * dxa + dxa[1, 0]) * q - dxa * q[1, 0, 0]) / (dxa + dxa[1, 0])
    )


@gtscript.function
def xt_dxa_edge_0(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_0_base(q, dxa)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dxa_edge_1(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_1_base(q, dxa)
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def compute_al(q: FloatField, dxa: FloatFieldIJ):
    """
    Interpolate q at interface.

    Inputs:
        q: Transported scalar
        dxa: dx on A-grid (?)

    Returns:
        Interpolated quantity
    """
    from __externals__ import i_end, i_start, iord

    compile_assert(iord < 8)

    al = yppm.p1 * (q[-1, 0, 0] + q) + yppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])

    if __INLINED(iord < 0):
        compile_assert(False)
        al = max(al, 0.0)

    with horizontal(region[i_start - 1, :], region[i_end, :]):
        al = yppm.c1 * q[-2, 0, 0] + yppm.c2 * q[-1, 0, 0] + yppm.c3 * q
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        al = 0.5 * (
            ((2.0 * dxa[-1, 0] + dxa[-2, 0]) * q[-1, 0, 0] - dxa[-1, 0] * q[-2, 0, 0])
            / (dxa[-2, 0] + dxa[-1, 0])
            + ((2.0 * dxa[0, 0] + dxa[1, 0]) * q[0, 0, 0] - dxa[0, 0] * q[1, 0, 0])
            / (dxa[0, 0] + dxa[1, 0])
        )
    with horizontal(region[i_start + 1, :], region[i_end + 2, :]):
        al = yppm.c3 * q[-1, 0, 0] + yppm.c2 * q[0, 0, 0] + yppm.c1 * q[1, 0, 0]

    return al


@gtscript.function
def bl_br_edges(bl, br, q, dxa, al, dm):
    from __externals__ import i_end, i_start

    with horizontal(region[i_start - 1, :]):
        xt_bl = yppm.s14 * dm[-1, 0, 0] + yppm.s11 * (q[-1, 0, 0] - q) + q
        xt_br = xt_dxa_edge_0(q, dxa)

    with horizontal(region[i_start, :]):
        xt_bl = xt_dxa_edge_1(q, dxa)
        xt_br = yppm.s15 * q + yppm.s11 * q[1, 0, 0] - yppm.s14 * dm[1, 0, 0]

    with horizontal(region[i_start + 1, :]):
        xt_bl = yppm.s15 * q[-1, 0, 0] + yppm.s11 * q - yppm.s14 * dm
        xt_br = al[1, 0, 0]

    with horizontal(region[i_end - 1, :]):
        xt_bl = al
        xt_br = yppm.s15 * q[1, 0, 0] + yppm.s11 * q + yppm.s14 * dm

    with horizontal(region[i_end, :]):
        xt_bl = yppm.s15 * q + yppm.s11 * q[-1, 0, 0] + yppm.s14 * dm[-1, 0, 0]
        xt_br = xt_dxa_edge_0(q, dxa)

    with horizontal(region[i_end + 1, :]):
        xt_bl = xt_dxa_edge_1(q, dxa)
        xt_br = yppm.s11 * (q[1, 0, 0] - q) - yppm.s14 * dm[1, 0, 0] + q

    with horizontal(
        region[i_start - 1 : i_start + 2, :], region[i_end - 1 : i_end + 2, :]
    ):
        bl = xt_bl - q
        br = xt_br - q

    return bl, br


@gtscript.function
def compute_blbr_ord8plus(q: FloatField, dxa: FloatFieldIJ):
    from __externals__ import i_end, i_start, iord

    dm = dm_iord8plus(q)
    al = al_iord8plus(q, dm)

    compile_assert(iord == 8)

    bl, br = blbr_iord8(q, al, dm)
    bl, br = bl_br_edges(bl, br, q, dxa, al, dm)

    with horizontal(
        region[i_start - 1 : i_start + 2, :], region[i_end - 1 : i_end + 2, :]
    ):
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    return bl, br


def compute_x_flux(
    q: FloatField, courant: FloatField, dxa: FloatFieldIJ, xflux: FloatField
):
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al(q, dxa)
            xflux = get_flux(q, courant, al)
        else:
            bl, br = compute_blbr_ord8plus(q, dxa)
            xflux = get_flux_ord8plus(q, courant, bl, br)


class XPiecewiseParabolic:
    """
    Fortran name is xppm
    """

    def __init__(self, namelist, iord, jfirst, jlast):
        grid = spec.grid
        assert namelist.grid_type < 3
        self._dxa = grid.dxa
        origin = (grid.is_, jfirst, 0)
        domain = (grid.nic + 1, jlast - jfirst + 1, grid.npz + 1)
        ax_offsets = axis_offsets(grid, origin, domain)
        self._compute_flux_stencil = FrozenStencil(
            func=compute_x_flux,
            externals={
                "iord": iord,
                "mord": abs(iord),
                "xt_minmax": True,
                "i_start": ax_offsets["i_start"],
                "i_end": ax_offsets["i_end"],
            },
            origin=origin,
            domain=domain,
        )

    def __call__(self, q: FloatField, c: FloatField, xflux: FloatField):
        """
        Compute x-flux using the PPM method.

        Args:
            q (in): Transported scalar
            c (in): Courant number
            xflux (out): Flux
            jfirst: Starting index of the J-dir compute domain
            jlast: Final index of the J-dir compute domain
        """

        self._compute_flux_stencil(q, c, self._dxa, xflux)
