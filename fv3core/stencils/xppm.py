from typing import Optional

from gt4py import gtscript
from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.stencils import yppm
from fv3core.stencils.basic_operations import sign
from fv3core.utils.typing import FloatField


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
    # al, dm = al_iord8plus_fn(q, al, dm, r3)
    xt = 2.0 * dm
    bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
    br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)
    return bl, br


@gtscript.function
def xt_dxa_edge_0_base(q, dxa):
    return 0.5 * (
        ((2.0 * dxa + dxa[-1, 0, 0]) * q - dxa * q[-1, 0, 0]) / (dxa[-1, 0, 0] + dxa)
        + ((2.0 * dxa[1, 0, 0] + dxa[2, 0, 0]) * q[1, 0, 0] - dxa[1, 0, 0] * q[2, 0, 0])
        / (dxa[1, 0, 0] + dxa[2, 0, 0])
    )


@gtscript.function
def xt_dxa_edge_1_base(q, dxa):
    return 0.5 * (
        (
            (2.0 * dxa[-1, 0, 0] + dxa[-2, 0, 0]) * q[-1, 0, 0]
            - dxa[-1, 0, 0] * q[-2, 0, 0]
        )
        / (dxa[-2, 0, 0] + dxa[-1, 0, 0])
        + ((2.0 * dxa + dxa[1, 0, 0]) * q - dxa * q[1, 0, 0]) / (dxa + dxa[1, 0, 0])
    )


@gtscript.function
def xt_dxa_edge_0(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_0_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dxa_edge_1(q, dxa):
    from __externals__ import xt_minmax

    xt = xt_dxa_edge_1_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if __INLINED(xt_minmax):
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def west_edge_iord8plus_0(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    bl = yppm.s14 * dm[-1, 0, 0] + yppm.s11 * (q[-1, 0, 0] - q)
    xt = xt_dxa_edge_0(q, dxa)
    br = xt - q
    return bl, br


@gtscript.function
def west_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    xt = xt_dxa_edge_1(q, dxa)
    bl = xt - q
    xt = yppm.s15 * q + yppm.s11 * q[1, 0, 0] - yppm.s14 * dm[1, 0, 0]
    br = xt - q
    return bl, br


@gtscript.function
def west_edge_iord8plus_2(
    q: FloatField,
    dm: FloatField,
    al: FloatField,
):
    xt = yppm.s15 * q[-1, 0, 0] + yppm.s11 * q - yppm.s14 * dm
    bl = xt - q
    br = al[1, 0, 0] - q
    return bl, br


@gtscript.function
def east_edge_iord8plus_0(
    q: FloatField,
    dm: FloatField,
    al: FloatField,
):
    bl = al - q
    xt = yppm.s15 * q[1, 0, 0] + yppm.s11 * q + yppm.s14 * dm
    br = xt - q
    return bl, br


@gtscript.function
def east_edge_iord8plus_1(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    xt = yppm.s15 * q + yppm.s11 * q[-1, 0, 0] + yppm.s14 * dm[-1, 0, 0]
    bl = xt - q
    xt = xt_dxa_edge_0(q, dxa)
    br = xt - q
    return bl, br


@gtscript.function
def east_edge_iord8plus_2(
    q: FloatField,
    dxa: FloatField,
    dm: FloatField,
):
    xt = xt_dxa_edge_1(q, dxa)
    bl = xt - q
    br = yppm.s11 * (q[1, 0, 0] - q) - yppm.s14 * dm[1, 0, 0]
    return bl, br


@gtscript.function
def compute_al(q: FloatField, dxa: FloatField):
    """
    Interpolate q at interface.

    Inputs:
        q: Transported scalar
        dxa: dx on A-grid (?)

    Returns:
        Interpolated quantity
    """
    from __externals__ import i_end, i_start, iord

    assert __INLINED(iord < 8), "The code in this function requires iord < 8"

    al = yppm.p1 * (q[-1, 0, 0] + q) + yppm.p2 * (q[-2, 0, 0] + q[1, 0, 0])

    if __INLINED(iord < 0):
        assert __INLINED(False), "Not tested"
        al = max(al, 0.0)

    with horizontal(region[i_start - 1, :], region[i_end, :]):
        al = yppm.c1 * q[-2, 0, 0] + yppm.c2 * q[-1, 0, 0] + yppm.c3 * q
    with horizontal(region[i_start, :], region[i_end + 1, :]):
        al = 0.5 * (
            (
                (2.0 * dxa[-1, 0, 0] + dxa[-2, 0, 0]) * q[-1, 0, 0]
                - dxa[-1, 0, 0] * q[-2, 0, 0]
            )
            / (dxa[-2, 0, 0] + dxa[-1, 0, 0])
            + (
                (2.0 * dxa[0, 0, 0] + dxa[1, 0, 0]) * q[0, 0, 0]
                - dxa[0, 0, 0] * q[1, 0, 0]
            )
            / (dxa[0, 0, 0] + dxa[1, 0, 0])
        )
    with horizontal(region[i_start + 1, :], region[i_end + 2, :]):
        al = yppm.c3 * q[-1, 0, 0] + yppm.c2 * q[0, 0, 0] + yppm.c1 * q[1, 0, 0]

    return al


@gtscript.function
def compute_blbr_ord8plus(q: FloatField, dxa: FloatField):
    from __externals__ import i_end, i_start, iord, namelist

    dm = dm_iord8plus(q)
    al = al_iord8plus(q, dm)

    assert __INLINED(iord == 8), "Unimplemented iord"
    # {
    bl, br = blbr_iord8(q, al, dm)
    # }

    assert __INLINED(namelist.grid_type < 3)
    # {
    with horizontal(region[i_start - 1, :]):
        bl, br = west_edge_iord8plus_0(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_start, :]):
        bl, br = west_edge_iord8plus_1(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_start + 1, :]):
        bl, br = west_edge_iord8plus_2(q, dm, al)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_end - 1, :]):
        bl, br = east_edge_iord8plus_0(q, dm, al)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_end, :]):
        bl, br = east_edge_iord8plus_1(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)

    with horizontal(region[i_end + 1, :]):
        bl, br = east_edge_iord8plus_2(q, dxa, dm)
        bl, br = yppm.pert_ppm_standard_constraint_fcn(q, bl, br)
    # }

    return bl, br


def _compute_flux_stencil(
    q: FloatField, courant: FloatField, dxa: FloatField, xflux: FloatField
):
    from __externals__ import mord

    with computation(PARALLEL), interval(...):
        if __INLINED(mord < 8):
            al = compute_al(q, dxa)
            xflux = get_flux(q, courant, al)
        else:
            bl, br = compute_blbr_ord8plus(q, dxa)
            xflux = get_flux_ord8plus(q, courant, bl, br)


def compute_flux(
    q: FloatField,
    c: FloatField,
    xflux: FloatField,
    iord: int,
    jfirst: int,
    jlast: int,
    kstart: int = 0,
    nk: Optional[int] = None,
):
    """
    Compute x-flux using the PPM method.

    Args:
        q (in): Transported scalar
        c (in): Courant number
        xflux (out): Flux
        iord: Method selector
        jfirst: Starting index of the J-dir compute domain
        jlast: Final index of the J-dir compute domain
        kstart: First index of the K-dir compute domain
        nk: Number of indices in the K-dir compute domain
    """
    # Tests: xppm, fvtp2d, tracer2d1l
    grid = spec.grid
    if nk is None:
        nk = spec.grid.npz - kstart
    stencil = gtstencil(
        definition=_compute_flux_stencil,
        externals={
            "iord": iord,
            "mord": abs(iord),
            "xt_minmax": True,
        },
    )
    nj = jlast - jfirst + 1
    stencil(
        q,
        c,
        grid.dxa,
        xflux,
        origin=(grid.is_, jfirst, kstart),
        domain=(grid.nic + 1, nj, nk),
    )
