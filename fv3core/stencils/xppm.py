#!/usr/bin/env python3
import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import floor_cap, sign

from .yppm import (
    c1,
    c2,
    c3,
    floor_cap,
    fx1_c_negative,
    get_b0,
    get_bl,
    is_smt5_mord5,
    is_smt5_most_mords,
    p1,
    p2,
    pert_ppm,
    s11,
    s14,
    s15,
)


sd = utils.sd
origin = (2, 0, 0)


def grid():
    return spec.grid


@gtstencil(externals={"p1": p1, "p2": p2})
def main_al(q: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = p1 * (q[-1, 0, 0] + q) + p2 * (q[-2, 0, 0] + q[1, 0, 0])


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_0(q: sd, dxa: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c1 * q[-2, 0, 0] + c2 * q[-1, 0, 0] + c3 * q


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_1(q: sd, dxa: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = 0.5 * (
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


@gtstencil(externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_2(q: sd, dxa: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c3 * q[-1, 0, 0] + c2 * q[0, 0, 0] + c1 * q[1, 0, 0]


@gtscript.function
def get_br(al, q):
    br = al[1, 0, 0] - q
    return br


@gtscript.function
def fx1_c_positive(c, br, b0):
    return (1.0 - c) * (br[-1, 0, 0] - c * b0[-1, 0, 0])


@gtscript.function
def flux_intermediates(q, al, mord):
    bl = get_bl(al=al, q=q)
    br = get_br(al=al, q=q)
    b0 = get_b0(bl=bl, br=br)
    smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
    tmp = smt5[-1, 0, 0] + smt5 * (smt5[-1, 0, 0] == 0)
    return bl, br, b0, tmp


@gtscript.function
def fx1_fn(c, br, b0, bl):
    return fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)


@gtscript.function
def final_flux(c, q, fx1, tmp):
    return q[-1, 0, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


@gtstencil()
def get_flux(q: sd, c: sd, al: sd, flux: sd, *, mord: int):
    with computation(PARALLEL), interval(0, None):
        bl, br, b0, tmp = flux_intermediates(q, al, mord)
        fx1 = fx1_fn(c, br, b0, bl)
        # TODO: add [0, 0, 0] when gt4py bug gets fixed
        flux = final_flux(c, q, fx1, tmp)  # noqa
        # bl = get_bl(al=al, q=q)
        # br = get_br(al=al, q=q)
        # b0 = get_b0(bl=bl, br=br)
        # smt5 = is_smt5_mord5(bl, br) if mord == 5 else is_smt5_most_mords(bl, br, b0)
        # tmp = smt5[-1, 0, 0] + smt5 * (smt5[-1, 0, 0] == 0)
        # fx1 = fx1_c_positive(c, br, b0) if c > 0.0 else fx1_c_negative(c, bl, b0)
        # flux = q[-1, 0, 0] + fx1 * tmp if c > 0.0 else q + fx1 * tmp


@gtstencil()
def finalflux_ord8plus(q: sd, c: sd, bl: sd, br: sd, flux: sd):
    with computation(PARALLEL), interval(...):
        b0 = get_b0(bl, br)
        fx1 = fx1_fn(c, br, b0, bl)
        flux = q[-1, 0, 0] + fx1 if c > 0.0 else q + fx1


@gtstencil()
def dm_iord8plus(q: sd, al: sd, dm: sd):
    with computation(PARALLEL), interval(...):
        xt = 0.25 * (q[1, 0, 0] - q[-1, 0, 0])
        dqr = max(max(q, q[-1, 0, 0]), q[1, 0, 0]) - q
        dql = q - min(min(q, q[-1, 0, 0]), q[1, 0, 0])
        dm = sign(min(min(abs(xt), dqr), dql), xt)


@gtstencil()
def al_iord8plus(q: sd, al: sd, dm: sd, r3: float):
    with computation(PARALLEL), interval(...):
        al = 0.5 * (q[-1, 0, 0] + q) + r3 * (dm[-1, 0, 0] - dm)


@gtstencil()
def blbr_iord8(q: sd, al: sd, bl: sd, br: sd, dm: sd):
    with computation(PARALLEL), interval(...):
        # al, dm = al_iord8plus_fn(q, al, dm, r3)
        xt = 2.0 * dm
        bl = -1.0 * sign(min(abs(xt), abs(al - q)), xt)
        br = sign(min(abs(xt), abs(al[1, 0, 0] - q)), xt)


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
def xt_dxa_edge_0(q, dxa, xt_minmax):
    xt = xt_dxa_edge_0_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if xt_minmax:
        minq = min(min(min(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        maxq = max(max(max(q[-1, 0, 0], q), q[1, 0, 0]), q[2, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtscript.function
def xt_dxa_edge_1(q, dxa, xt_minmax):
    xt = xt_dxa_edge_1_base(q, dxa)
    minq = 0.0
    maxq = 0.0
    if xt_minmax:
        minq = min(min(min(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        maxq = max(max(max(q[-2, 0, 0], q[-1, 0, 0]), q), q[1, 0, 0])
        xt = min(max(xt, minq), maxq)
    return xt


@gtstencil()
def west_edge_iord8plus_0(q: sd, dxa: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        bl = s14 * dm[-1, 0, 0] + s11 * (q[-1, 0, 0] - q)
        xt = xt_dxa_edge_0(q, dxa, xt_minmax)
        br = xt - q


@gtstencil()
def west_edge_iord8plus_1(q: sd, dxa: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1(q, dxa, xt_minmax)
        bl = xt - q
        xt = s15 * q + s11 * q[1, 0, 0] - s14 * dm[1, 0, 0]
        br = xt - q


@gtstencil()
def west_edge_iord8plus_2(q: sd, dxa: sd, dm: sd, al: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        xt = s15 * q[-1, 0, 0] + s11 * q - s14 * dm
        bl = xt - q
        br = al[1, 0, 0] - q


@gtstencil()
def east_edge_iord8plus_0(q: sd, dxa: sd, dm: sd, al: sd, bl: sd, br: sd):
    with computation(PARALLEL), interval(...):
        bl = al - q
        xt = s15 * q[1, 0, 0] + s11 * q + s14 * dm
        br = xt - q


@gtstencil()
def east_edge_iord8plus_1(q: sd, dxa: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = s15 * q + s11 * q[-1, 0, 0] + s14 * dm[-1, 0, 0]
        bl = xt - q
        xt = xt_dxa_edge_0(q, dxa, xt_minmax)
        br = xt - q


@gtstencil()
def east_edge_iord8plus_2(q: sd, dxa: sd, dm: sd, bl: sd, br: sd, xt_minmax: bool):
    with computation(PARALLEL), interval(...):
        xt = xt_dxa_edge_1(q, dxa, xt_minmax)
        bl = xt - q
        br = s11 * (q[1, 0, 0] - q) - s14 * dm[1, 0, 0]


def compute_al(q, dxa, iord, is1, ie3, jfirst, jlast, kstart=0, nk=None):
    if nk is None:
        nk = grid().npz - kstart
    dimensions = q.shape
    local_origin = (origin[0], origin[1], kstart)
    al = utils.make_storage_from_shape(dimensions, local_origin)
    domain_y = (1, dimensions[1], nk)
    if iord < 8:
        main_al(
            q,
            al,
            origin=(is1, jfirst, kstart),
            domain=(ie3 - is1 + 1, jlast - jfirst + 1, nk),
        )
        if not grid().nested and spec.namelist.grid_type < 3:
            if grid().west_edge:
                al_y_edge_0(
                    q, dxa, al, origin=(grid().is_ - 1, 0, kstart), domain=domain_y
                )
                al_y_edge_1(q, dxa, al, origin=(grid().is_, 0, kstart), domain=domain_y)
                al_y_edge_2(
                    q, dxa, al, origin=(grid().is_ + 1, 0, kstart), domain=domain_y
                )
            if grid().east_edge:
                al_y_edge_0(q, dxa, al, origin=(grid().ie, 0, kstart), domain=domain_y)
                al_y_edge_1(
                    q, dxa, al, origin=(grid().ie + 1, 0, kstart), domain=domain_y
                )
                al_y_edge_2(
                    q, dxa, al, origin=(grid().ie + 2, 0, kstart), domain=domain_y
                )
        if iord < 0:
            floor_cap(
                al,
                0.0,
                origin=(grid().is_ - 1, jfirst, kstart),
                domain=(grid().nic + 3, jlast - jfirst + 1, nk),
            )
    return al


def compute_blbr_ord8plus(q, iord, jfirst, jlast, is1, ie1, kstart, nk):
    r3 = 1.0 / 3.0
    grid = spec.grid
    local_origin = (origin[0], origin[1], kstart)
    bl = utils.make_storage_from_shape(q.shape, local_origin)
    br = utils.make_storage_from_shape(q.shape, local_origin)
    dm = utils.make_storage_from_shape(q.shape, local_origin)
    al = utils.make_storage_from_shape(q.shape, local_origin)
    dj = jlast - jfirst + 1
    dm_iord8plus(
        q, al, dm, origin=(grid.is_ - 2, jfirst, kstart), domain=(grid.nic + 4, dj, nk)
    )
    al_iord8plus(
        q, al, dm, r3, origin=(is1, jfirst, kstart), domain=(ie1 - is1 + 2, dj, nk)
    )
    if iord == 8:
        blbr_iord8(
            q,
            al,
            bl,
            br,
            dm,
            origin=(is1, jfirst, kstart),
            domain=(ie1 - is1 + 1, dj, nk),
        )
    else:
        raise Exception("Unimplemented iord=" + str(iord))

    if spec.namelist.grid_type < 3 and not (grid.nested or spec.namelist.regional):
        y_edge_domain = (1, dj, nk)
        do_xt_minmax = True
        if grid.west_edge:
            west_edge_iord8plus_0(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.is_ - 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            west_edge_iord8plus_1(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.is_, jfirst, kstart),
                domain=y_edge_domain,
            )
            west_edge_iord8plus_2(
                q,
                grid.dxa,
                dm,
                al,
                bl,
                br,
                origin=(grid.is_ + 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            pert_ppm(q, bl, br, 1, grid.is_ - 1, jfirst, kstart, 3, dj, nk)
        if grid.east_edge:
            east_edge_iord8plus_0(
                q,
                grid.dxa,
                dm,
                al,
                bl,
                br,
                origin=(grid.ie - 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            east_edge_iord8plus_1(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.ie, jfirst, kstart),
                domain=y_edge_domain,
            )
            east_edge_iord8plus_2(
                q,
                grid.dxa,
                dm,
                bl,
                br,
                do_xt_minmax,
                origin=(grid.ie + 1, jfirst, kstart),
                domain=y_edge_domain,
            )
            pert_ppm(q, bl, br, 1, grid.ie - 1, jfirst, kstart, 3, dj, nk)
        return bl, br


def compute_flux(q, c, xflux, iord, jfirst, jlast, kstart=0, nk=None):
    grid = spec.grid
    if nk is None:
        nk = grid.npz - kstart
    mord = abs(iord)
    if mord not in [5, 6, 7, 8]:
        raise Exception(
            "We have only implemented yppm for hord=5, 6, 7, and 8, not " + str(iord)
        )
    # output  storage
    is1 = grid.is_ + 2 if grid.west_edge else grid.is_ - 1
    ie3 = grid.ie - 1 if grid.east_edge else grid.ie + 2
    ie1 = grid.ie - 2 if grid.east_edge else grid.ie + 1
    flux_origin = (grid.is_, jfirst, kstart)
    flux_domain = (grid.nic + 1, jlast - jfirst + 1, nk)
    if mord < 8:
        al = compute_al(q, grid.dxa, iord, is1, ie3, jfirst, jlast, kstart, nk)
        get_flux(q, c, al, xflux, mord=mord, origin=flux_origin, domain=flux_domain)
    else:
        bl, br = compute_blbr_ord8plus(q, iord, jfirst, jlast, is1, ie1, kstart, nk)
        finalflux_ord8plus(q, c, bl, br, xflux, origin=flux_origin, domain=flux_domain)
