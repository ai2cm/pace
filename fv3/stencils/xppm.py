#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from .yppm import (
    p1,
    p2,
    c1,
    c2,
    c3,
    get_bl,
    get_b0,
    is_smt5_mord5,
    is_smt5_most_mords,
    fx1_c_negative,
)
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd
origin = (2, 0, 0)


def grid():
    return spec.grid


@gtscript.stencil(backend=utils.exec_backend, externals={"p1": p1, "p2": p2})
def main_al(q: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = p1 * (q[-1, 0, 0] + q) + p2 * (q[-2, 0, 0] + q[1, 0, 0])


@gtscript.stencil(backend=utils.exec_backend, externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_0(q: sd, dya: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = c1 * q[-2, 0, 0] + c2 * q[-1, 0, 0] + c3 * q


@gtscript.stencil(backend=utils.exec_backend, externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_1(q: sd, dya: sd, al: sd):
    with computation(PARALLEL), interval(0, None):
        al[0, 0, 0] = 0.5 * (
            (
                (2.0 * dya[-1, 0, 0] + dya[-2, 0, 0]) * q[-1, 0, 0]
                - dya[-1, 0, 0] * q[-2, 0, 0]
            )
            / (dya[-2, 0, 0] + dya[-1, 0, 0])
            + (
                (2.0 * dya[0, 0, 0] + dya[1, 0, 0]) * q[0, 0, 0]
                - dya[0, 0, 0] * q[1, 0, 0]
            )
            / (dya[0, 0, 0] + dya[1, 0, 0])
        )


@gtscript.stencil(backend=utils.exec_backend, externals={"c1": c1, "c2": c2, "c3": c3})
def al_y_edge_2(q: sd, dya: sd, al: sd):
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


@gtscript.stencil(backend=utils.exec_backend)
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


def compute_al(q, dxa, iord, is1, ie3, jfirst, jlast):
    dimensions = q.shape
    al = utils.make_storage_from_shape(dimensions, origin)
    domain_y = (1, dimensions[1], dimensions[2])
    if iord < 8:
        main_al(
            q,
            al,
            origin=(is1, jfirst, 0),
            domain=(ie3 - is1 + 1, jlast - jfirst + 1, grid().npz),
        )
        if not grid().nested and spec.namelist["grid_type"] < 3:
            if grid().west_edge:
                al_y_edge_0(q, dxa, al, origin=(grid().is_ - 1, 0, 0), domain=domain_y)
                al_y_edge_1(q, dxa, al, origin=(grid().is_, 0, 0), domain=domain_y)
                al_y_edge_2(q, dxa, al, origin=(grid().is_ + 1, 0, 0), domain=domain_y)
            if grid().east_edge:
                al_y_edge_0(q, dxa, al, origin=(grid().ie, 0, 0), domain=domain_y)
                al_y_edge_1(q, dxa, al, origin=(grid().ie + 1, 0, 0), domain=domain_y)
                al_y_edge_2(q, dxa, al, origin=(grid().ie + 2, 0, 0), domain=domain_y)
    return al


def compute_flux(q, c, iord, jfirst, jlast):
    mord = abs(iord)
    # output  storage
    is1 = max(5, grid().is_ - 1)
    ie3 = min(grid().npx, grid().ie + 2)
    flux = utils.make_storage_from_shape(q.shape, origin)
    al = compute_al(q, grid().dxa, iord, is1, ie3, jfirst, jlast)
    get_flux(
        q,
        c,
        al,
        flux,
        mord=mord,
        origin=(grid().is_, jfirst, 0),
        domain=(grid().nic + 1, jlast - jfirst + 1, grid().npz),
    )
    return flux
