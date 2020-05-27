#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd

a1 = 0.5625
a2 = -0.0625
c1 = 1.125
c2 = -0.125


@utils.stencil()
def c2l_ord2(
    u: sd, v: sd, dx: sd, dy: sd, a11: sd, a12: sd, a21: sd, a22: sd, ua: sd, va: sd
):
    with computation(PARALLEL), interval(...):
        wu = u * dx
        wv = v * dy
        # Co-variant vorticity-conserving interpolation
        u1 = 2.0 * (wu + wu[0, 1, 0]) / (dx + dx[0, 1, 0])
        v1 = 2.0 * (wv + wv[1, 0, 0]) / (dy + dy[1, 0, 0])
        # Cubed (cell center co-variant winds) to lat-lon
        ua = a11 * u1 + a12 * v1
        va = a21 * u1 + a22 * v1


def compute_ord2(u, v, ua, va, do_halo=False):
    grid = spec.grid
    i1 = grid.is_
    i2 = grid.ie
    j1 = grid.js
    j2 = grid.je
    # Usually used for nesting
    if do_halo:
        i1 -= -1
        i2 += 1
        j1 -= 1
        j2 += 1

    c2l_ord2(
        u,
        v,
        grid.dx,
        grid.dy,
        grid.a11,
        grid.a12,
        grid.a21,
        grid.a22,
        ua,
        va,
        origin=(i1, j1, 0),
        domain=(i2 - i1 + 1, j2 - j1 + 1, grid.npz),
    )


@utils.stencil()
def vector_tmp(u: sd, v: sd, utmp: sd, vtmp: sd):
    with computation(PARALLEL), interval(...):
        utmp = c2 * (u[0, -1, 0] + u[0, 2, 0]) + c1 * (u + u[0, 1, 0])
        vtmp = c2 * (v[-1, 0, 0] + v[2, 0, 0]) + c1 * (v + v[1, 0, 0])


@utils.stencil()
def y_edge_tmp(u: sd, v: sd, utmp: sd, vtmp: sd, dx: sd, dy: sd):
    with computation(PARALLEL), interval(...):
        wv = v * dy
        vtmp = 2.0 * (wv + wv[1, 0, 0]) / (dy + dy[1, 0, 0])
        utmp = 2.0 * (u * dx + u[0, 1, 0] * dx[0, 1, 0]) / (dx + dx[0, 1, 0])


@utils.stencil()
def x_edge_wv(v: sd, dy: sd, wv: sd):
    with computation(PARALLEL), interval(...):
        wv = v * dy


@utils.stencil()
def x_edge_tmp(wv: sd, u: sd, v: sd, utmp: sd, vtmp: sd, dx: sd, dy: sd):
    with computation(PARALLEL), interval(...):
        wu = u * dx
        utmp = 2.0 * (wu + wu[0, 1, 0]) / (dx + dx[0, 1, 0])
        vtmp = 2.0 * (wv + wv[1, 0, 0]) / (dy + dy[1, 0, 0])


@utils.stencil()
def ord4_transform(
    utmp: sd, vtmp: sd, a11: sd, a12: sd, a21: sd, a22: sd, ua: sd, va: sd
):
    with computation(PARALLEL), interval(...):
        # Transform local a-grid winds into latitude-longitude coordinates
        ua = a11 * utmp + a12 * vtmp
        va = a21 * utmp + a22 * vtmp


def compute_ord4(u, v, ua, va, mode=1):
    grid = spec.grid
    # TODO needs a halo update? but already passes tests without...
    # if mode > 0:
    # 1) vector halo update (u, v)
    utmp = utils.make_storage_from_shape(u.shape, utils.origin)
    vtmp = utils.make_storage_from_shape(v.shape, utils.origin)
    j1 = grid.js + 1 if grid.south_edge else grid.js
    j2 = grid.je - 1 if grid.north_edge else grid.je
    i1 = grid.is_ + 1 if grid.west_edge else grid.is_
    i2 = grid.ie - 1 if grid.east_edge else grid.ie
    vector_tmp(
        u,
        v,
        utmp,
        vtmp,
        origin=(i1, j1, 0),
        domain=(i2 - i1 + 1, j2 - j1 + 1, grid.npz),
    )
    if grid.south_edge:
        y_edge_tmp(
            u,
            v,
            utmp,
            vtmp,
            grid.dx,
            grid.dy,
            origin=(grid.is_, grid.js, 0),
            domain=(grid.nic, 1, grid.npz),
        )
    if grid.north_edge:
        y_edge_tmp(
            u,
            v,
            utmp,
            vtmp,
            grid.dx,
            grid.dy,
            origin=(grid.is_, grid.je, 0),
            domain=(grid.nic, 1, grid.npz),
        )
    if grid.west_edge:
        wv = utils.make_storage_from_shape(u.shape, utils.origin)
        x_edge_wv(
            v,
            grid.dy,
            wv,
            origin=(grid.is_, grid.js, 0),
            domain=(2, grid.njc, grid.npz),
        )
        x_edge_tmp(
            wv,
            u,
            v,
            utmp,
            vtmp,
            grid.dx,
            grid.dy,
            origin=(grid.is_, grid.js, 0),
            domain=(1, grid.njc, grid.npz),
        )
    if grid.east_edge:
        wv = utils.make_storage_from_shape(u.shape, utils.origin)
        x_edge_wv(
            v, grid.dy, wv, origin=(grid.ie, grid.js, 0), domain=(2, grid.njc, grid.npz)
        )
        x_edge_tmp(
            wv,
            u,
            v,
            utmp,
            vtmp,
            grid.dx,
            grid.dy,
            origin=(grid.ie, grid.js, 0),
            domain=(1, grid.njc, grid.npz),
        )

    ord4_transform(
        utmp,
        vtmp,
        grid.a11,
        grid.a12,
        grid.a21,
        grid.a22,
        ua,
        va,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )


def compute_cubed_to_latlon(u, v, ua, va):
    if spec.namelist["c2l_ord"] == 2:
        compute_ord2(u, v, ua, va, False)
    else:
        compute_ord4(u, v, ua, va)
