#!/usr/bin/env python3
import math

import fv3gfs.util as fv3util
import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.c2l_ord as c2l_ord
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd
U0 = 60.0
SDAY = 86400.0
RCV = 1.0 / (constants.CP_AIR - constants.RDGAS)
# NOTE The fortran version of this computes rf in the first timestep only. Then rf_initialized let's you know you can skip it.
# Here we calculate it every time
@gtscript.function
def compute_rf_vals(pfull, bdt, rf_cutoff, tau0, ptop):
    return (
        bdt
        / tau0
        * sin(0.5 * constants.PI * log(rf_cutoff / pfull) / log(rf_cutoff / ptop)) ** 2
    )


@gtstencil()
def initialize_u2f(pfull: sd, u2f: sd, bdt: float, tau0: float, ptop: float):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            rf = compute_rf_vals(pfull, bdt, spec.namelist.rf_cutoff, tau0, ptop)
            u2f = 1.0 / (1.0 + rf)
        else:
            u2f = 1.0


@gtstencil()
def rayleigh_pt_vert(
    pt: sd,
    ua: sd,
    va: sd,
    w: sd,
    pfull: sd,
    u2f: sd,
    ptop: float,
    hydrostatic: bool,
):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            if hydrostatic:
                pt = pt + 0.5 * (ua ** 2 + va ** 2) * (1.0 - u2f ** 2) / (
                    constants.CP_AIR - constants.RDGAS * ptop / pfull
                )
            else:
                pt = pt + 0.5 * (ua ** 2 + va ** 2 + w ** 2) * (1.0 - u2f ** 2) * RCV
            if not hydrostatic:
                w = u2f * w


@gtstencil()
def rayleigh_u(u: sd, pfull: sd, u2f: sd):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            u = 0.5 * (u2f[0, -1, 0] + u2f) * u


@gtstencil()
def rayleigh_v(
    v: sd,
    pfull: sd,
    u2f: sd,
):
    with computation(PARALLEL), interval(...):
        if pfull < spec.namelist.rf_cutoff:
            v = 0.5 * (u2f[-1, 0, 0] + u2f) * v


def compute(u, v, w, ua, va, pt, delz, phis, bdt, ptop, pfull, comm):
    grid = spec.grid
    c2l_ord.compute_ord2(u, v, ua, va)

    # TODO this really only needs to be kmax size in the 3rd dimension...
    u2f = grid.quantity_factory.zeros(
        [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "m/s"
    )

    initialize_u2f(
        pfull,
        u2f.storage,
        bdt,
        abs(spec.namelist.tau * SDAY),
        ptop,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )

    comm.halo_update(u2f, n_points=utils.halo)
    rayleigh_pt_vert(
        pt,
        ua,
        va,
        w,
        pfull,
        u2f.storage,
        ptop,
        spec.namelist.hydrostatic,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )

    rayleigh_u(
        u,
        pfull,
        u2f.storage,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, grid.npz),
    )
    rayleigh_v(
        v,
        pfull,
        u2f.storage,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, grid.npz),
    )
