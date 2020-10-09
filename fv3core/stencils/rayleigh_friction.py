#!/usr/bin/env python3
import math

import fv3gfs.util as fv3util
import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.c2l_ord as c2l_ord
import fv3core.stencils.rayleigh_super as ray_super
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd
SDAY = 86400.0  # seconds per day
U000 = 4900.0  # scaling velocity
RCV = ray_super.RCV


@gtstencil()
def initialize_u2f_friction(ua: sd, va: sd, w: sd, u2f: sd, hydrostatic: bool):
    with computation(PARALLEL), interval(...):
        if hydrostatic:
            u2f = ua ** 2 + va ** 2
        else:
            u2f = ua ** 2 + va ** 2 + w ** 2


@gtstencil()
def rayleigh_pt_friction(
    pt: sd,
    rf: sd,
    pfull: sd,
    u2f: sd,
    delz: sd,
    ptop: float,
    rf_cutoff: float,
    conserve: bool,
    hydrostatic: bool,
):
    with computation(PARALLEL), interval(...):
        if conserve:
            if hydrostatic:
                pt = pt + 0.5 * u2f / (
                    constants.CP_AIR - constants.RDGAS * ptop / pfull
                ) * (1.0 - 1.0 / (1.0 + rk * (u2f / U000) ** 0.5))
            else:
                delz = delz / pt
                pt = pt + 0.5 * u2f * RCV * (
                    1.0 - 1.0 / (1.0 + rf * (u2f / U000) ** 0.5)
                )
        if not hydrostatic:
            w = w / (1.0 + u2f)


@gtstencil()
def update_u2f(u2f: sd, rf: sd):
    with computation(PARALLEL), interval(...):
        u2f = rf * (u2f / U000) ** 0.5


@gtstencil()
def rayleigh_u_friction(u: sd, pfull: sd, u2f: sd, rf_cutoff: float):
    with computation(PARALLEL), interval(...):
        u = u / (1.0 + 0.5 * (u2f[0, -1, 0] + u2f))


@gtstencil()
def rayleigh_v_friction(v: sd, pfull: sd, u2f: sd, rf_cutoff: float):
    with computation(PARALLEL), interval(...):
        v = v / (1.0 + 0.5 * (u2f[-1, 0, 0] + u2f))


def compute(u, v, w, ua, va, pt, delz, phis, bdt, ptop, pfull, comm):
    grid = spec.grid
    # TODO pull this into a state dict or arguments that get updated when called
    rf_initialized = False
    conserve = not (grid.nested or spec.namelist.regional)
    rf_cutoff = spec.namelist.rf_cutoff
    if not rf_initialized:
        # is only a column actually
        rf = np.zeros(grid.npz)
        rfvals = ray_super.rayleigh_rfvals(
            bdt, spec.namelist.tau * SDAY, rf_cutoff, pfull, ptop
        )
        rf, kmax = ray_super.fill_rf(rf, rfvals, rf_cutoff, pfull, u.shape)
        rf_initialized = True  # TODO propagate to global scope
    c2l_ord.compute_ord2(u, v, ua, va)

    # TODO this really only needs to be kmax size in the 3rd dimension...
    u2f = grid.quantity_factory.zeros(
        [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "m/s"
    )

    initialize_u2f_friction(
        ua,
        va,
        w,
        u2f.storage,
        spec.namelist.hydrostatic,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, kmax),
    )

    comm.halo_update(u2f, n_points=utils.halo)
    rayleigh_pt_friction(
        pt,
        rf,
        pfull,
        u2f.storage,
        delz,
        ptop,
        rf_cutoff,
        conserve,
        spec.namelist.hydrostatic,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, kmax),
    )
    update_u2f(
        u2f.storage,
        rf,
        origin=(grid.is_ - 1, grid.js - 1, 0),
        domain=(grid.nic + 2, grid.njc + 2, kmax),
    )
    rayleigh_u_friction(
        u,
        u2f.storage,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, kmax),
    )
    rayleigh_v_friction(
        v,
        u2f.storage,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, kmax),
    )
