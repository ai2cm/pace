#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.stencils.c2l_ord as c2l_ord
from gt4py.gtscript import computation, interval, PARALLEL
import fv3.utils.global_constants as constants
import numpy as np
import math
import fv3util

sd = utils.sd
U0 = 60.0
SDAY = 86400.0


@utils.stencil()
def initialize_u2f(rf: sd, pfull: sd, u2f: sd, rf_cutoff: float):
    with computation(PARALLEL), interval(...):
        if pfull < rf_cutoff:
            u2f = 1.0 / (1.0 + rf)
        else:
            u2f = 1.0


@utils.stencil()
def rayleigh_pt_vert(
    pt: sd,
    ua: sd,
    va: sd,
    w: sd,
    pfull: sd,
    u2f: sd,
    rcv: float,
    ptop: float,
    rf_cutoff: float,
    conserve: bool,
    hydrostatic: bool,
):
    with computation(PARALLEL), interval(...):
        if pfull < rf_cutoff:
            if conserve:
                if hydrostatic:
                    pt = pt + 0.5 * (ua ** 2 + va ** 2) * (1.0 - u2f ** 2) / (
                        constants.CP_AIR - constants.RDGAS * ptop / pfull
                    )
                else:
                    pt = (
                        pt + 0.5 * (ua ** 2 + va ** 2 + w ** 2) * (1.0 - u2f ** 2) * rcv
                    )
            if not hydrostatic:
                w = u2f * w


@utils.stencil()
def rayleigh_u(u: sd, pfull: sd, u2f: sd, rf_cutoff: float):
    with computation(PARALLEL), interval(...):
        if pfull < rf_cutoff:
            u = 0.5 * (u2f[0, -1, 0] + u2f) * u


@utils.stencil()
def rayleigh_v(v: sd, pfull: sd, u2f: sd, rf_cutoff: float):
    with computation(PARALLEL), interval(...):
        if pfull < rf_cutoff:
            v = 0.5 * (u2f[-1, 0, 0] + u2f) * v


def compute(u, v, w, ua, va, pt, delz, phis, bdt, ptop, pfull, comm):
    grid = spec.grid
    rf_initialized = False  # TODO pull this into a state dict or arguments that get updated when called
    conserve = not (grid.nested or spec.namelist["regional"])
    rf_cutoff = spec.namelist["rf_cutoff"]
    rcv = 1.0 / (constants.CP_AIR - constants.RDGAS)
    if not rf_initialized:
        tau0 = abs(spec.namelist["tau"] * SDAY)
        # is only a column actually
        rf = np.zeros(grid.npz)
        if spec.namelist["tau"] < 0:
            rfvals = bdt / tau0 * (np.log(rf_cutoff / pfull[0, 0, 0 : grid.npz])) ** 2
        else:
            rfvals = (
                bdt
                / tau0
                * np.sin(
                    0.5
                    * constants.PI
                    * np.log(rf_cutoff / np.squeeze(pfull[0, 0, 0 : grid.npz]))
                    / math.log(rf_cutoff / ptop)
                )
                ** 2
            )
        neg_pfull = np.argwhere(pfull[0, 0, 0 : grid.npz] < rf_cutoff)
        kmax = (
            neg_pfull[-1][-1] + 1
        )  # TODO why + 1? this doesn't seem like it should be + 1
        rf[neg_pfull] = rfvals[neg_pfull]
        # TODO this makes the column 3d, undo when you can
        rf = utils.make_storage_data(rf, u.shape, origin=grid.default_origin())
        rf_initialized = True  # TODO propoagate to global scope
    c2l_ord.compute_ord2(u, v, ua, va)

    # TODO this really only needs to be kmax size in the 3rd dimension...
    u2f = grid.quantity_factory.zeros(
        [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "m/s"
    )

    initialize_u2f(
        rf,
        pfull,
        u2f.data,
        rf_cutoff,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, kmax),
    )
    comm.halo_update(u2f, n_points=utils.halo)
    rayleigh_pt_vert(
        pt,
        ua,
        va,
        w,
        pfull,
        u2f.data,
        rcv,
        ptop,
        rf_cutoff,
        conserve,
        spec.namelist["hydrostatic"],
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, kmax),
    )
    rayleigh_u(
        u,
        pfull,
        u2f.data,
        rf_cutoff,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, kmax),
    )
    rayleigh_v(
        v,
        pfull,
        u2f.data,
        rf_cutoff,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, kmax),
    )
