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


@gtstencil()
def ray_fast_u(u: sd, rf: sd, dp: sd, dmu: sd):
    with computation(FORWARD):
        with interval(0, 1):
            dmu = (1.0 - rf) * dp * u
            u = rf * u
        with interval(1, None):
            dmu = dmu[0, 0, -1] + (1.0 - rf) * dp * u
            u = rf * u


@gtstencil()
def ray_fast_v(v: sd, rf: sd, dp: sd, dmv: sd):
    with computation(FORWARD):
        with interval(0, 1):
            dmv = (1.0 - rf) * dp * v
            v = rf * v
        with interval(1, None):
            dmv = dmv[0, 0, -1] + (1.0 - rf) * dp * v
            v = rf * v


@gtstencil()
def ray_fast_w(w: sd, rf: sd):
    with computation(PARALLEL), interval(...):
        w = rf * w


@gtstencil()
def ray_fast_horizontal_dm(wind: sd, dmwind: sd, dm: sd):
    with computation(PARALLEL):
        with interval(...):
            dmwind = dmwind / dm
            wind = wind + dmwind


@gtstencil()
def dm_stencil(dp: sd, dm: sd):
    with computation(FORWARD):
        with interval(0, 1):
            dm = dp
        with interval(1, None):
            dm = dm[0, 0, -1] + dp


def compute(u, v, w, dp, pfull, dt, ptop, ks):
    grid = spec.grid
    rff_initialized = (
        False  # TODO pull this out to higher level so don't do over and over
    )
    rf_cutoff = spec.namelist.rf_cutoff
    rf_cutoff_nudge = rf_cutoff + min(100.0, 10.0 * ptop)
    if not rff_initialized:
        # is only a column actually
        rf = np.ones(grid.npz)
        rffvals = ray_super.rayleigh_rfvals(
            dt, spec.namelist.tau * ray_super.SDAY, rf_cutoff, pfull, ptop
        )
        rffvals = 1.0 / (1.0 + rffvals)  # TODO put in stencil with the rayleigh_rfvals
        rf, kmax = ray_super.fill_rf(rf, rffvals, rf_cutoff, pfull, u.shape)
        # dm_k_rf(pfull, dp, dm,rf_cutoff_nudge, origin=grid.compute_origin(), domain=(grid.nic+1, grid.njc+1, ks))
        # TODO do something better here
        neg_pfull = np.argwhere(
            pfull[spec.grid.is_, spec.grid.js, 0:ks] < rf_cutoff_nudge
        )
        if len(neg_pfull) == 0:
            k_rf = 1
        else:
            k_rf = neg_pfull[-1][-1] + 1

        rff_initialized = (
            True  # TODO propagate to rest of dyncore so this isn't unecessarily redone
        )
    dm = utils.make_storage_from_shape(rf.shape, grid.default_origin())
    dmu = utils.make_storage_from_shape(rf.shape, grid.default_origin())
    dmv = utils.make_storage_from_shape(rf.shape, grid.default_origin())
    dm_stencil(
        dp, dm, origin=grid.compute_origin(), domain=(grid.nic + 1, grid.njc + 1, k_rf)
    )
    dm = utils.make_storage_data(np.squeeze(dm[:, :, k_rf - 1]), dm.shape)
    ray_fast_u(
        u,
        rf,
        dp,
        dmu,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc + 1, kmax),
    )
    ray_fast_v(
        v,
        rf,
        dp,
        dmv,
        origin=grid.compute_origin(),
        domain=(grid.nic + 1, grid.njc, kmax),
    )
    if not spec.namelist.hydrostatic:
        ray_fast_w(
            w, rf, origin=grid.compute_origin(), domain=(grid.nic, grid.njc, kmax)
        )
    dmu = utils.make_storage_data(np.squeeze(dmu[:, :, kmax - 1]), dm.shape)
    dmv = utils.make_storage_data(np.squeeze(dmv[:, :, kmax - 1]), dm.shape)
    ray_fast_horizontal_dm(
        u, dmu, dm, origin=grid.compute_origin(), domain=(grid.nic, grid.njc + 1, k_rf)
    )
    ray_fast_horizontal_dm(
        v, dmv, dm, origin=grid.compute_origin(), domain=(grid.nic + 1, grid.njc, k_rf)
    )
