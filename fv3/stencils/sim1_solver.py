#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.copy_stencil as cp

sd = utils.sd


# TODO: implement MOIST_CAPPA=false
def solve(is_, ie, dt, gm2, cp2, pe2, dm, pm2, pem, w2, dz2, ptr, wsr):
    nic = ie - is_ + 1
    km = spec.grid.npz - 1
    npz = spec.grid.npz
    tmpshape = (nic, km + 1)
    tmpshape_p1 = (nic, km + 2)
    t1g = 2.0 * dt * dt
    rdt = 1.0 / dt
    slice_m = slice(0, km + 1)
    slice_m2 = slice(1, km + 1)
    slice_n = slice(0, km)
    g_rat = np.zeros(tmpshape)
    bb = np.zeros(tmpshape)
    aa = np.zeros(tmpshape)
    dd = np.zeros(tmpshape)
    gam = np.zeros(tmpshape)
    w1 = np.zeros(tmpshape)
    pp = np.zeros(tmpshape_p1)
    bet = np.zeros(nic)
    p1 = np.zeros(nic)
    pp = np.zeros(pem.shape)
    pe2[:, slice_m] = np.exp(gm2 * np.log(-dm / dz2 * constants.RDGAS * ptr)) - pm2
    w1 = np.copy(w2)

    g_rat[:, slice_n] = dm[:, slice_n] / dm[:, slice_m2]
    bb[:, slice_n] = 2.0 * (1.0 + g_rat[:, slice_n])
    dd[:, slice_n] = 3.0 * (pe2[:, slice_n] + g_rat[:, slice_n] * pe2[:, slice_m2])

    bet[:] = bb[:, 0]
    pp[:, 0] = 0.0
    pp[:, 1] = dd[:, 0] / bet
    bb[:, km] = 2.0
    dd[:, km] = 3.0 * pe2[:, km]

    for k in range(1, npz):
        for i in range(nic):
            gam[i, k] = g_rat[i, k - 1] / bet[i]
            bet[i] = bb[i, k] - gam[i, k]
            pp[i, k + 1] = (dd[i, k] - pp[i, k]) / bet[i]

    for k in range(km, 0, -1):
        pp[:, k] = pp[:, k] - gam[:, k] * pp[:, k + 1]

    # w solver
    aa[:, slice_m2] = (
        t1g
        * 0.5
        * (gm2[:, slice_n] + gm2[:, slice_m2])
        / (dz2[:, slice_n] + dz2[:, slice_m2])
        * (pem[:, slice_m2] + pp[:, slice_m2])
    )

    bet[:] = dm[:, 0] - aa[:, 1]
    w2[:, 0] = (dm[:, 0] * w1[:, 0] + dt * pp[:, 1]) / bet
    for k in range(1, km):
        for i in range(nic):
            gam[i, k] = aa[i, k] / bet[i]
            bet[i] = dm[i, k] - (aa[i, k] + aa[i, k + 1] + aa[i, k] * gam[i, k])
            w2[i, k] = (
                dm[i, k] * w1[i, k]
                + dt * (pp[i, k + 1] - pp[i, k])
                - aa[i, k] * w2[i, k - 1]
            ) / bet[i]

    for i in range(nic):
        p1[i] = t1g * gm2[i, km] / dz2[i, km] * (pem[i, km + 1] + pp[i, km + 1])
        gam[i, km] = aa[i, km] / bet[i]
        bet[i] = dm[i, km] - (aa[i, km] + p1[i] + aa[i, km] * gam[i, km])
        w2[i, km] = (
            dm[i, km] * w1[i, km]
            + dt * (pp[i, km + 1] - pp[i, km])
            - p1[i] * wsr[i, 0]
            - aa[i, km] * w2[i, km - 1]
        ) / bet[i]

    for k in range(km - 1, -1, -1):
        w2[:, k] = w2[:, k] - gam[:, k + 1] * w2[:, k + 1]

    pe2[:, 0] = 0.0
    for k in range(npz):
        pe2[:, k + 1] = pe2[:, k] + dm[:, k] * (w2[:, k] - w1[:, k]) * rdt

    p1[:] = (pe2[:, km] + 2.0 * pe2[:, km + 1]) * 1.0 / 3.0

    for i in range(nic):
        dz2[i, km] = (
            -dm[i, km]
            * constants.RDGAS
            * ptr[i, km]
            * np.exp(
                (cp2[i, km] - 1.0)
                * np.log(max(spec.namelist["p_fac"] * pm2[i, km], p1[i] + pm2[i, km]))
            )
        )

    for k in range(npz - 2, -1, -1):
        for i in range(nic):
            p1[i] = (
                pe2[i, k] + bb[i, k] * pe2[i, k + 1] + g_rat[i, k] * pe2[i, k + 2]
            ) * 1.0 / 3.0 - g_rat[i, k] * p1[i]
            dz2[i, k] = (
                -dm[i, k]
                * constants.RDGAS
                * ptr[i, k]
                * np.exp(
                    (cp2[i, k] - 1.0)
                    * np.log(max(spec.namelist["p_fac"] * pm2[i, k], p1[i] + pm2[i, k]))
                )
            )
