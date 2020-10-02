#!/usr/bin/env python3
import gt4py as gt
import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.copy_stencil as cp
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils


sd = utils.sd


@utils.stencil()
def sim1_solver(
    w: sd,
    dm: sd,
    gm: sd,
    dz: sd,
    ptr: sd,
    pm: sd,
    pe: sd,
    pem: sd,
    wsr: sd,
    cp3: sd,
    dt: float,
    t1g: float,
    rdt: float,
    p_fac: float,
):
    # TODO: we only want to bottom level of wsr, so this could be removed once wsr_top is a 2d field
    with computation(FORWARD):
        with interval(0, 1):
            wsr_top = wsr
        with interval(1, None):
            wsr_top = wsr_top[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        pe = exp(gm * log(-dm / dz * constants.RDGAS * ptr)) - pm
        w1 = w
    with computation(FORWARD):
        with interval(0, -2):
            g_rat = dm / dm[0, 0, 1]
            bb = 2.0 * (1.0 + g_rat)
            dd = 3.0 * (pe + g_rat * pe[0, 0, 1])
        with interval(-2, -1):
            bb = 2.0
            dd = 3.0 * pe
    # bet[i,j,k] = bb[i,j,0]
    with computation(FORWARD):
        with interval(0, 1):
            bet = bb
        with interval(1, -1):
            bet = bet[0, 0, -1]

    ### stencils: w_solver
    # {
    with computation(PARALLEL):
        with interval(0, 1):
            pp = 0.0
        with interval(1, 2):
            pp = dd[0, 0, -1] / bet
    with computation(FORWARD), interval(1, -1):
        gam = g_rat[0, 0, -1] / bet[0, 0, -1]
        bet = bb - gam
    with computation(FORWARD), interval(2, None):
        pp = (dd[0, 0, -1] - pp[0, 0, -1]) / bet[0, 0, -1]
    with computation(BACKWARD), interval(1, -1):
        pp = pp - gam * pp[0, 0, 1]
        # w solver
        aa = t1g * 0.5 * (gm[0, 0, -1] + gm) / (dz[0, 0, -1] + dz) * (pem + pp)
    # }
    ## updates on bet:
    with computation(FORWARD):
        with interval(0, 1):
            bet = dm[0, 0, 0] - aa[0, 0, 1]
        with interval(1, None):
            bet = bet[0, 0, -1]
    ### w_pe_dz_compute
    # {
    with computation(FORWARD):
        with interval(0, 1):
            w = (dm * w1 + dt * pp[0, 0, 1]) / bet
        with interval(1, -2):
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + aa[0, 0, 1] + aa * gam)
            w = (dm * w1 + dt * (pp[0, 0, 1] - pp) - aa * w[0, 0, -1]) / bet
        with interval(-2, -1):
            p1 = t1g * gm / dz * (pem[0, 0, 1] + pp[0, 0, 1])
            gam = aa / bet[0, 0, -1]
            bet = dm - (aa + p1 + aa * gam)
            w = (
                dm * w1 + dt * (pp[0, 0, 1] - pp) - p1 * wsr_top - aa * w[0, 0, -1]
            ) / bet
    with computation(BACKWARD), interval(0, -2):
        w = w - gam[0, 0, 1] * w[0, 0, 1]
    with computation(FORWARD):
        with interval(0, 1):
            pe = 0.0
        with interval(1, None):
            pe = pe[0, 0, -1] + dm[0, 0, -1] * (w[0, 0, -1] - w1[0, 0, -1]) * rdt
    with computation(BACKWARD):
        with interval(-2, -1):
            p1 = (pe + 2.0 * pe[0, 0, 1]) * 1.0 / 3.0
        with interval(0, -2):
            p1 = (pe + bb * pe[0, 0, 1] + g_rat * pe[0, 0, 2]) * 1.0 / 3.0 - g_rat * p1[
                0, 0, 1
            ]
    with computation(PARALLEL), interval(0, -1):
        maxp = p_fac * pm if p_fac * dm > p1 + pm else p1 + pm
        dz = -dm * constants.RDGAS * ptr * exp((cp3 - 1.0) * log(maxp))
    # }


# TODO: implement MOIST_CAPPA=false
def solve(is_, ie, js, je, dt, gm, cp3, pe, dm, pm, pem, w, dz, ptr, wsr):
    grid = spec.grid
    nic = ie - is_ + 1
    njc = je - js + 1
    simshape = pe.shape
    simorigin = (is_, js, 0)
    simdomainplus = (nic, njc, grid.npz + 1)
    t1g = 2.0 * dt * dt
    rdt = 1.0 / dt
    sim1_solver(
        w,
        dm,
        gm,
        dz,
        ptr,
        pm,
        pe,
        pem,
        wsr,
        cp3,
        dt,
        t1g,
        rdt,
        spec.namelist.p_fac,
        origin=simorigin,
        domain=simdomainplus,
    )
