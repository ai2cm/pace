#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.copy_stencil as cp
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd


@utils.stencil()
def initial(
    w: sd,
    dm: sd,
    gm: sd,
    dz: sd,
    ptr: sd,
    pm: sd,
    pe: sd,
    g_rat: sd,
    bb: sd,
    dd: sd,
    w1: sd,
):
    with computation(PARALLEL), interval(...):
        w1 = w
        # pe = (-dm / dz * constants.RDGAS * ptr)**gm - pm
    with computation(PARALLEL):
        with interval(0, -1):
            g_rat = dm / dm[0, 0, 1]
            bb = 2.0 * (1.0 + g_rat)
            dd = 3.0 * (pe + g_rat * pe[0, 0, 1])
        with interval(-1, None):
            bb = 2.0
            dd = 3.0 * pe


@utils.stencil()
def w_solver(
    aa: sd,
    bet: sd,
    g_rat: sd,
    gam: sd,
    pp: sd,
    dd: sd,
    gm: sd,
    dz: sd,
    pem: sd,
    dm: sd,
    pe: sd,
    bb: sd,
    t1g: float,
):
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


@utils.stencil()
def w_pe_dz_compute(
    dm: sd,
    w1: sd,
    pp: sd,
    aa: sd,
    gm: sd,
    dz: sd,
    pem: sd,
    wsr_top: sd,
    bb: sd,
    g_rat: sd,
    bet: sd,
    gam: sd,
    p1: sd,
    pe: sd,
    w: sd,
    pm: sd,
    ptr: sd,
    cp3: sd,
    maxp: sd,
    dt: float,
    t1g: float,
    rdt: float,
    p_fac: float,
):
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
    with computation(BACKWARD), interval(0, -1):
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
        # dz = -dm * constants.RDGAS * ptr * exp((cp3 - 1.0) * log(maxp)
        # dz = -dm * constants.RDGAS * ptr * maxp ** (cp3 - 1.0)


# TODO: implement MOIST_CAPPA=false
def solve(is_, ie, js, je, dt, gm, cp3, pe, dm, pm, pem, w, dz, ptr, wsr):
    grid = spec.grid
    nic = ie - is_ + 1
    njc = je - js + 1
    simshape = pe.shape
    simorigin = (is_, js, 0)
    simdomain = (nic, njc, grid.npz)
    simdomainplus = (nic, njc, grid.npz + 1)
    g_rat = utils.make_storage_from_shape(simshape, simorigin)
    bb = utils.make_storage_from_shape(simshape, simorigin)
    aa = utils.make_storage_from_shape(simshape, simorigin)
    dd = utils.make_storage_from_shape(simshape, simorigin)
    gam = utils.make_storage_from_shape(simshape, simorigin)
    w1 = utils.make_storage_from_shape(simshape, simorigin)
    pp = utils.make_storage_from_shape(simshape, simorigin)
    p1 = utils.make_storage_from_shape(simshape, simorigin)
    pp = utils.make_storage_from_shape(simshape, simorigin)
    t1g = 2.0 * dt * dt
    rdt = 1.0 / dt
    tmpslice = (slice(is_, ie + 1), slice(js, je + 1), slice(0, grid.npz))
    # putting this into stencil removing the exp and log from the equation makes it not validate
    pe[tmpslice] = (
        np.exp(
            gm[tmpslice]
            * np.log(-dm[tmpslice] / dz[tmpslice] * constants.RDGAS * ptr[tmpslice])
        )
        - pm[tmpslice]
    )
    initial(
        w,
        dm,
        gm,
        dz,
        ptr,
        pm,
        pe,
        g_rat,
        bb,
        dd,
        w1,
        origin=simorigin,
        domain=simdomain,
    )
    bet = utils.make_storage_data(bb.data[:, :, 0], simshape)
    w_solver(
        aa,
        bet,
        g_rat,
        gam,
        pp,
        dd,
        gm,
        dz,
        pem,
        dm,
        pe,
        bb,
        t1g,
        origin=simorigin,
        domain=simdomainplus,
    )

    # reset bet column to the new value. TODO reuse the same storage
    bet = utils.make_storage_data(dm.data[:, :, 0] - aa.data[:, :, 1], simshape)
    wsr_top = utils.make_storage_data(wsr.data[:, :, 0], simshape)
    # TODO remove when put exponential function into stencil
    maxp = utils.make_storage_from_shape(simshape, simorigin)
    w_pe_dz_compute(
        dm,
        w1,
        pp,
        aa,
        gm,
        dz,
        pem,
        wsr_top,
        bb,
        g_rat,
        bet,
        gam,
        p1,
        pe,
        w,
        pm,
        ptr,
        cp3,
        maxp,
        dt,
        t1g,
        rdt,
        spec.namelist["p_fac"],
        origin=simorigin,
        domain=simdomainplus,
    )
    # TODO put back into w_pe_dz stencil when have exp and log
    dz[tmpslice] = (
        -dm[tmpslice]
        * constants.RDGAS
        * ptr[tmpslice]
        * np.exp((cp3[tmpslice] - 1.0) * np.log(maxp[tmpslice]))
    )
