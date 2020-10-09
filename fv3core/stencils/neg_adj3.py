#!/usr/bin/env python3
import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd
ZVIR = constants.RVGAS / constants.RDGAS - 1.0


@gtscript.function
def fix_negative_ice(qv, qi, qs, qg, qr, ql, pt, lcpk, icpk, dq):
    qsum = qi + qs
    if qsum > 0.0:
        if qi < 0.0:
            qi = 0.0
            qs = qsum
        elif qs < 0.0:
            qs = 0.0
            qi = qsum
    else:
        qi = 0.0
        qs = 0.0
        qg = qg + qsum
    if qg < 0.0:
        dq = qs if qs < -qg else -qg
        qs = qs - dq
        qg = qg + dq
        if qg < 0.0:
            dq = qi if qi < -qg else -qg
            qi = qi - dq
            qg = qg + dq
    # If qg still negative, borrow from rain water
    if qg < 0.0 and qr > 0.0:
        dq = qr if qr < -qg else -qg
        qg = qg + dq
        ql = ql - dq
        pt = pt + dq * icpk  # conserve total energy
    # If qg is still negative then borrow from cloud water: phase change
    if qg < 0.0 and ql > 0.0:
        dq = ql if ql < -qg else -qg
        qg = qg + dq
        ql = ql - dq
        pt = pt + dq * icpk
    # Last resort; borrow from water vapor
    if qg < 0.0 and qv > 0.0:
        dq = 0.999 * qv if 0.999 * qv < -qg else -qg
        qg = qg + dq
        qv = qv - dq
        pt = pt + dq * (icpk + lcpk)
    return qv, qi, qs, qg, qr, ql, pt


@gtscript.function
def fix_negative_liq(qv, qi, qs, qg, qr, ql, pt, lcpk, icpk, dq):
    qsum = ql + qr
    posqg = 0.0 if 0.0 > qg else qg
    qrtmp = 0.0
    dq1 = 0.0
    if qsum > 0.0:
        if qr < 0.0:
            qr = 0.0
            ql = qsum
        elif ql < 0.0:
            ql = 0.0
            qr = qsum
    else:
        ql = 0.0
        qrtmp = qsum
        dq = posqg if posqg < -qrtmp else -qrtmp
        qrtmp = qrtmp + dq
        qg = qg - dq
        pt = pt - dq * icpk
        # fill negative rain with available qi & qs (cooling)
        if qr < 0.0:
            dq = qi + qs if (qi + qs) < -qrtmp else -qrtmp
            qrtmp = qrtmp + dq
            dq1 = dq if dq < qs else qs
            qs = qs - dq1
            qi = qi + dq1 - dq
            pt = pt - dq * icpk
        qr = qrtmp
        # fix negative rain water with available vapor
        if qr < 0.0 and qv > 0.0:
            dq = 0.999 * qv if 0.999 * qv < -qr else -qr
            qv = qv - dq
            qr = qr + dq
            pt = pt + dq * lcpk
    return qv, qi, qs, qg, qr, ql, pt


# TODO, turn into stencil (started below, but need refactor to remove floats
def fillq(q, dp, grid):
    for j in range(grid.js, grid.je):
        for i in range(grid.is_, grid.ie + 1):
            sum1 = 0.0
            for k in range(grid.npz):
                if q[i, j, k] > 0.0:
                    sum1 += q[i, j, k] * dp[i, j, k]
            if sum1 < 1.0e-12:
                continue
            sum2 = 0.0
            for k in range(grid.npz - 1, -1, -1):
                if q[i, j, k] < 0.0 and sum1 > 0.0:
                    dq = min(sum1, -q[i, j, k] * dp[i, j, k])
                    sum1 -= dq
                    sum2 += dq
                    q[i, j, k] += dq / dp[i, j, k]
            for k in range(grid.npz - 1, -1, -1):
                if q[i, j, k] > 0.0 and sum2 > 0.0:
                    dq = min(sum2, q[i, j, k] * dp[i, j, k])
                    sum2 -= dq
                    q[i, j, k] -= dq / dp[i, j, k]


"""
# TODO fix this to do fillq with a stencil that validates
# need sum1, sum2 to be an accumulating floats
@gtstencil()
def fillq(q:sd, dp:sd):
    with computation(FORWARD), interval(...):
        if q > 0:
            sum1 = sum1[0, 0, -1] + q * dp
    with computation(BACKWARD), interval(...):
        if q < 0. and sum1 >= 0:
            dq = sum1 if sum1 < -q * dp else -q * dp
            sum1 = sum1 - dq
            sum2 = sum2 + dq
            q = q + dq / dp
    with computation(BACKWARD), interval(...):
        if q > 0. and sum1 >= 1e-12 and sum2 > 0:
            dq = sum2 if sum2 < q * dp else q * dp
            sum2 = sum2 - dq
            q = q - dq / dp
"""


@gtstencil()
def fix_neg_water(
    pt: sd,
    dp: sd,
    delz: sd,
    qv: sd,
    ql: sd,
    qr: sd,
    qs: sd,
    qi: sd,
    qg: sd,
    lv00: float,
    d0_vap: float,
):
    with computation(PARALLEL), interval(...):
        q_liq = 0.0 if 0.0 > ql + qr else ql + qr
        q_sol = 0.0 if 0.0 > qi + qs else qi + qs
        # only for is not GFS_PHYS
        # p2 = -dp / (constants.GRAV * delz) * constants.RDGAS * pt * (1. + ZVIR * qv)
        cpm = (
            (1.0 - (qv + q_liq + q_sol)) * constants.CV_AIR
            + qv * constants.CV_VAP
            + q_liq * constants.C_LIQ
            + q_sol * constants.C_ICE
        )
        lcpk = (lv00 + d0_vap * pt) / cpm
        icpk = (constants.LI0 + constants.DC_ICE * pt) / cpm
        dq = 0.0
        qv, qi, qs, qg, qr, ql, pt = fix_negative_ice(
            qv, qi, qs, qg, qr, ql, pt, lcpk, icpk, dq
        )
        qv, qi, qs, qg, qr, ql, pt = fix_negative_liq(
            qv, qi, qs, qg, qr, ql, pt, lcpk, icpk, dq
        )
        # Fast moist physics: Saturation adjustment
        # no GFS_PHYS compiler flag -- additional saturation adjustment calculations!


@gtstencil()
def fix_neg_cloud(dp: sd, qcld: sd):
    with computation(FORWARD), interval(1, -1):
        if qcld[0, 0, -1] < 0.0:
            qcld = qcld + qcld[0, 0, -1] * dp[0, 0, -1] / dp
    with computation(PARALLEL), interval(1, -1):
        if qcld < 0.0:
            qcld = 0.0
    with computation(FORWARD):
        with interval(-2, -1):
            dq = 0.0
            if qcld[0, 0, 1] < 0.0 and qcld > 0:
                dq = (
                    -qcld * dp
                    if -qcld * dp < qcld[0, 0, 1] * dp[0, 0, 1]
                    else qcld[0, 0, 1] * dp[0, 0, 1]
                )
                qcld = qcld - dq / dp
        with interval(-1, None):
            dq = 0.0
            if qcld < 0 and qcld[0, 0, -1] > 0.0:
                dq = (
                    -qcld * dp
                    if -qcld * dp < qcld[0, 0, -1] * dp[0, 0, -1]
                    else qcld[0, 0, -1] * dp[0, 0, -1]
                )
                qcld = qcld + dq / dp
                qcld = 0.0 if 0.0 > qcld else qcld


# Nonstencil code for reference:
def fix_water_vapor_nonstencil(grid, qv, dp):
    k = 0
    for j in range(grid.js, grid.je + 1):
        for i in range(grid.is_, grid.ie + 1):
            if qv[i, j, k] < 0.0:
                qv[i, j, k + 1] = (
                    qv[i, j, k + 1] + qv[i, j, k] * dp[i, j, k] / dp[i, j, k + 1]
                )

    kbot = grid.npz - 1
    for j in range(grid.js, grid.je + 1):
        for k in range(1, kbot - 1):
            for i in range(grid.is_, grid.ie + 1):
                if qv[i, j, k] < 0 and qv[i, j, k - 1] > 0.0:
                    dq = min(
                        -qv[i, j, k] * dp[i, j, k], qv[i, j, k - 1] * dp[i, j, k - 1]
                    )
                    qv[i, j, k - 1] -= dq / dp[i, j, k - 1]
                    qv[i, j, k] += dq / dp[i, j, k]
                if qv[i, j, k] < 0.0:
                    qv[i, j, k + 1] += qv[i, j, k] * dp[i, j, k] / dp[i, j, k + 1]
                    qv[i, j, k] = 0.0


def fix_water_vapor_bottom(grid, qv, dp):
    kbot = grid.npz - 1
    for j in range(grid.js, grid.je + 1):
        for i in range(grid.is_, grid.ie + 1):
            if qv[i, j, kbot] < 0:
                fix_water_vapor_k_loop(i, j, kbot, qv, dp)


def fix_water_vapor_k_loop(i, j, kbot, qv, dp):
    for k in range(kbot - 1, -1, -1):
        if qv[i, j, kbot] >= 0.0:
            return
        if qv[i, j, k] > 0.0:
            dq = min(-qv[i, j, kbot] * dp[i, j, kbot], qv[i, j, k] * dp[i, j, k])
            qv[i, j, k] = qv[i, j, k] - dq / dp[i, j, k]
            qv[i, j, kbot] = qv[i, j, kbot] + dq / dp[i, j, kbot]


# Stencil version
@gtstencil()
def fix_water_vapor_down(qv: sd, dp: sd, upper_fix: sd, lower_fix: sd, dp_bot: sd):
    with computation(PARALLEL):
        with interval(1, 2):
            if qv[0, 0, -1] < 0:
                qv = qv + qv[0, 0, -1] * dp[0, 0, -1] / dp
        with interval(0, 1):
            qv = qv if qv >= 0 else 0
    with computation(FORWARD), interval(1, -1):
        dq = qv[0, 0, -1] * dp[0, 0, -1]
        if lower_fix[0, 0, -1] != 0:
            qv = qv + lower_fix[0, 0, -1] / dp
        if (qv < 0) and (qv[0, 0, -1] > 0):
            dq = dq if dq < -qv * dp else -qv * dp
            upper_fix = dq
            qv = qv + dq / dp
        if qv < 0:
            lower_fix = qv * dp
            qv = 0
    with computation(PARALLEL), interval(0, -2):
        if upper_fix[0, 0, 1] != 0:
            qv = qv - upper_fix[0, 0, 1] / dp
    with computation(PARALLEL), interval(-1, None):
        if lower_fix[0, 0, -1] > 0:
            qv = qv + lower_fix / dp
        # Here we're re-using upper_fix to represent the current version of qv[k_bot] fixed from above
        # we could also re-use lower_fix instead of dp_bot, but that's probably over-optimized for now
        upper_fix = qv
        # if we didn't have to worry about float valitation and negative column mass we could set qv[k_bot] to 0 here...
    with computation(BACKWARD), interval(0, -1):
        dq = qv * dp
        if (upper_fix[0, 0, 1] < 0) and (qv > 0):
            dq = (
                dq
                if dq < -upper_fix[0, 0, 1] * dp_bot
                else -upper_fix[0, 0, 1] * dp_bot
            )
            qv = qv - dq / dp
            upper_fix = upper_fix[0, 0, 1] + dq / dp_bot
        else:
            upper_fix = upper_fix[0, 0, 1]


def compute(qvapor, qliquid, qrain, qsnow, qice, qgraupel, qcld, pt, delp, delz, peln):
    grid = spec.grid
    i_ext = grid.domain_shape_compute()[0]
    j_ext = grid.domain_shape_compute()[1]
    k_ext = grid.domain_shape_compute()[2]
    if spec.namelist.check_negative:
        raise Exception("Unimplemented namelist value check_negative=True")
    if spec.namelist.hydrostatic:
        d0_vap = constants.CP_VAP - constants.C_LIQ
        raise Exception("Unimplemented namelist hydrostatic=True")
    else:
        d0_vap = constants.CV_VAP - constants.C_LIQ
    lv00 = constants.HLV - d0_vap * constants.TICE
    fix_neg_water(
        pt,
        delp,
        delz,
        qvapor,
        qliquid,
        qrain,
        qsnow,
        qice,
        qgraupel,
        lv00,
        d0_vap,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )
    fillq(qgraupel, delp, grid)
    fillq(qrain, delp, grid)
    # # fix_water_vapor(delp, qvapor, origin=grid.compute_origin(), domain=grid.domain_shape_compute())
    # fix_water_vapor_nonstencil(grid, qvapor, delp)
    # fix_water_vapor_bottom(grid, qvapor, delp)
    upper_fix = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    lower_fix = utils.make_storage_from_shape(qvapor.shape, origin=(0, 0, 0))
    bot_dp = delp[:, :, grid.npz - 1]
    full_bot_arr = utils.repeat(bot_dp[:, :, np.newaxis], k_ext + 1, axis=2)
    dp_bot = utils.make_storage_data(full_bot_arr, full_bot_arr.shape)
    fix_water_vapor_down(
        qvapor,
        delp,
        upper_fix,
        lower_fix,
        dp_bot,
        origin=grid.compute_origin(),
        domain=grid.domain_shape_compute(),
    )
    qvapor[:, :, grid.npz] = upper_fix[:, :, 0]
    fix_neg_cloud(
        delp, qcld, origin=grid.compute_origin(), domain=grid.domain_shape_compute()
    )
