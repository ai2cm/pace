import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval

import pace.util
import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util import X_DIM, Y_DIM


ZVIR = constants.RVGAS / constants.RDGAS - 1.0


@gtscript.function
def fix_negative_ice(qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt, lcpk, icpk, dq):
    qsum = qice + qsnow
    if qsum > 0.0:
        if qice < 0.0:
            qice = 0.0
            qsnow = qsum
        elif qsnow < 0.0:
            qsnow = 0.0
            qice = qsum
    else:
        qice = 0.0
        qsnow = 0.0
        qgraupel = qgraupel + qsum
    if qgraupel < 0.0:
        dq = qsnow if qsnow < -qgraupel else -qgraupel
        qsnow = qsnow - dq
        qgraupel = qgraupel + dq
        if qgraupel < 0.0:
            dq = qice if qice < -qgraupel else -qgraupel
            qice = qice - dq
            qgraupel = qgraupel + dq
    # If qgraupel still negative, borrow from rain water
    if qgraupel < 0.0 and qrain > 0.0:
        dq = qrain if qrain < -qgraupel else -qgraupel
        qgraupel = qgraupel + dq
        qliquid = qliquid - dq
        pt = pt + dq * icpk  # conserve total energy
    # If qgraupel is still negative then borrow from cloud water: phase change
    if qgraupel < 0.0 and qliquid > 0.0:
        dq = qliquid if qliquid < -qgraupel else -qgraupel
        qgraupel = qgraupel + dq
        qliquid = qliquid - dq
        pt = pt + dq * icpk
    # Last resort; borrow from water vapor
    if qgraupel < 0.0 and qvapor > 0.0:
        dq = 0.999 * qvapor if 0.999 * qvapor < -qgraupel else -qgraupel
        qgraupel = qgraupel + dq
        qvapor = qvapor - dq
        pt = pt + dq * (icpk + lcpk)
    return qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt


@gtscript.function
def fix_negative_liq(qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt, lcpk, icpk, dq):
    qsum = qliquid + qrain
    pos_qgraupel = 0.0 if 0.0 > qgraupel else qgraupel
    qrain_tmp = 0.0
    dq1 = 0.0
    if qsum > 0.0:
        if qrain < 0.0:
            qrain = 0.0
            qliquid = qsum
        elif qliquid < 0.0:
            qliquid = 0.0
            qrain = qsum
    else:
        qliquid = 0.0
        qrain_tmp = qsum
        dq = pos_qgraupel if pos_qgraupel < -qrain_tmp else -qrain_tmp
        qrain_tmp = qrain_tmp + dq
        qgraupel = qgraupel - dq
        pt = pt - dq * icpk
        # fill negative rain with available qice & qsnow (cooling)
        if qrain < 0.0:
            dq = qice + qsnow if (qice + qsnow) < -qrain_tmp else -qrain_tmp
            qrain_tmp = qrain_tmp + dq
            dq1 = dq if dq < qsnow else qsnow
            qsnow = qsnow - dq1
            qice = qice + dq1 - dq
            pt = pt - dq * icpk
        qrain = qrain_tmp
        # fix negative rain water with available vapor
        if qrain < 0.0 and qvapor > 0.0:
            dq = 0.999 * qvapor if 0.999 * qvapor < -qrain else -qrain
            qvapor = qvapor - dq
            qrain = qrain + dq
            pt = pt + dq * lcpk
    return qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt


def fix_neg_water(
    pt: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qsnow: FloatField,
    qice: FloatField,
    qgraupel: FloatField,
    lv00: float,
    d0_vap: float,
):
    """
    Args:
        pt (inout):
        qvapor (inout):
        qliquid (inout):
        qrain (inout):
        qsnow (inout):
        qice (inout):
        qgraupel (inout):
    """
    with computation(PARALLEL), interval(...):
        q_liq = 0.0 if 0.0 > qliquid + qrain else qliquid + qrain
        q_sol = 0.0 if 0.0 > qice + qsnow else qice + qsnow
        cpm = (
            (1.0 - (qvapor + q_liq + q_sol)) * constants.CV_AIR
            + qvapor * constants.CV_VAP
            + q_liq * constants.C_LIQ
            + q_sol * constants.C_ICE
        )
        lcpk = (lv00 + d0_vap * pt) / cpm
        icpk = (constants.LI0 + constants.DC_ICE * pt) / cpm
        dq = 0.0
        qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt = fix_negative_ice(
            qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt, lcpk, icpk, dq
        )
        qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt = fix_negative_liq(
            qvapor, qice, qsnow, qgraupel, qrain, qliquid, pt, lcpk, icpk, dq
        )
        # Fast moist physics: Saturation adjustment
        # no GFS_PHYS compiler flag -- additional saturation adjustment calculations!


def fillq(q: FloatField, dp: FloatField, sum1: FloatFieldIJ, sum2: FloatFieldIJ):
    """
    Args:
        q (inout):
        dp (in):
        sum1 (out):
        sum2 (out):
    """
    with computation(FORWARD), interval(...):
        # reset accumulating fields
        sum1 = 0.0
        sum2 = 0.0
    with computation(FORWARD), interval(...):
        if q > 0:
            sum1 = sum1 + q * dp
    with computation(BACKWARD), interval(...):
        if q < 0.0 and sum1 >= 0:
            dq = sum1 if sum1 < -q * dp else -q * dp
            sum1 = sum1 - dq
            sum2 = sum2 + dq
            q = q + dq / dp
    with computation(BACKWARD), interval(...):
        if q > 0.0 and sum1 >= 1e-12 and sum2 > 0:
            dq = sum2 if sum2 < q * dp else q * dp
            sum2 = sum2 - dq
            q = q - dq / dp


# Stencil version
def fix_water_vapor_down(qvapor: FloatField, dp: FloatField):
    """
    Args:
        qvapor (inout):
        dp (in):
    """
    with computation(PARALLEL), interval(...):
        upper_fix = 0.0  # type: FloatField
        lower_fix = 0.0  # type: FloatField
    with computation(BACKWARD):
        with interval(1, 2):
            if qvapor[0, 0, -1] < 0:  # top level is negative
                # reduce level 1 by that amount to compensate:
                qvapor = qvapor + qvapor[0, 0, -1] * dp[0, 0, -1] / dp
        with interval(0, 1):
            if qvapor < 0.0:
                qvapor = 0.0  # top level is now 0
    with computation(FORWARD), interval(1, -1):
        dq = qvapor[0, 0, -1] * dp[0, 0, -1]
        # if we borrowed from this level to fix the upper level, account for that here:
        if lower_fix[0, 0, -1] != 0:
            qvapor += lower_fix[0, 0, -1] / dp
        # if we're now negative and can borrow from above do so:
        if (qvapor < 0) and (qvapor[0, 0, -1] > 0):
            dq = dq if dq < -qvapor * dp else -qvapor * dp
            upper_fix = dq
            qvapor += dq / dp
        if qvapor < 0:  # If still negative borrow from below
            lower_fix = qvapor * dp
            qvapor = 0
    with computation(PARALLEL), interval(0, -2):
        # if we had to borrow from upper levels before account for that in this loop
        if upper_fix[0, 0, 1] != 0:
            qvapor = qvapor - upper_fix[0, 0, 1] / dp
    with computation(PARALLEL), interval(-1, None):
        # if we borrowed from the bottom level account for that here:
        if lower_fix[0, 0, -1] > 0:
            qvapor = qvapor + lower_fix / dp
        # Here we're re-using upper_fix to represent the current version of
        # qvapor[k_bot] fixed from above. We could also re-use lower_fix instead of
        # dp_bot, but that's probably over-optimized for now
        upper_fix = qvapor
        # If we didn't have to worry about float valitation and negative column
        # mass we could set qvapor[k_bot] to 0 here...
        dp_bottom = dp
    with computation(BACKWARD), interval(0, -1):
        dp_bottom = dp_bottom[0, 0, 1]
        dq = qvapor * dp
        # (if qvapor[kbot] isn't negative we will just loop through and do nothing)
        # if the level below us is negative and we are positive:
        if (upper_fix[0, 0, 1] < 0) and (qvapor > 0):
            # AND if we have enough mass to fill the level below us:
            if dq >= -upper_fix[0, 0, 1] * dp_bottom:
                # set dq to the amount needed to fill the level below us
                dq = -upper_fix[0, 0, 1] * dp_bottom
                # (otherwise dq is all of the vapor mass)
            qvapor = qvapor - dq / dp  # subtract dq from current mass
            upper_fix = upper_fix[0, 0, 1] + dq / dp_bottom  # add mass to qvapor[kbot]
        # if qvapor[kbot] is still negative move to the next level
        else:
            upper_fix = upper_fix[
                0, 0, 1
            ]  # now upper_fix[k] is q[kbot] after adjusting
    # after that loop upper_fix[ktop] = qvapot[kbot] so bring that value back to kbot
    with computation(FORWARD), interval(1, None):
        upper_fix = upper_fix[0, 0, -1]
    with computation(PARALLEL), interval(-1, None):
        qvapor = upper_fix  # and finally set qvapor[kbot]


def fix_neg_cloud(dp: FloatField, qcld: FloatField):
    with computation(FORWARD), interval(1, -1):
        if qcld[0, 0, -1] < 0.0:
            qcld = qcld + qcld[0, 0, -1] * dp[0, 0, -1] / dp
    with computation(PARALLEL), interval(1, -1):
        if qcld < 0.0:
            qcld = 0.0
    with computation(FORWARD):
        with interval(-2, -1):
            if qcld[0, 0, 1] < 0.0 and qcld > 0:
                dq = (
                    -qcld * dp
                    if -qcld * dp < qcld[0, 0, 1] * dp[0, 0, 1]
                    else qcld[0, 0, 1] * dp[0, 0, 1]
                )
                qcld = qcld - dq / dp
        with interval(-1, None):
            if qcld < 0 and qcld[0, 0, -1] > 0.0:
                dq = (
                    -qcld * dp
                    if -qcld * dp < qcld[0, 0, -1] * dp[0, 0, -1]
                    else qcld[0, 0, -1] * dp[0, 0, -1]
                )
                qcld = qcld + dq / dp
                qcld = 0.0 if 0.0 > qcld else qcld


"""
Nonstencil code for reference:

def fix_water_vapor_nonstencil(grid, qvapor, dp):
    k = 0
    for j in range(grid.js, grid.je + 1):
        for i in range(grid.is_, grid.ie + 1):
            if qvapor[i, j, k] < 0.0:
                qvapor[i, j, k + 1] = (
                    qvapor[i, j, k + 1]
                    + qvapor[i, j, k] * dp[i, j, k] / dp[i, j, k + 1]
                )

    kbot = grid_indexing.domain[2] - 1
    for j in range(grid.js, grid.je + 1):
        for k in range(1, kbot - 1):
            for i in range(grid.is_, grid.ie + 1):
                if qvapor[i, j, k] < 0 and qvapor[i, j, k - 1] > 0.0:
                    dq = min(
                        -qvapor[i, j, k] * dp[i, j, k],
                        qvapor[i, j, k - 1] * dp[i, j, k - 1],
                    )
                    qvapor[i, j, k - 1] -= dq / dp[i, j, k - 1]
                    qvapor[i, j, k] += dq / dp[i, j, k]
                if qvapor[i, j, k] < 0.0:
                    qvapor[i, j, k + 1] += (
                        qvapor[i, j, k] * dp[i, j, k] / dp[i, j, k + 1]
                    )
                    qvapor[i, j, k] = 0.0


def fix_water_vapor_bottom(grid, qvapor, dp):
    kbot = grid_indexing.domain[2] - 1
    for j in range(grid.js, grid.je + 1):
        for i in range(grid.is_, grid.ie + 1):
            if qvapor[i, j, kbot] < 0:
                fix_water_vapor_k_loop(i, j, kbot, qvapor, dp)


def fix_water_vapor_k_loop(i, j, kbot, qvapor, dp):
    for k in range(kbot - 1, -1, -1):
        if qvapor[i, j, kbot] >= 0.0:
            return
        if qvapor[i, j, k] > 0.0:
            dq = min(
                -qvapor[i, j, kbot] * dp[i, j, kbot], qvapor[i, j, k] * dp[i, j, k]
            )
            qvapor[i, j, k] = qvapor[i, j, k] - dq / dp[i, j, k]
            qvapor[i, j, kbot] = qvapor[i, j, kbot] + dq / dp[i, j, kbot]
"""


class AdjustNegativeTracerMixingRatio:
    """Adjust tracer mixing ratios to fix negative values

    Named neg_adj3 in fortran

    Args:
        qvapor: Water vapor mixing ration (inout)
        qliquid: Liquid water mixing ration (inout)
        qrain: Rain mixing ration (inout)
        qsnow: Snow mixing ration (inout)
        qice: Ice mixing ration (inout)
        qgraupel: Graupel mixing ration (inout)
        qcld: Cloud mixing ration (inout)
        pt: Air temperature (in)
        delp: Pressur thickness of atmosphere layers (in)
        delz: Vertical thickness of atmosphere layers (in)
        peln: Logarithm of interface pressure (in)
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
        check_negative: bool,
        hydrostatic: bool,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._sum1 = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown")
        self._sum2 = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown")
        if check_negative:
            raise NotImplementedError(
                "Unimplemented namelist value check_negative=True"
            )
        if hydrostatic:
            self._d0_vap = constants.CP_VAP - constants.C_LIQ
            raise NotImplementedError("Unimplemented namelist hydrostatic=True")
        else:
            self._d0_vap = constants.CV_VAP - constants.C_LIQ
        self._lv00 = constants.HLV - self._d0_vap * constants.TICE

        self._fix_neg_water = stencil_factory.from_origin_domain(
            func=fix_neg_water,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._fillq = stencil_factory.from_origin_domain(
            func=fillq,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._fix_water_vapor_down = stencil_factory.from_origin_domain(
            func=fix_water_vapor_down,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._fix_neg_cloud = stencil_factory.from_origin_domain(
            func=fix_neg_cloud,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        qvapor,
        qliquid,
        qrain,
        qsnow,
        qice,
        qgraupel,
        qcld,
        pt,
        delp,
    ):
        """
        Args:
            qvapor (inout):
            qliquid (inout):
            qrain (inout):
            qsnow (inout):
            qice (inout):
            qgraupel (inout):
            qcld (inout):
            pt (inout):
            delp (in):
        """
        # TODO: remove delz and peln from args
        self._fix_neg_water(
            pt,
            qvapor,
            qliquid,
            qrain,
            qsnow,
            qice,
            qgraupel,
            self._lv00,
            self._d0_vap,
        )
        # TODO - optimisation: those could be merged into one stencil. To keep
        # the physical meaning we could keep the structure as @gtstencil.function
        # TODO: when gt4py supports 2D temporaries, refactor sum1 and sum2 to internal
        # stencil temporaries
        self._fillq(qgraupel, delp, self._sum1, self._sum2)
        self._fillq(qrain, delp, self._sum1, self._sum2)
        self._fix_water_vapor_down(qvapor, delp)
        self._fix_neg_cloud(delp, qcld)
