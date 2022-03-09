# mypy: ignore-errors
import collections

import gt4py.gtscript as gtscript
from gt4py.gtscript import __INLINED, BACKWARD, PARALLEL, computation, interval

import pace.dsl.gt4py_utils as utils
from fv3core.initialization.dycore_state import DycoreState
from fv3core.stencils.basic_operations import dim
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField
from pace.util.constants import (
    C_ICE,
    C_LIQ,
    CP_AIR,
    CP_VAP,
    CV_AIR,
    CV_VAP,
    GRAV,
    RDGAS,
    ZVIR,
)
from pace.util.quantity import Quantity


RK = CP_AIR / RDGAS + 1.0
G2 = 0.5 * GRAV
T1_MIN = 160.0
T2_MIN = 165.0
T2_MAX = 315.0
T3_MAX = 325.0
USTAR2 = 1.0e-4
RI_MAX = 1.0
RI_MIN = 0.25


@gtscript.function
def standard_cm(cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel):
    q_liq = q0_liquid + q0_rain
    q_sol = q0_ice + q0_snow + q0_graupel
    cpm = (
        (1.0 - (q0_vapor + q_liq + q_sol)) * CP_AIR
        + q0_vapor * CP_VAP
        + q_liq * C_LIQ
        + q_sol * C_ICE
    )
    cvm = (
        (1.0 - (q0_vapor + q_liq + q_sol)) * CV_AIR
        + q0_vapor * CV_VAP
        + q_liq * C_LIQ
        + q_sol * C_ICE
    )
    return cpm, cvm


@gtscript.function
def tvol(gz, u0, v0, w0):
    return gz + 0.5 * (u0 ** 2 + v0 ** 2 + w0 ** 2)


def init(
    gz: FloatField,
    t0: FloatField,
    u0: FloatField,
    v0: FloatField,
    w0: FloatField,
    static_energy: FloatField,
    cvm: FloatField,
    cpm: FloatField,
    total_energy: FloatField,
    ua: FloatField,
    va: FloatField,
    w: FloatField,
    ta: FloatField,
    delz: FloatField,
    q0_vapor: FloatField,
    q0_liquid: FloatField,
    q0_rain: FloatField,
    q0_ice: FloatField,
    q0_snow: FloatField,
    q0_graupel: FloatField,
    q0_o3mr: FloatField,
    q0_sgs_tke: FloatField,
    q0_cld: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qsgs_tke: FloatField,
    qcld: FloatField,
):

    with computation(PARALLEL), interval(...):
        t0 = ta
        u0 = ua
        v0 = va
        w0 = w
        # TODO: in a loop over tracers
        q0_vapor = qvapor
        q0_liquid = qliquid
        q0_rain = qrain
        q0_ice = qice
        q0_snow = qsnow
        q0_graupel = qgraupel
        q0_o3mr = qo3mr
        q0_sgs_tke = qsgs_tke
        q0_cld = qcld
        gzh = 0.0
    with computation(BACKWARD), interval(0, -1):
        # note only for nwat = 6
        cpm, cvm = standard_cm(
            cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel
        )
        gz = gzh[0, 0, 1] - G2 * delz
        tmp = tvol(gz, u0, v0, w0)
        static_energy = cpm * t0 + tmp
        total_energy = cvm * t0 + tmp
        gzh = gzh[0, 0, 1] - GRAV * delz


@gtscript.function
def qcon_func(q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel):
    return q0_liquid + q0_ice + q0_snow + q0_rain + q0_graupel


@gtscript.function
def adjust_cvm(
    cpm,
    cvm,
    q0_vapor,
    q0_liquid,
    q0_rain,
    q0_ice,
    q0_snow,
    q0_graupel,
    gz,
    u0,
    v0,
    w0,
    t0,
    total_energy,
    static_energy,
):
    """
    Non-hydrostatic under constant volume heating/cooling
    """
    cpm, cvm = standard_cm(
        cpm, cvm, q0_vapor, q0_liquid, q0_rain, q0_ice, q0_snow, q0_graupel
    )
    tv = tvol(gz, u0, v0, w0)
    t0 = (total_energy - tv) / cvm
    static_energy = cpm * t0 + tv
    return cpm, cvm, t0, static_energy


@gtscript.function
def compute_richardson_number(
    t0, q0_vapor, qcon, pkz, delp, peln, gz, u0, v0, xvir, t_max, t_min
):
    tv1 = t0[0, 0, -1] * (1.0 + xvir * q0_vapor[0, 0, -1] - qcon[0, 0, -1])
    tv2 = t0 * (1.0 + xvir * q0_vapor - qcon)
    pt1 = tv1 / pkz[0, 0, -1]
    pt2 = tv2 / pkz
    ri = (
        (gz[0, 0, -1] - gz)
        * (pt1 - pt2)
        / (
            0.5
            * (pt1 + pt2)
            * ((u0[0, 0, -1] - u0) ** 2 + (v0[0, 0, -1] - v0) ** 2 + USTAR2)
        )
    )
    if tv1 > t_max and tv1 > tv2:
        ri = 0
    elif tv2 < t_min:
        ri = ri if ri < 0.1 else 0.1
    ri_ref = (
        RI_MIN
        + (RI_MAX - RI_MIN) * dim(400.0e2, delp / (peln[0, 0, 1] - peln)) / 200.0e2
    )
    if RI_MAX < ri_ref:
        ri_ref = RI_MAX
    return ri, ri_ref


@gtscript.function
def compute_mass_flux(ri, ri_ref, delp, ratio):
    max_ri_ratio = ri / ri_ref
    mc = 0.0
    if max_ri_ratio < 0.0:
        max_ri_ratio = 0.0
    if ri < ri_ref:
        mc = (
            ratio
            * delp[0, 0, -1]
            * delp
            / (delp[0, 0, -1] + delp)
            * (1.0 - max_ri_ratio) ** 2.0
        )
    return mc


@gtscript.function
def kh_adjust_down(mc, delp, q0, h0):
    h0 = mc * (q0 - q0[0, 0, -1])
    return q0 - h0 / delp, h0


@gtscript.function
def kh_adjust_energy_down(mc, delp, static_energy, total_energy, h0):
    h0 = mc * (static_energy - static_energy[0, 0, -1])
    return total_energy - h0 / delp, h0


@gtscript.function
def kh_adjust_up(delp, h0, q0):
    return q0 + h0[0, 0, 1] / delp


def m_loop(
    u0: FloatField,
    v0: FloatField,
    w0: FloatField,
    t0: FloatField,
    static_energy: FloatField,
    gz: FloatField,
    delp: FloatField,
    peln: FloatField,
    pkz: FloatField,
    q0_vapor: FloatField,
    q0_liquid: FloatField,
    q0_rain: FloatField,
    q0_ice: FloatField,
    q0_snow: FloatField,
    q0_graupel: FloatField,
    q0_o3mr: FloatField,
    q0_sgs_tke: FloatField,
    q0_cld: FloatField,
    total_energy: FloatField,
    cpm: FloatField,
    cvm: FloatField,
    t_min: float,
    ratio: float,
):
    from __externals__ import t_max, xvir

    with computation(PARALLEL), interval(...):
        qcon = qcon_func(q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
        h0_vapor = 0.0
        h0_liquid = 0.0
        h0_rain = 0.0
        h0_ice = 0.0
        h0_snow = 0.0
        h0_graupel = 0.0
        h0_o3mr = 0.0
        h0_sgs_tke = 0.0
        h0_cld = 0.0
        h0_u = 0.0
        h0_v = 0.0
        h0_w = 0.0
        h0_total_energy = 0.0
        ri = 0.0
        ref = 0.0
    with computation(BACKWARD):
        with interval(-1, None):
            ri, ri_ref = compute_richardson_number(
                t0, q0_vapor, qcon, pkz, delp, peln, gz, u0, v0, xvir, t_max, t_min
            )
            mc = compute_mass_flux(ri, ri_ref, delp, ratio)
            if ri < ri_ref:
                # TODO: loop over tracers not hardcoded
                # Note combining into functions breaks
                # validation, may want to try again with changes to gt4py
                q0_vapor, h0_vapor = kh_adjust_down(mc, delp, q0_vapor, h0_vapor)
                q0_liquid, h0_liquid = kh_adjust_down(mc, delp, q0_liquid, h0_liquid)
                q0_rain, h0_rain = kh_adjust_down(mc, delp, q0_rain, h0_rain)
                q0_ice, h0_ice = kh_adjust_down(mc, delp, q0_ice, h0_ice)
                q0_snow, h0_snow = kh_adjust_down(mc, delp, q0_snow, h0_snow)
                q0_graupel, h0_graupel = kh_adjust_down(
                    mc, delp, q0_graupel, h0_graupel
                )
                q0_o3mr, h0_o3mr = kh_adjust_down(mc, delp, q0_o3mr, h0_o3mr)
                q0_sgs_tke, h0_sgs_tke = kh_adjust_down(
                    mc, delp, q0_sgs_tke, h0_sgs_tke
                )
                q0_cld, h0_cld = kh_adjust_down(mc, delp, q0_cld, h0_cld)
                u0, h0_u = kh_adjust_down(mc, delp, u0, h0_u)
                v0, h0_v = kh_adjust_down(mc, delp, v0, h0_v)
                w0, h0_w = kh_adjust_down(mc, delp, w0, h0_w)
                total_energy, h0_total_energy = kh_adjust_energy_down(
                    mc, delp, static_energy, total_energy, h0_total_energy
                )
            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
        with interval(4, -1):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = kh_adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = kh_adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = kh_adjust_up(delp, h0_rain, q0_rain)
                q0_ice = kh_adjust_up(delp, h0_ice, q0_ice)
                q0_snow = kh_adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = kh_adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = kh_adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = kh_adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = kh_adjust_up(delp, h0_cld, q0_cld)
                qcon = qcon_func(q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = kh_adjust_up(delp, h0_u, u0)
                v0 = kh_adjust_up(delp, h0_v, v0)
                w0 = kh_adjust_up(delp, h0_w, w0)
                total_energy = kh_adjust_up(delp, h0_total_energy, total_energy)
            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
            ri, ri_ref = compute_richardson_number(
                t0, q0_vapor, qcon, pkz, delp, peln, gz, u0, v0, xvir, t_max, t_min
            )
            mc = compute_mass_flux(ri, ri_ref, delp, ratio)
            if ri < ri_ref:
                q0_vapor, h0_vapor = kh_adjust_down(mc, delp, q0_vapor, h0_vapor)
                q0_liquid, h0_liquid = kh_adjust_down(mc, delp, q0_liquid, h0_liquid)
                q0_rain, h0_rain = kh_adjust_down(mc, delp, q0_rain, h0_rain)
                q0_ice, h0_ice = kh_adjust_down(mc, delp, q0_ice, h0_ice)
                q0_snow, h0_snow = kh_adjust_down(mc, delp, q0_snow, h0_snow)
                q0_graupel, h0_graupel = kh_adjust_down(
                    mc, delp, q0_graupel, h0_graupel
                )
                q0_o3mr, h0_o3mr = kh_adjust_down(mc, delp, q0_o3mr, h0_o3mr)
                q0_sgs_tke, h0_sgs_tke = kh_adjust_down(
                    mc, delp, q0_sgs_tke, h0_sgs_tke
                )
                q0_cld, h0_cld = kh_adjust_down(mc, delp, q0_cld, h0_cld)
                u0, h0_u = kh_adjust_down(mc, delp, u0, h0_u)
                v0, h0_v = kh_adjust_down(mc, delp, v0, h0_v)
                w0, h0_w = kh_adjust_down(mc, delp, w0, h0_w)
                total_energy, h0_total_energy = kh_adjust_energy_down(
                    mc, delp, static_energy, total_energy, h0_total_energy
                )
            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
        with interval(3, 4):
            # TODO: this is repetitive, but using functions did not work as
            # expected. spend some more time here so not so much needs
            # to be repeated just to multiply ri_ref by a constant
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = kh_adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = kh_adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = kh_adjust_up(delp, h0_rain, q0_rain)
                q0_ice = kh_adjust_up(delp, h0_ice, q0_ice)
                q0_snow = kh_adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = kh_adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = kh_adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = kh_adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = kh_adjust_up(delp, h0_cld, q0_cld)
                qcon = qcon_func(q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = kh_adjust_up(delp, h0_u, u0)
                v0 = kh_adjust_up(delp, h0_v, v0)
                w0 = kh_adjust_up(delp, h0_w, w0)
                total_energy = kh_adjust_up(delp, h0_total_energy, total_energy)

            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
            ri, ri_ref = compute_richardson_number(
                t0, q0_vapor, qcon, pkz, delp, peln, gz, u0, v0, xvir, t_max, t_min
            )
            # TODO, can we just check if index(K) == 3?
            ri_ref = ri_ref * 1.5
            mc = compute_mass_flux(ri, ri_ref, delp, ratio)
            if ri < ri_ref:
                q0_vapor, h0_vapor = kh_adjust_down(mc, delp, q0_vapor, h0_vapor)
                q0_liquid, h0_liquid = kh_adjust_down(mc, delp, q0_liquid, h0_liquid)
                q0_rain, h0_rain = kh_adjust_down(mc, delp, q0_rain, h0_rain)
                q0_ice, h0_ice = kh_adjust_down(mc, delp, q0_ice, h0_ice)
                q0_snow, h0_snow = kh_adjust_down(mc, delp, q0_snow, h0_snow)
                q0_graupel, h0_graupel = kh_adjust_down(
                    mc, delp, q0_graupel, h0_graupel
                )
                q0_o3mr, h0_o3mr = kh_adjust_down(mc, delp, q0_o3mr, h0_o3mr)
                q0_sgs_tke, h0_sgs_tke = kh_adjust_down(
                    mc, delp, q0_sgs_tke, h0_sgs_tke
                )
                q0_cld, h0_cld = kh_adjust_down(mc, delp, q0_cld, h0_cld)
                u0, h0_u = kh_adjust_down(mc, delp, u0, h0_u)
                v0, h0_v = kh_adjust_down(mc, delp, v0, h0_v)
                w0, h0_w = kh_adjust_down(mc, delp, w0, h0_w)
                total_energy, h0_total_energy = kh_adjust_energy_down(
                    mc, delp, static_energy, total_energy, h0_total_energy
                )
            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
        with interval(2, 3):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = kh_adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = kh_adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = kh_adjust_up(delp, h0_rain, q0_rain)
                q0_ice = kh_adjust_up(delp, h0_ice, q0_ice)
                q0_snow = kh_adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = kh_adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = kh_adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = kh_adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = kh_adjust_up(delp, h0_cld, q0_cld)
                qcon = qcon_func(q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = kh_adjust_up(delp, h0_u, u0)
                v0 = kh_adjust_up(delp, h0_v, v0)
                w0 = kh_adjust_up(delp, h0_w, w0)
                total_energy = kh_adjust_up(delp, h0_total_energy, total_energy)

            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
            ri, ri_ref = compute_richardson_number(
                t0, q0_vapor, qcon, pkz, delp, peln, gz, u0, v0, xvir, t_max, t_min
            )
            ri_ref = ri_ref * 2.0
            mc = compute_mass_flux(ri, ri_ref, delp, ratio)
            if ri < ri_ref:
                q0_vapor, h0_vapor = kh_adjust_down(mc, delp, q0_vapor, h0_vapor)
                q0_liquid, h0_liquid = kh_adjust_down(mc, delp, q0_liquid, h0_liquid)
                q0_rain, h0_rain = kh_adjust_down(mc, delp, q0_rain, h0_rain)
                q0_ice, h0_ice = kh_adjust_down(mc, delp, q0_ice, h0_ice)
                q0_snow, h0_snow = kh_adjust_down(mc, delp, q0_snow, h0_snow)
                q0_graupel, h0_graupel = kh_adjust_down(
                    mc, delp, q0_graupel, h0_graupel
                )
                q0_o3mr, h0_o3mr = kh_adjust_down(mc, delp, q0_o3mr, h0_o3mr)
                q0_sgs_tke, h0_sgs_tke = kh_adjust_down(
                    mc, delp, q0_sgs_tke, h0_sgs_tke
                )
                q0_cld, h0_cld = kh_adjust_down(mc, delp, q0_cld, h0_cld)
                u0, h0_u = kh_adjust_down(mc, delp, u0, h0_u)
                v0, h0_v = kh_adjust_down(mc, delp, v0, h0_v)
                w0, h0_w = kh_adjust_down(mc, delp, w0, h0_w)
                total_energy, h0_total_energy = kh_adjust_energy_down(
                    mc, delp, static_energy, total_energy, h0_total_energy
                )
            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
        with interval(1, 2):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = kh_adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = kh_adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = kh_adjust_up(delp, h0_rain, q0_rain)
                q0_ice = kh_adjust_up(delp, h0_ice, q0_ice)
                q0_snow = kh_adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = kh_adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = kh_adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = kh_adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = kh_adjust_up(delp, h0_cld, q0_cld)
                qcon = qcon_func(q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = kh_adjust_up(delp, h0_u, u0)
                v0 = kh_adjust_up(delp, h0_v, v0)
                w0 = kh_adjust_up(delp, h0_w, w0)
                total_energy = kh_adjust_up(delp, h0_total_energy, total_energy)

            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
            ri, ri_ref = compute_richardson_number(
                t0, q0_vapor, qcon, pkz, delp, peln, gz, u0, v0, xvir, t_max, t_min
            )
            ri_ref = ri_ref * 4.0
            mc = compute_mass_flux(ri, ri_ref, delp, ratio)
            if ri < ri_ref:
                q0_vapor, h0_vapor = kh_adjust_down(mc, delp, q0_vapor, h0_vapor)
                q0_liquid, h0_liquid = kh_adjust_down(mc, delp, q0_liquid, h0_liquid)
                q0_rain, h0_rain = kh_adjust_down(mc, delp, q0_rain, h0_rain)
                q0_ice, h0_ice = kh_adjust_down(mc, delp, q0_ice, h0_ice)
                q0_snow, h0_snow = kh_adjust_down(mc, delp, q0_snow, h0_snow)
                q0_graupel, h0_graupel = kh_adjust_down(
                    mc, delp, q0_graupel, h0_graupel
                )
                q0_o3mr, h0_o3mr = kh_adjust_down(mc, delp, q0_o3mr, h0_o3mr)
                q0_sgs_tke, h0_sgs_tke = kh_adjust_down(
                    mc, delp, q0_sgs_tke, h0_sgs_tke
                )
                q0_cld, h0_cld = kh_adjust_down(mc, delp, q0_cld, h0_cld)
                u0, h0_u = kh_adjust_down(mc, delp, u0, h0_u)
                v0, h0_v = kh_adjust_down(mc, delp, v0, h0_v)
                w0, h0_w = kh_adjust_down(mc, delp, w0, h0_w)
                total_energy, h0_total_energy = kh_adjust_energy_down(
                    mc, delp, static_energy, total_energy, h0_total_energy
                )
            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )
        with interval(0, 1):
            if ri[0, 0, 1] < ri_ref[0, 0, 1]:
                q0_vapor = kh_adjust_up(delp, h0_vapor, q0_vapor)
                q0_liquid = kh_adjust_up(delp, h0_liquid, q0_liquid)
                q0_rain = kh_adjust_up(delp, h0_rain, q0_rain)
                q0_ice = kh_adjust_up(delp, h0_ice, q0_ice)
                q0_snow = kh_adjust_up(delp, h0_snow, q0_snow)
                q0_graupel = kh_adjust_up(delp, h0_graupel, q0_graupel)
                q0_o3mr = kh_adjust_up(delp, h0_o3mr, q0_o3mr)
                q0_sgs_tke = kh_adjust_up(delp, h0_sgs_tke, q0_sgs_tke)
                q0_cld = kh_adjust_up(delp, h0_cld, q0_cld)
                qcon = qcon_func(q0_liquid, q0_ice, q0_snow, q0_rain, q0_graupel)
                u0 = kh_adjust_up(delp, h0_u, u0)
                v0 = kh_adjust_up(delp, h0_v, v0)
                w0 = kh_adjust_up(delp, h0_w, w0)
                total_energy = kh_adjust_up(delp, h0_total_energy, total_energy)

            cpm, cvm, t0, static_energy = adjust_cvm(
                cpm,
                cvm,
                q0_vapor,
                q0_liquid,
                q0_rain,
                q0_ice,
                q0_snow,
                q0_graupel,
                gz,
                u0,
                v0,
                w0,
                t0,
                total_energy,
                static_energy,
            )


@gtscript.function
def readjust_by_frac(a0, a, fra):
    return a + (a0 - a) * fra


def finalize(
    u0: FloatField,
    v0: FloatField,
    w0: FloatField,
    t0: FloatField,
    ua: FloatField,
    va: FloatField,
    ta: FloatField,
    w: FloatField,
    u_dt: FloatField,
    v_dt: FloatField,
    q0_vapor: FloatField,
    q0_liquid: FloatField,
    q0_rain: FloatField,
    q0_ice: FloatField,
    q0_snow: FloatField,
    q0_graupel: FloatField,
    q0_o3mr: FloatField,
    q0_sgs_tke: FloatField,
    q0_cld: FloatField,
    qvapor: FloatField,
    qliquid: FloatField,
    qrain: FloatField,
    qice: FloatField,
    qsnow: FloatField,
    qgraupel: FloatField,
    qo3mr: FloatField,
    qsgs_tke: FloatField,
    qcld: FloatField,
    timestep: float,
):
    from __externals__ import fv_sg_adj, hydrostatic

    with computation(PARALLEL), interval(...):
        fra = timestep / fv_sg_adj
        if fra < 1.0:
            t0 = readjust_by_frac(t0, ta, fra)
            u0 = readjust_by_frac(u0, ua, fra)
            v0 = readjust_by_frac(v0, va, fra)
            if __INLINED(not hydrostatic):
                w0 = readjust_by_frac(w0, w, fra)
            q0_vapor = readjust_by_frac(q0_vapor, qvapor, fra)
            q0_liquid = readjust_by_frac(q0_liquid, qliquid, fra)
            q0_rain = readjust_by_frac(q0_rain, qrain, fra)
            q0_ice = readjust_by_frac(q0_ice, qice, fra)
            q0_snow = readjust_by_frac(q0_snow, qsnow, fra)
            q0_graupel = readjust_by_frac(q0_graupel, qgraupel, fra)
            q0_o3mr = readjust_by_frac(q0_o3mr, qo3mr, fra)
            q0_sgs_tke = readjust_by_frac(q0_sgs_tke, qsgs_tke, fra)
            q0_cld = readjust_by_frac(q0_cld, qcld, fra)
        rdt = 1.0 / timestep
        u_dt = rdt * (u0 - ua)
        v_dt = rdt * (v0 - va)
        ta = t0
        ua = u0
        va = v0
        w = w0
        qvapor = q0_vapor
        qliquid = q0_liquid
        qrain = q0_rain
        qice = q0_ice
        qsnow = q0_snow
        qgraupel = q0_graupel
        qo3mr = q0_o3mr
        qsgs_tke = q0_sgs_tke
        qcld = q0_cld


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)


class DryConvectiveAdjustment:
    """
    Corresponds to fv_subgrid_z in Fortran's fv_sg module
    """

    arg_specs = (
        ArgSpec("delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="in"),
        ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="in"),
        ArgSpec("pe", "interface_pressure", "Pa", intent="in"),
        ArgSpec(
            "pkz",
            "layer_mean_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="in",
        ),
        ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="in"),
        ArgSpec("pt", "air_temperature", "degK", intent="inout"),
        ArgSpec("ua", "eastward_wind", "m/s", intent="inout"),
        ArgSpec("va", "northward_wind", "m/s", intent="inout"),
        ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
        ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
        ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qice", "cloud_ice_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
        ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
        ArgSpec(
            "u_dt", "eastward_wind_tendency_due_to_physics", "m/s**2", intent="inout"
        ),
        ArgSpec(
            "v_dt", "northward_wind_tendency_due_to_physics", "m/s**2", intent="inout"
        ),
    )

    def __init__(
        self,
        stencil_factory: StencilFactory,
        nwat: int,
        fv_sg_adj: float,
        n_sponge: int,
        hydrostatic: bool,
    ):
        assert not hydrostatic, "Hydrostatic not implemented for fv_subgridz"
        grid_indexing = stencil_factory.grid_indexing
        self._k_sponge = n_sponge
        if self._k_sponge is not None:
            if self._k_sponge < 3:
                return
        else:
            self._k_sponge = grid_indexing.domain[2]
        if self._k_sponge < min(grid_indexing.domain[2], 24):
            t_max = T2_MAX
        else:
            t_max = T3_MAX
        if nwat == 0:
            xvir = 0.0
        else:
            xvir = ZVIR
        self._m = 3
        self._fv_sg_adj = float(fv_sg_adj)
        self._is = grid_indexing.isc
        self._js = grid_indexing.jsc
        kbot_domain = (grid_indexing.domain[0], grid_indexing.domain[1], self._k_sponge)
        origin = grid_indexing.origin_compute()

        self._init_stencil = stencil_factory.from_origin_domain(
            init,
            origin=origin,
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                self._k_sponge + 1,
            ),
        )
        self._m_loop_stencil = stencil_factory.from_origin_domain(
            m_loop,
            externals={"t_max": t_max, "xvir": xvir},
            origin=origin,
            domain=kbot_domain,
        )
        self._finalize_stencil = stencil_factory.from_origin_domain(
            finalize,
            externals={
                "hydrostatic": hydrostatic,
                "fv_sg_adj": fv_sg_adj,
            },
            origin=origin,
            domain=kbot_domain,
        )

        def make_storage():
            return utils.make_storage_from_shape(
                grid_indexing.domain_full(add=(1, 1, 0)),
                backend=stencil_factory.backend,
            )

        self._q0 = {}
        for tracername in utils.tracer_variables:
            self._q0[tracername] = make_storage()
        self._tmp_u0 = make_storage()
        self._tmp_v0 = make_storage()
        self._tmp_w0 = make_storage()
        self._tmp_gz = make_storage()
        self._tmp_t0 = make_storage()
        self._tmp_static_energy = make_storage()
        self._tmp_total_energy = make_storage()
        self._tmp_cvm = make_storage()
        self._tmp_cpm = make_storage()
        self._ratios = {0: 0.25, 1: 0.5, 2: 0.999}

    def __call__(
        self, state: DycoreState, u_dt: Quantity, v_dt: Quantity, timestep: float
    ):
        """
        Performs dry convective adjustment mixing on the subgrid vertical scale.
        Args:
            state: see arg_specs, includes mainly windspeed, temperature,
                   pressure and tracer variables that are in the DycoreState
            u_dt: x-wind tendency for the dry convective windspeed adjustment
            v_dt: y-wind tendency for the dry convective windspeed adjustment
            timestep:  time to progress forward in seconds
        """
        if state.pe.data[self._is, self._js, 0] < 2.0:
            t_min = T1_MIN
        else:
            t_min = T2_MIN

        self._init_stencil(
            self._tmp_gz,
            self._tmp_t0,
            self._tmp_u0,
            self._tmp_v0,
            self._tmp_w0,
            self._tmp_static_energy,
            self._tmp_cvm,
            self._tmp_cpm,
            self._tmp_total_energy,
            state.ua,
            state.va,
            state.w,
            state.pt,
            state.delz,
            self._q0["qvapor"],
            self._q0["qliquid"],
            self._q0["qrain"],
            self._q0["qice"],
            self._q0["qsnow"],
            self._q0["qgraupel"],
            self._q0["qo3mr"],
            self._q0["qsgs_tke"],
            self._q0["qcld"],
            state.qvapor,
            state.qliquid,
            state.qrain,
            state.qice,
            state.qsnow,
            state.qgraupel,
            state.qo3mr,
            state.qsgs_tke,
            state.qcld,
        )

        for n in range(self._m):
            self._m_loop_stencil(
                self._tmp_u0,
                self._tmp_v0,
                self._tmp_w0,
                self._tmp_t0,
                self._tmp_static_energy,
                self._tmp_gz,
                state.delp,
                state.peln,
                state.pkz,
                self._q0["qvapor"],
                self._q0["qliquid"],
                self._q0["qrain"],
                self._q0["qice"],
                self._q0["qsnow"],
                self._q0["qgraupel"],
                self._q0["qo3mr"],
                self._q0["qsgs_tke"],
                self._q0["qcld"],
                self._tmp_total_energy,
                self._tmp_cpm,
                self._tmp_cvm,
                t_min,
                self._ratios[n],
            )

        self._finalize_stencil(
            self._tmp_u0,
            self._tmp_v0,
            self._tmp_w0,
            self._tmp_t0,
            state.ua,
            state.va,
            state.pt,
            state.w,
            u_dt,
            v_dt,
            self._q0["qvapor"],
            self._q0["qliquid"],
            self._q0["qrain"],
            self._q0["qice"],
            self._q0["qsnow"],
            self._q0["qgraupel"],
            self._q0["qo3mr"],
            self._q0["qsgs_tke"],
            self._q0["qcld"],
            state.qvapor,
            state.qliquid,
            state.qrain,
            state.qice,
            state.qsnow,
            state.qgraupel,
            state.qo3mr,
            state.qsgs_tke,
            state.qcld,
            timestep,
        )
