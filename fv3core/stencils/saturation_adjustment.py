#!/usr/bin/env python3
import math

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import dim


# TODO, this code could be reduced greatly with abstraction, but first gt4py needs to support gtscript function calls of arbitrary depth embedded in conditionals
sd = utils.sd
si = utils.si
DC_VAP = constants.CP_VAP - constants.C_LIQ  # - 2339.5, isobaric heating / cooling
DC_ICE = constants.C_LIQ - constants.C_ICE  # 2213.5, isobaric heating / cooling
LV0 = (
    constants.HLV - DC_VAP * constants.TICE
)  # 3.13905782e6, evaporation latent heat coefficient at 0 deg k
LI00 = (
    constants.HLF - DC_ICE * constants.TICE
)  # -2.7105966e5, fusion latent heat coefficient at 0 deg k
LI2 = LV0 + LI00  # 2.86799816e6, sublimation latent heat coefficient at 0 deg k
D2ICE = DC_VAP + DC_ICE  # - 126, isobaric heating / cooling
E00 = 611.21  # saturation vapor pressure at 0 deg c
TMIN = constants.TICE - 160.0
DELT = 0.1
satmix = {"table": None, "table2": None, "tablew": None, "des2": None, "desw": None}
TICE = constants.TICE
T_WFR = TICE - 40.0  # homogeneous freezing temperature
TICE0 = TICE - 0.01

LAT2 = (constants.HLV + constants.HLF) ** 2  # used in bigg mechanism
# melting of cloud ice to cloud water and rain
# TODO, when if blocks are possible,, only compute when 'melting'
QS_LENGTH = 2621


@gtscript.function
def tem_lower(i):
    return TMIN + DELT * i


@gtscript.function
def tem_upper(i):
    return 253.16 + DELT * i


"""
# TODO this abstraction is not possible in gt4py as these are called in conditionals and temporaries get created. but would be nice to have 1 version of the equation
@gtscript.function
def q_table_oneline(delta_heat_capacity, latent_heat_coefficient, tem):
    return E00 * exp(
        (delta_heat_capacity * log(tem / TICE) + (tem - TICE) / (tem * TICE) *  latent_heat_coefficient) / constants.RVGAS
    )

@gtscript.function
def table_vapor_oneline(tem):
    return q_table_oneline(DC_VAP, LV0, tem)


@gtscript.function
def table_ice_oneline(tem):
    return q_table_oneline(D2ICE, LI2, tem)
"""


@gtscript.function
def table_vapor_oneline(tem):
    return E00 * exp(
        (DC_VAP * log(tem / TICE) + (tem - TICE) / (tem * TICE) * LV0) / constants.RVGAS
    )


@gtscript.function
def table_ice_oneline(tem):
    return E00 * exp(
        (D2ICE * log(tem / TICE) + (tem - TICE) / (tem * TICE) * LI2) / constants.RVGAS
    )


# TODO math can be consolidated if we can call gtscript functions from conditionals, fac0 and fac2 functions and others
@gtscript.function
def qs_table_fn(i):
    table = 0.0
    tem_l = tem_lower(i)
    tem_u = tem_upper(i - 1400)
    if i < 1600:
        table = table_ice_oneline(tem_l)
    if i >= 1600:
        table = 0.0
    if i >= 1600 and i < (1400 + 1221):
        table = table_vapor_oneline(tem_u)
    wice = 0.0
    wh2o = 0.0
    esupc = 0.0
    if i >= 1400 and i < 1600:
        esupc = table_vapor_oneline(tem_u)
        wice = 0.05 * (TICE - tem_u)
        wh2o = 0.05 * (tem_u - 253.16)
        table = wice * table + wh2o * esupc
    return table


# TODO math can be consolidated if we can call gtscript functions from conditionals, fac0 and fac2 functions and others
@gtscript.function
def qs_table2_fn(i):
    tem0 = tem_lower(i)
    table2 = 0.0
    if i < 1600:
        # compute es over ice between - 160 deg c and 0 deg c.
        table2 = table_ice_oneline(tem0)
    else:
        # compute es over water between 0 deg c and 102 deg c.
        table2 = table_vapor_oneline(tem0)
    table2_m1 = 0.0
    table2_p1 = 0.0
    # TODO is there way to express the code below with something closer to this?:
    # if i == 1599:
    #    table2 = 0.25 * (table2(i-1) + 2.0 * qs_table_fn(i) + qs_table2_fn(i+1))
    # if i == 1600:
    #    table2 = 0.25 * (table2(i-1) + 2.0 * qs_table_fn(i) + qs_table2_fn(i+1))
    table = 0.0
    if i == 1599:
        # table(i)
        table = table_ice_oneline(tem0)
        tem0 = 253.16 + DELT * (i - 1400)  # tem_upper(i - 1400)
        # table_vapor_oneline(tem0)
        table = (0.05 * (TICE - tem0)) * table + (0.05 * (tem0 - 253.16)) * (
            E00
            * exp(
                (DC_VAP * log(tem0 / TICE) + (tem0 - TICE) / (tem0 * TICE) * LV0)
                / constants.RVGAS
            )
        )
        # table2(i - 1)
        tem0 = TMIN + DELT * 1598  # tem_lower(1598)
        table2_m1 = table_ice_oneline(tem0)
        # table2(i + 1)
        tem0 = TMIN + DELT * 1600  # tem_lower(1600)
        table2_p1 = table_vapor_oneline(tem0)
        table2 = 0.25 * (table2_m1 + 2.0 * table + table2_p1)
    if i == 1600:
        # table(i)
        tem0 = 253.16 + DELT * (i - 1400)  # tem_upper(i - 1400)
        table = table_vapor_oneline(tem0)
        # table2(i - 1)
        tem0 = TMIN + DELT * 1599  # tem_lower(1599)
        table2_m1 = table_ice_oneline(tem0)
        # table2(i + 1)
        tem0 = TMIN + DELT * 1601  # tem_lower(1601)
        table2_p1 = table_vapor_oneline(tem0)
        table2 = 0.25 * (table2_m1 + 2.0 * table + table2_p1)
    return table2


@gtscript.function
def qs_tablew_fn(i):
    tem = tem_lower(i)
    return table_vapor_oneline(tem)


@gtscript.function
def des_end(t, i, z, des2):
    t_m1 = 0.0
    tem0 = 0.0
    diff = 0.0
    if i == QS_LENGTH - 1:
        # TODO if able to call function inside of conditional
        # t_m1 = qs_table2_fn(i - 1)
        tem0 = TMIN + DELT * (i - 1)  # tem_lower(i - 1)
        t_m1 = table_vapor_oneline(tem0)
        diff = t - t_m1
        des2 = max(z, diff)
    return des2


# TODO there might be a cleaner way to set des2[QS_LENGTH - 1] to des2[QS_LENGTH - 2]
@gtscript.function
def des2_table(i):
    t_p1 = qs_table2_fn(i + 1)
    t = qs_table2_fn(i)
    diff = t_p1 - t
    z = 0.0
    des2 = max(z, diff)
    des2 = des_end(t, i, z, des2)
    return des2


# TODO there might be a cleaner way to set desw[QS_LENGTH - 1] to desw[QS_LENGTH - 2]
@gtscript.function
def desw_table(i):
    t_p1 = qs_tablew_fn(i + 1)
    t = qs_tablew_fn(i)
    diff = t_p1 - t
    z = 0.0
    desw = max(z, diff)
    desw = des_end(t, i, z, desw)
    return desw


@gtscript.function
def compute_cvm(mc_air, qv, c_vap, q_liq, q_sol):
    return mc_air + qv * c_vap + q_liq * constants.C_LIQ + q_sol * constants.C_ICE


@gtscript.function
def add_src_pt1(pt1, src, lhl, cvm):
    return pt1 + src * lhl / cvm


@gtscript.function
def subtract_sink_pt1(pt1, sink, lhl, cvm):
    return pt1 - sink * lhl / cvm


@gtscript.function
def melt_cloud_ice(
    qv, qi, ql, q_liq, q_sol, pt1, icp2, fac_imlt, mc_air, c_vap, lhi, cvm
):
    # TODO, if temporaries inside of if-statements become supported, remove factmp and sink
    factmp = 0.0
    sink = 0.0
    if (qi > 1.0e-8) and (pt1 > TICE):
        factmp = fac_imlt * (pt1 - TICE) / icp2
        sink = qi if qi < factmp else factmp
        qi = qi - sink
        ql = ql + sink
        q_liq = q_liq + sink
        q_sol = q_sol - sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        sink = -sink
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return qi, ql, q_liq, q_sol, cvm, pt1


@gtscript.function
def minmax_tmp_h20(qa, qb):
    tmpmax = qb if qb > 0.0 else 0.0
    tmp = -qa if -qa < tmpmax else tmpmax
    return tmp


# fix negative snow with graupel or graupel with available snow
# TODO fix so only compute tmp when it is needed
@gtscript.function
def fix_negative_snow(qs, qg):
    tmp = minmax_tmp_h20(qg, qs)
    if qs < 0.0:
        qg = qg + qs
        qs = 0.0
    elif qg < 0.0:
        qg = qg + tmp
        qs = qs - tmp
    return qs, qg


# fix negative cloud water with rain or rain with available cloud water
# TODO fix so only compute tmp when it is needed
@gtscript.function
def fix_negative_cloud_water(ql, qr):
    tmpl = minmax_tmp_h20(ql, qr)
    tmpr = minmax_tmp_h20(qr, ql)
    if ql < 0.0:
        ql = ql + tmpl
        qr = qr - tmpl
    elif qr < 0.0:
        ql = ql - tmpr
        qr = qr + tmpr
    return ql, qr


# enforce complete freezing of cloud water to cloud ice below - 48 c
@gtscript.function
def complete_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap):
    dtmp = TICE - 48.0 - pt1
    sink = 0.0
    if ql > 0.0 and dtmp > 0.0:
        sink = ql if ql < dtmp / icp2 else dtmp / icp2
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return ql, qi, q_liq, q_sol, cvm, pt1


@gtscript.function
def homogenous_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap):
    dtmp = T_WFR - pt1  # [ - 40, - 48]
    sink = 0.0
    if ql > 0.0 and dtmp > 0.0:
        sink = ql if ql < dtmp / icp2 else dtmp / icp2
        sink = (
            sink if sink < ql * dtmp * 0.125 else ql * dtmp * 0.125
        )  # min (ql, ql * dtmp * 0.125, dtmp / icp2)
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return ql, qi, q_liq, q_sol, cvm, pt1


# bigg mechanism for heterogeneous freezing
@gtscript.function
def heterogeneous_freezing(
    exptc, pt1, cvm, ql, qi, q_liq, q_sol, den, icp2, dt_bigg, mc_air, lhi, qv, c_vap
):
    tc = TICE0 - pt1
    sink = 0.0
    if ql > 0.0 and tc > 0.0:
        sink = 3.3333e-10 * dt_bigg * (exptc - 1.0) * den * ql ** 2
        sink = ql if ql < sink else sink
        sink = sink if sink < tc / icp2 else tc / icp2
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return ql, qi, q_liq, q_sol, cvm, pt1


@gtscript.function
def make_graupel(pt1, cvm, fac_r2g, qr, qg, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap):
    dtmp = (TICE - 0.1) - pt1
    tmp = 0.0
    sinktmp = 0.0
    rainfac = 0.0
    sink = 0.0
    if qr > 1e-7 and dtmp > 0.0:
        rainfac = (dtmp * 0.025) ** 2
        #  no limit on freezing below - 40 deg c
        tmp = qr if 1.0 < rainfac else rainfac * qr
        sinktmp = fac_r2g * dtmp / icp2
        sink = tmp if tmp < sinktmp else sinktmp
        qr = qr - sink
        qg = qg + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return qr, qg, q_liq, q_sol, cvm, pt1


# qr, qg, q_liq, q_sol, cvm, pt1 = make_graupel(
#            pt1, cvm, fac_r2g, qr, qg, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap


@gtscript.function
def melt_snow(
    pt1, cvm, fac_smlt, qs, ql, qr, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap, qs_mlt
):
    dtmp = pt1 - (TICE + 0.1)
    tmp = 0.0
    sink = 0.0
    snowfac = 0.0
    sinktmp = 0.0
    dimqs = dim(qs_mlt, ql)
    if qs > 1e-7 and dtmp > 0.0:
        snowfac = (dtmp * 0.1) ** 2
        tmp = (
            qs if 1.0 < snowfac else snowfac * qs
        )  # no limiter on melting above 10 deg c
        sinktmp = fac_smlt * dtmp / icp2
        sink = tmp if tmp < sinktmp else sinktmp
        tmp = sink if sink < dimqs else dimqs
        qs = qs - sink
        ql = ql + tmp
        qr = qr + sink - tmp
        q_liq = q_liq + sink
        q_sol = q_sol - sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = subtract_sink_pt1(pt1, sink, lhi, cvm)
    return qs, ql, qr, q_liq, q_sol, cvm, pt1


@gtscript.function
def autoconversion_cloud_to_rain(ql, qr, fac_l2r, ql0_max):
    sink = 0.0
    if ql > ql0_max:
        sink = fac_l2r * (ql - ql0_max)
        qr = qr + sink
        ql = ql - sink
    return ql, qr


@gtscript.function
def sublimation(
    pt1,
    cvm,
    expsubl,
    qv,
    qi,
    q_liq,
    q_sol,
    iqs2,
    tcp2,
    den,
    dqsdt,
    sdt,
    adj_fac,
    mc_air,
    c_vap,
    lhl,
    lhi,
    t_sub,
    qi_gen,
    qi_lim,
):
    src = 0.0
    dq = 0.0
    sink = 0.0
    pidep = 0.0
    tmp = 0.0
    maxtmp = 0.0
    dimtmp = 0.0
    qi_crt = 0.0
    if pt1 < t_sub:
        src = qv - 1e-6 if (qv - 1e-6) > 0.0 else 0.0  # dim(qv, 1e-6) TODO THIS BREAKS
    elif pt1 < TICE0:
        # qsi = iqs2
        dq = qv - iqs2
        sink = adj_fac * dq / (1.0 + tcp2 * dqsdt)
        if qi > 1.0e-8:
            pidep = (
                sdt
                * dq
                * 349138.78
                * expsubl
                / (
                    iqs2 * den * LAT2 / (0.0243 * constants.RVGAS * pt1 ** 2.0)
                    + 4.42478e4
                )
            )
        else:
            pidep = 0.0
        if dq > 0.0:
            tmp = TICE - pt1
            qi_crt = (
                qi_gen * qi_lim / den
                if qi_lim < 0.1 * tmp
                else qi_gen * 0.1 * tmp / den
            )
            maxtmp = qi_crt - qi if qi_crt - qi > pidep else pidep
            src = sink if sink < maxtmp else maxtmp
            src = src if src < tmp / tcp2 else tmp / tcp2
        else:
            dimtmp = (
                pt1 - t_sub if (pt1 - t_sub) > 0.0 else 0.0
            )  # dim(pt1, t_sub) * 0.2 TODO WHY DOES THIS NOT WORK
            pidep = pidep if 1.0 < (dimtmp * 0.2) else pidep * dimtmp * 0.2
            src = pidep if pidep > sink else sink
            src = src if src > -qi else -qi
    qv = qv - src
    qi = qi + src
    q_sol = q_sol + src
    cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
    lh = lhl + lhi
    pt1 = add_src_pt1(pt1, src, lh, cvm)
    return qv, qi, q_sol, cvm, pt1


@gtscript.function
def update_latent_heat_coefficient_i(pt1, cvm):
    lhi = LI00 + DC_ICE * pt1
    icp2 = lhi / cvm
    return lhi, icp2


@gtscript.function
def update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap):
    lhl = lv00 + d0_vap * pt1
    lhi = LI00 + DC_ICE * pt1
    lcp2 = lhl / cvm
    icp2 = lhi / cvm
    # lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
    return lhl, lhi, lcp2, icp2


@gtscript.function
def compute_dq0(qv, wqsat, dq2dt, tcp3):
    return (qv - wqsat) / (1.0 + tcp3 * dq2dt)


@gtscript.function
def get_factor(wqsat, qv, fac_l2v):
    factor = fac_l2v * 10.0 * (1.0 - qv / wqsat)
    factor = -min(1, factor)
    return factor


@gtscript.function
def get_src(ql, factor, dq0):
    src = -min(ql, factor * dq0)
    return src


@gtscript.function
def ql_evaporation(wqsat, qv, ql, dq0, fac_l2v):
    factor = get_factor(wqsat, qv, fac_l2v)
    src = get_src(ql, factor, dq0)
    return factor, src


@gtscript.function
def wqsat_correct(src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap):
    qv = qv - src
    ql = ql + src
    q_liq = q_liq + src
    cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
    pt1 = add_src_pt1(pt1, src, lhl, cvm)  # pt1 + src * lhl / cvm
    return qv, ql, q_liq, cvm, pt1


@gtstencil()
def ap1_stencil(ta: sd, ap1: sd):
    with computation(PARALLEL), interval(...):
        ap1 = ap1_for_wqs2(ta)


@gtscript.function
def ap1_for_wqs2(ta):
    ap1 = 10.0 * dim(ta, TMIN) + 1.0
    return min(ap1, QS_LENGTH) - 1


@gtscript.function
def ap1_index(ap1):
    return floor(ap1)


@gtscript.function
def ap1_indices(ap1):
    it = ap1_index(ap1)
    it2 = floor(ap1 - 0.5)
    it2_p1 = it2 + 1
    return it, it2, it2_p1


@gtscript.function
def ap1_and_indices(ta):
    ap1 = ap1_for_wqs2(ta)
    it, it2, it2_p1 = ap1_indices(ap1)
    return ap1, it, it2, it2_p1


@gtscript.function
def ap1_and_index(ta):
    ap1 = ap1_for_wqs2(ta)
    it = ap1_index(ap1)
    return it, ap1


@gtscript.function
def wqsat_and_dqdt(tablew, desw, desw2, desw_p1, ap1, it, it2, ta, den):
    es = tablew + (ap1 - it) * desw
    denom = constants.RVGAS * ta * den
    wqsat = es / denom
    dqdt = 10.0 * (desw2 + (ap1 - it2) * (desw_p1 - desw2))
    dqdt = dqdt / denom
    return wqsat, dqdt


@gtscript.function
def wqsat_wsq1(table, des, ap1, it, ta, den):
    es = table + (ap1 - it) * des
    return es / (constants.RVGAS * ta * den)


@gtscript.function
def wqs2_fn_2(ta, den):
    ap1, it, it2, it2_p1 = ap1_and_indices(ta)
    table2 = qs_table2_fn(it)
    des2 = des2_table(it)
    des22 = des2_table(it2)
    des2_p1 = des2_table(it2_p1)
    wqsat, dqdt = wqsat_and_dqdt(table2, des2, des22, des2_p1, ap1, it, it2, ta, den)
    return wqsat, dqdt


@gtscript.function
def wqs2_fn_w(ta, den):
    ap1, it, it2, it2_p1 = ap1_and_indices(ta)
    tablew = qs_tablew_fn(it)
    desw = desw_table(it)
    desw2 = desw_table(it2)
    desw_p1 = desw_table(it2_p1)
    wqsat, dqdt = wqsat_and_dqdt(tablew, desw, desw2, desw_p1, ap1, it, it2, ta, den)
    return wqsat, dqdt


@gtscript.function
def wqs1_fn_w(it, ap1, ta, den):
    tablew = qs_tablew_fn(it)
    desw = desw_table(it)
    return wqsat_wsq1(tablew, desw, ap1, it, ta, den)


@gtscript.function
def wqs1_fn_2(it, ap1, ta, den):
    table2 = qs_table2_fn(it)
    des2 = des2_table(it)
    return wqsat_wsq1(table2, des2, ap1, it, ta, den)


@gtstencil()
def wqs2_stencil_w(ta: sd, den: sd, wqsat: sd, dqdt: sd):
    with computation(PARALLEL), interval(...):
        wqsat, dqdt = wqs2_fn_w(ta, den)


@gtstencil()
def compute_q_tables(index: sd, tablew: sd, table2: sd, table: sd, desw: sd, des2: sd):
    with computation(PARALLEL), interval(...):
        tablew = qs_tablew_fn(index)
        table2 = qs_table2_fn(index)
        table = qs_table_fn(index)
        desw = desw_table(index)
        des2 = des2_table(index)


@gtstencil()
def satadjust_part1(
    wqsat: sd,
    dq2dt: sd,
    dpln: sd,
    den: sd,
    pt1: sd,
    cvm: sd,
    mc_air: sd,
    peln: sd,
    qv: sd,
    ql: sd,
    q_liq: sd,
    qi: sd,
    qr: sd,
    qs: sd,
    q_sol: sd,
    qg: sd,
    pt: sd,
    dp: sd,
    delz: sd,
    te0: sd,
    qpz: sd,
    lhl: sd,
    lhi: sd,
    lcp2: sd,
    icp2: sd,
    tcp3: sd,
    zvir: float,
    hydrostatic: bool,
    consv_te: bool,
    c_air: float,
    c_vap: float,
    fac_imlt: float,
    d0_vap: float,
    lv00: float,
    fac_v2l: float,
    fac_l2v: float,
):
    with computation(FORWARD), interval(1, None):
        if hydrostatic:
            delz = delz[0, 0, -1]
    with computation(PARALLEL), interval(...):
        dpln = peln[0, 0, 1] - peln
        q_liq = ql + qr
        q_sol = qi + qs + qg
        qpz = q_liq + q_sol
        pt1 = pt / ((1.0 + zvir * qv) * (1.0 - qpz))
        t0 = pt1  # true temperature
        qpz = qpz + qv  # total_wat conserved in this routine
        # define air density based on hydrostatical property
        den = (
            dp / (dpln * constants.RDGAS * pt)
            if hydrostatic
            else -dp / (constants.GRAV * delz)
        )
        # define heat capacity and latend heat coefficient
        mc_air = (1.0 - qpz) * c_air
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        #  fix energy conservation
        if consv_te:
            if hydrostatic:
                te0 = -c_air * t0
            else:
                te0 = -cvm * t0
        # fix negative cloud ice with snow
        if qi < 0.0:
            qs = qs + qi
            qi = 0.0

        #  melting of cloud ice to cloud water and rain
        qi, ql, q_liq, q_sol, cvm, pt1 = melt_cloud_ice(
            qv, qi, ql, q_liq, q_sol, pt1, icp2, fac_imlt, mc_air, c_vap, lhi, cvm
        )
        # update latend heat coefficient
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        # fix negative snow with graupel or graupel with available snow
        qs, qg = fix_negative_snow(qs, qg)
        # after this point cloud ice & snow are positive definite
        # fix negative cloud water with rain or rain with available cloud water
        ql, qr = fix_negative_cloud_water(ql, qr)
        # enforce complete freezing of cloud water to cloud ice below - 48 c
        ql, qi, q_liq, q_sol, cvm, pt1 = complete_freezing(
            qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap
        )
        wqsat, dq2dt = wqs2_fn_w(pt1, den)
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = min(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin

        dq0 = (qv - wqsat) / (
            1.0 + tcp3 * dq2dt
        )  # compute_dq0(qv, wqsat, dq2dt, tcp3)  #(qv - wqsat) / (1.0 + tcp3 * dq2dt)
        # TODO might be able to get rid of these temporary allocations when not used?
        if dq0 > 0:  # whole grid - box saturated
            src = min(
                spec.namelist.sat_adj0 * dq0,
                max(spec.namelist.ql_gen - ql, fac_v2l * dq0),
            )
        else:
            # TODO -- we'd like to use this abstraction rather than duplicate code, but inside the if conditional complains 'not implemented'
            # factor, src = ql_evaporation(wqsat, qv, ql, dq0,fac_l2v)
            factor = -1.0 * min(1, fac_l2v * 10.0 * (1.0 - qv / wqsat))
            src = -1.0 * min(ql, factor * dq0)

        qv, ql, q_liq, cvm, pt1 = wqsat_correct(
            src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap
        )
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        # TODO remove duplicate
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = min(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin


# TODO reading in ql0_max as a runtime argument causes problems for the if statement
@gtstencil()
def satadjust_part2(
    wqsat: sd,
    dq2dt: sd,
    pt1: sd,
    pt: sd,
    cvm: sd,
    mc_air: sd,
    tcp3: sd,
    lhl: sd,
    lhi: sd,
    lcp2: sd,
    icp2: sd,
    qv: sd,
    ql: sd,
    q_liq: sd,
    qi: sd,
    q_sol: sd,
    den: sd,
    qr: sd,
    qg: sd,
    qs: sd,
    cappa: sd,
    dp: sd,
    tin: sd,
    te0: sd,
    q_cond: sd,
    q_con: sd,
    sdt: float,
    adj_fac: float,
    zvir: float,
    fac_i2s: float,
    c_air: float,
    consv_te: bool,
    hydrostatic: bool,
    do_qa: bool,
    fac_v2l: float,
    fac_l2v: float,
    lv00: float,
    d0_vap: float,
    c_vap: float,
    mdt: float,
    fac_r2g: float,
    fac_smlt: float,
    fac_l2r: float,
    last_step: bool,
    rad_snow: bool,
    rad_rain: bool,
    rad_graupel: bool,
    tintqs: bool,
):
    with computation(PARALLEL), interval(...):
        dq0 = 0.0
        if last_step:
            dq0 = compute_dq0(qv, wqsat, dq2dt, tcp3)
            if dq0 > 0:
                src = dq0
            else:
                # TODO -- we'd like to use this abstraction rather than duplicate code, but inside the if conditional complains 'not implemented'
                # factor, src = ql_evaporation(wqsat, qv, ql, dq0,fac_l2v)
                factor = -1.0 * min(1, fac_l2v * 10.0 * (1.0 - qv / wqsat))
                src = -1.0 * min(ql, factor * dq0)
            # TODO causes a visit_if error 'NoneType' object has no attribute 'inputs'
            # qv, ql, q_liq, cvm, pt1 = wqsat_correct(src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap)
            # lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
            qv = qv - src
            ql = ql + src
            q_liq = q_liq + src
            cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
            pt1 = add_src_pt1(pt1, src, lhl, cvm)  # pt1 + src * lhl / cvm
            # TODO, revisit when gt4py updated, causes an Assertion error
            # lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
            lhl = lv00 + d0_vap * pt1
            lhi = LI00 + DC_ICE * pt1
            lcp2 = lhl / cvm
            icp2 = lhi / cvm
        # homogeneous freezing of cloud water to cloud ice
        ql, qi, q_liq, q_sol, cvm, pt1 = homogenous_freezing(
            qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap
        )
        # update some of the latent heat coefficients
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        exptc = exp(0.66 * (TICE0 - pt1))
        # bigg mechanism (heterogeneous freezing of cloud water to cloud ice)
        ql, qi, q_liq, q_sol, cvm, pt1 = heterogeneous_freezing(
            exptc,
            pt1,
            cvm,
            ql,
            qi,
            q_liq,
            q_sol,
            den,
            icp2,
            mdt,
            mc_air,
            lhi,
            qv,
            c_vap,
        )
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        # freezing of rain to graupel
        qr, qg, q_liq, q_sol, cvm, pt1 = make_graupel(
            pt1, cvm, fac_r2g, qr, qg, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap
        )
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        # melting of snow to rain or cloud water
        qs, ql, qr, q_liq, q_sol, cvm, pt1 = melt_snow(
            pt1,
            cvm,
            fac_smlt,
            qs,
            ql,
            qr,
            q_liq,
            q_sol,
            lhi,
            icp2,
            mc_air,
            qv,
            c_vap,
            spec.namelist.qs_mlt,
        )
        #  autoconversion from cloud water to rain
        ql, qr = autoconversion_cloud_to_rain(ql, qr, fac_l2r, spec.namelist.ql0_max)
        iqs2, dqsdt = wqs2_fn_2(pt1, den)
        expsubl = exp(0.875 * log(qi * den))
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        tcp2 = lcp2 + icp2
        qv, qi, q_sol, cvm, pt1 = sublimation(
            pt1,
            cvm,
            expsubl,
            qv,
            qi,
            q_liq,
            q_sol,
            iqs2,
            tcp2,
            den,
            dqsdt,
            sdt,
            adj_fac,
            mc_air,
            c_vap,
            lhl,
            lhi,
            spec.namelist.t_sub,
            spec.namelist.qi_gen,
            spec.namelist.qi_lim,
        )
        # virtual temp updated
        q_con = q_liq + q_sol
        tmp = 1.0 + zvir * qv
        pt = pt1 * tmp * (1.0 - q_con)
        tmp = constants.RDGAS * tmp
        cappa = tmp / (tmp + cvm)
        #  fix negative graupel with available cloud ice
        maxtmp = 0.0
        if qg < 0:
            maxtmp = 0.0 if 0.0 > qi else qi
            tmp = -qg if -qg < maxtmp else maxtmp
            qg = qg + tmp
            qi = qi - tmp
        else:
            qg = qg
        #  autoconversion from cloud ice to snow
        qim = spec.namelist.qi0_max / den
        sink = 0.0
        if qi > qim:
            sink = fac_i2s * (qi - qim)
            qi = qi - sink
            qs = qs + sink
        # fix energy conservation
        if consv_te:
            if hydrostatic:
                te0 = dp * (te0 + c_air * pt1)
            else:
                te0 = dp * (te0 + cvm * pt1)
        # update latent heat coefficient
        cvm = mc_air + (qv + q_liq + q_sol) * c_vap
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        # compute cloud fraction
        tin = 0.0
        if do_qa and last_step:
            # combine water species
            if rad_snow:
                if rad_graupel:
                    q_sol = qi + qs + qg
                else:
                    q_sol = qi + qs
            else:
                q_sol = qi
            if rad_rain:
                q_liq = ql + qr
            else:
                q_liq = ql
            q_cond = q_sol + q_liq
            # use the "liquid - frozen water temperature" (tin) to compute saturated specific humidity
            if tintqs:
                tin = pt1
            else:
                tin = pt1 - (lcp2 * q_cond + icp2 * q_sol)


@gtstencil()
def satadjust_part3_laststep_qa(
    qa: sd,
    area: sd,
    qpz: sd,
    hs: sd,
    tin: sd,
    q_cond: sd,
    q_sol: sd,
    den: sd,
):
    with computation(PARALLEL), interval(...):
        it, ap1 = ap1_and_index(tin)
        wqs1 = wqs1_fn_w(it, ap1, tin, den)
        iqs1 = wqs1_fn_2(it, ap1, tin, den)
        # Determine saturated specific humidity
        if tin < T_WFR:
            # ice phase
            qstar = iqs1
        elif tin >= TICE:
            qstar = wqs1
        else:
            # qsw = wqs1
            if q_cond > 1e-6:
                rqi = q_sol / q_cond
            else:
                rqi = (TICE - tin) / (TICE - T_WFR)
            qstar = rqi * iqs1 + (1.0 - rqi) * wqs1
        #  higher than 10 m is considered "land" and will have higher subgrid variability
        mindw = min(1.0, abs(hs) / (10.0 * constants.GRAV))
        dw = (
            spec.namelist.dw_ocean
            + (spec.namelist.dw_land - spec.namelist.dw_ocean) * mindw
        )
        # "scale - aware" subgrid variability: 100 - km as the base
        dbl_sqrt_area = dw * (area ** 0.5 / 100.0e3) ** 0.5
        maxtmp = 0.01 if 0.01 > dbl_sqrt_area else dbl_sqrt_area
        hvar = min(0.2, maxtmp)
        # partial cloudiness by pdf:
        # assuming subgrid linear distribution in horizontal; this is effectively a smoother for the
        # binary cloud scheme; qa = 0.5 if qstar == qpz
        rh = qpz / qstar
        # icloud_f = 0: bug - fixed
        # icloud_f = 1: old fvgfs gfdl) mp implementation
        # icloud_f = 2: binary cloud scheme (0 / 1)
        if rh > 0.75 and qpz > 1.0e-8:
            dq = hvar * qpz
            q_plus = qpz + dq
            q_minus = qpz - dq
            if spec.namelist.icloud_f == 2:  # TODO untested
                if qpz > qstar:
                    qa = 1.0
                elif (qstar < q_plus) and (q_cond > 1.0e-8):
                    qa = min(1.0, ((q_plus - qstar) / dq) ** 2)
                else:
                    qa = 0.0
            else:
                if qstar < q_minus:
                    qa = 1.0
                else:
                    if qstar < q_plus:
                        if spec.namelist.icloud_f == 0:
                            qa = (q_plus - qstar) / (dq + dq)
                        else:
                            qa = (q_plus - qstar) / (2.0 * dq * (1.0 - q_cond))
                    else:
                        qa = 0.0
                    # impose minimum cloudiness if substantial q_cond exist
                    if q_cond > 1.0e-8:
                        qa = max(spec.namelist.cld_min, qa)
                    qa = min(1, qa)
        else:
            qa = 0.0


def compute(
    dpln,
    te,
    qvapor,
    qliquid,
    qice,
    qrain,
    qsnow,
    qgraupel,
    qcld,
    hs,
    peln,
    delp,
    delz,
    q_con,
    pt,
    pkz,
    cappa,
    r_vir,
    mdt,
    fast_mp_consv,
    last_step,
    akap,
    kmp,
):
    grid = spec.grid
    origin = (grid.is_, grid.js, kmp)
    domain = (grid.nic, grid.njc, (grid.npz - kmp))
    hydrostatic = spec.namelist.hydrostatic
    sdt = 0.5 * mdt  # half remapping time step
    # define conversion scalar / factor
    fac_i2s = 1.0 - math.exp(-mdt / spec.namelist.tau_i2s)
    fac_v2l = 1.0 - math.exp(-sdt / spec.namelist.tau_v2l)
    fac_r2g = 1.0 - math.exp(-mdt / spec.namelist.tau_r2g)
    fac_l2r = 1.0 - math.exp(-mdt / spec.namelist.tau_l2r)

    fac_l2v = 1.0 - math.exp(-sdt / spec.namelist.tau_l2v)
    fac_l2v = min(spec.namelist.sat_adj0, fac_l2v)

    fac_imlt = 1.0 - math.exp(-sdt / spec.namelist.tau_imlt)
    fac_smlt = 1.0 - math.exp(-mdt / spec.namelist.tau_smlt)

    # define heat capacity of dry air and water vapor based on hydrostatical property

    if hydrostatic:
        c_air = constants.CP_AIR
        c_vap = constants.CP_VAP
    else:
        c_air = constants.CV_AIR
        c_vap = constants.CV_VAP

    d0_vap = c_vap - constants.C_LIQ
    lv00 = constants.HLV - d0_vap * TICE
    # temporaries needed for passing data between stencil calls (break currently required by wqs2_vect, and a couple of exp/log calls)
    den = utils.make_storage_from_shape(peln.shape, utils.origin)
    wqsat = utils.make_storage_from_shape(peln.shape, utils.origin)
    dq2dt = utils.make_storage_from_shape(peln.shape, utils.origin)
    pt1 = utils.make_storage_from_shape(peln.shape, utils.origin)
    cvm = utils.make_storage_from_shape(peln.shape, utils.origin)
    q_liq = utils.make_storage_from_shape(peln.shape, utils.origin)
    mc_air = utils.make_storage_from_shape(peln.shape, utils.origin)
    q_sol = utils.make_storage_from_shape(peln.shape, utils.origin)
    tcp3 = utils.make_storage_from_shape(peln.shape, utils.origin)
    lhl = utils.make_storage_from_shape(peln.shape, utils.origin)
    lhi = utils.make_storage_from_shape(peln.shape, utils.origin)
    lcp2 = utils.make_storage_from_shape(peln.shape, utils.origin)
    icp2 = utils.make_storage_from_shape(peln.shape, utils.origin)
    tin = utils.make_storage_from_shape(peln.shape, utils.origin)
    q_cond = utils.make_storage_from_shape(peln.shape, utils.origin)
    qpz = utils.make_storage_from_shape(peln.shape, utils.origin)
    satadjust_part1(
        wqsat,
        dq2dt,
        dpln,
        den,
        pt1,
        cvm,
        mc_air,
        peln,
        qvapor,
        qliquid,
        q_liq,
        qice,
        qrain,
        qsnow,
        q_sol,
        qgraupel,
        pt,
        delp,
        delz,
        te,
        qpz,
        lhl,
        lhi,
        lcp2,
        icp2,
        tcp3,
        r_vir,
        hydrostatic,
        fast_mp_consv,
        c_air,
        c_vap,
        fac_imlt,
        d0_vap,
        lv00,
        fac_v2l,
        fac_l2v,
        origin=origin,
        domain=domain,
    )

    if last_step:
        adj_fac = 1.0
        # condensation / evaporation between water vapor and cloud water, last time step
        #  enforce upper (no super_sat) & lower (critical rh) bounds
        # final iteration:
        # TODO, if can call functions from conditionals, can call function inside the stencil
        wqs2_stencil_w(
            pt1,
            den,
            wqsat,
            dq2dt,
            origin=(0, 0, 0),
            domain=spec.grid.domain_shape_standard(),
        )
    else:
        adj_fac = spec.namelist.sat_adj0

    # TODO  -- this isn't a namelist option in Fortran, it is whether or not cld_amount is a tracer. If/when we support different sets of tracers, this will need to change
    do_qa = True
    satadjust_part2(
        wqsat,
        dq2dt,
        pt1,
        pt,
        cvm,
        mc_air,
        tcp3,
        lhl,
        lhi,
        lcp2,
        icp2,
        qvapor,
        qliquid,
        q_liq,
        qice,
        q_sol,
        den,
        qrain,
        qgraupel,
        qsnow,
        cappa,
        delp,
        tin,
        te,
        q_cond,
        q_con,
        sdt,
        adj_fac,
        r_vir,
        fac_i2s,
        c_air,
        fast_mp_consv,
        hydrostatic,
        do_qa,
        fac_v2l,
        fac_l2v,
        lv00,
        d0_vap,
        c_vap,
        mdt,
        fac_r2g,
        fac_smlt,
        fac_l2r,
        last_step,
        spec.namelist.rad_snow,
        spec.namelist.rad_rain,
        spec.namelist.rad_graupel,
        spec.namelist.tintqs,
        origin=origin,
        domain=domain,
    )

    if do_qa and last_step:
        satadjust_part3_laststep_qa(
            qcld,
            grid.area_64,
            qpz,
            hs,
            tin,
            q_cond,
            q_sol,
            den,
            origin=origin,
            domain=domain,
        )
    if not spec.namelist.hydrostatic:
        moist_cv.compute_pkz_stencil_func(
            pkz, cappa, delp, delz, pt, origin=origin, domain=domain
        )
