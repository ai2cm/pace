import math

import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    exp,
    floor,
    interval,
    log,
)

import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core._config import SatAdjustConfig
from pace.fv3core.stencils.basic_operations import dim
from pace.fv3core.stencils.moist_cv import compute_pkz_func


# TODO: This code could be reduced greatly with abstraction, but first gt4py
# needs to support gtscript function calls of arbitrary depth embedded in
# conditionals.


DELT = 0.1
satmix = {"table": None, "table2": None, "tablew": None, "des2": None, "desw": None}


# melting of cloud ice to cloud water and rain
# TODO, when if blocks are possible,, only compute when 'melting'
QS_LENGTH = 2621


@gtscript.function
def tem_lower(i):
    return constants.T_SAT_MIN + DELT * i


@gtscript.function
def tem_upper(i):
    return 253.16 + DELT * i


@gtscript.function
def q_table_oneline(delta_heat_capacity, latent_heat_coefficient, tem):
    return constants.E00 * exp(
        (
            delta_heat_capacity * log(tem / constants.TICE)
            + (tem - constants.TICE) / (tem * constants.TICE) * latent_heat_coefficient
        )
        / constants.RVGAS
    )


@gtscript.function
def table_vapor_oneline(tem):
    return q_table_oneline(constants.DC_VAP, constants.LV0, tem)


@gtscript.function
def table_ice_oneline(tem):
    return q_table_oneline(constants.D2ICE, constants.LI2, tem)


# TODO Math can be consolidated if we can call gtscript functions from
# conditionals, fac0 and fac2 functions and others.
@gtscript.function
def qs_table_fn(i):
    tem_l = tem_lower(i)
    tem_u = tem_upper(i - 1400)
    table = 0.0
    if i < 1600:
        table = table_ice_oneline(tem_l)
    if i >= 1600 and i < (1400 + 1221):
        table = table_vapor_oneline(tem_u)
    if i >= 1400 and i < 1600:
        esupc = table_vapor_oneline(tem_u)
        wice = 0.05 * (constants.TICE - tem_u)
        wh2o = 0.05 * (tem_u - 253.16)
        table = wice * table + wh2o * esupc
    return table


# TODO Math can be consolidated if we can call gtscript functions from
# conditionals, fac0 and fac2 functions and others.
@gtscript.function
def qs_table2_fn(i):
    tem0 = tem_lower(i)
    if i < 1600:
        # compute es over ice between - 160 deg c and 0 deg c.
        table2 = table_ice_oneline(tem0)
    else:
        # compute es over water between 0 deg c and 102 deg c.
        table2 = table_vapor_oneline(tem0)
    if i == 1599:
        # table(i)
        table = table_ice_oneline(tem0)
        tem0 = tem_upper(i - 1400)
        table = (0.05 * (constants.TICE - tem0)) * table + (
            0.05 * (tem0 - 253.16)
        ) * table_vapor_oneline(tem0)
        # table2(i - 1)
        tem0 = tem_lower(1598)
        table2_m1 = table_ice_oneline(tem0)
        # table2(i + 1)
        tem0 = tem_lower(1600)
        table2_p1 = table_vapor_oneline(tem0)
        table2 = 0.25 * (table2_m1 + 2.0 * table + table2_p1)
    if i == 1600:
        # table(i)
        tem0 = tem_upper(i - 1400)
        table = table_vapor_oneline(tem0)
        # table2(i - 1)
        tem0 = tem_lower(1599)
        table2_m1 = table_ice_oneline(tem0)
        # table2(i + 1)
        tem0 = tem_lower(1601)
        table2_p1 = table_vapor_oneline(tem0)
        table2 = 0.25 * (table2_m1 + 2.0 * table + table2_p1)
    return table2


@gtscript.function
def qs_tablew_fn(i):
    tem = tem_lower(i)
    return table_vapor_oneline(tem)


@gtscript.function
def des_end(t, i, z, des2):
    if i == QS_LENGTH - 1:
        t_m1 = qs_table2_fn(i - 1)
        diff = t - t_m1
        des2 = max(z, diff)
    return des2


# TODO There might be a cleaner way to set des2[QS_LENGTH - 1] to des2[QS_LENGTH
# - 2].
@gtscript.function
def des2_table(i):
    t = qs_table2_fn(i)
    diff = qs_table2_fn(i + 1) - t
    z = 0.0
    des2 = max(z, diff)
    des2 = des_end(t, i, z, des2)
    return des2


# TODO There might be a cleaner way to set desw[QS_LENGTH - 1] to desw[QS_LENGTH
# - 2].
@gtscript.function
def desw_table(i):
    t = qs_tablew_fn(i)
    diff = qs_tablew_fn(i + 1) - t
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
    if (qi > 1.0e-8) and (pt1 > constants.TICE):
        factmp = fac_imlt * (pt1 - constants.TICE) / icp2
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
    tmpmax = max(qb, 0.0)
    return min(-qa, tmpmax)


@gtscript.function
def fix_negative_snow(qs, qg):
    if qs < 0.0:
        qg = qg + qs
        qs = 0.0
    elif qg < 0.0:
        tmp = minmax_tmp_h20(qg, qs)
        qg = qg + tmp
        qs = qs - tmp
    return qs, qg


# Fix negative cloud water with rain or rain with available cloud water
@gtscript.function
def fix_negative_cloud_water(ql, qr):
    if ql < 0.0:
        tmp = minmax_tmp_h20(ql, qr)
        ql = ql + tmp
        qr = qr - tmp
    elif qr < 0.0:
        tmp = minmax_tmp_h20(qr, ql)
        ql = ql - tmp
        qr = qr + tmp
    return ql, qr


# Enforce complete freezing of cloud water to cloud ice below - 48 c
@gtscript.function
def complete_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap):
    dtmp = constants.TICE - 48.0 - pt1
    if ql > 0.0 and dtmp > 0.0:
        sink = min(ql, dtmp / icp2)
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return ql, qi, q_liq, q_sol, cvm, pt1


@gtscript.function
def homogenous_freezing(qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap):
    dtmp = constants.T_WFR - pt1  # [ - 40, - 48]
    if ql > 0.0 and dtmp > 0.0:
        sink = min(ql, dtmp / icp2)
        sink = min(sink, ql * dtmp * 0.125)
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return ql, qi, q_liq, q_sol, cvm, pt1


# Bigg mechanism for heterogeneous freezing
@gtscript.function
def heterogeneous_freezing(
    exptc, pt1, cvm, ql, qi, q_liq, q_sol, den, icp2, dt_bigg, mc_air, lhi, qv, c_vap
):
    tc = constants.TICE0 - pt1
    if ql > 0.0 and tc > 0.0:
        sink = 3.3333e-10 * dt_bigg * (exptc - 1.0) * den * ql ** 2
        sink = min(ql, sink)
        sink = min(sink, tc / icp2)
        ql = ql - sink
        qi = qi + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return ql, qi, q_liq, q_sol, cvm, pt1


@gtscript.function
def make_graupel(pt1, cvm, fac_r2g, qr, qg, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap):
    dtmp = (constants.TICE - 0.1) - pt1
    if qr > 1e-7 and dtmp > 0.0:
        rainfac = (dtmp * 0.025) ** 2
        #  no limit on freezing below - 40 deg c
        tmp = qr if 1.0 < rainfac else rainfac * qr
        sink = min(tmp, fac_r2g * dtmp / icp2)
        qr = qr - sink
        qg = qg + sink
        q_liq = q_liq - sink
        q_sol = q_sol + sink
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        pt1 = add_src_pt1(pt1, sink, lhi, cvm)
    return qr, qg, q_liq, q_sol, cvm, pt1


@gtscript.function
def melt_snow(
    pt1, cvm, fac_smlt, qs, ql, qr, q_liq, q_sol, lhi, icp2, mc_air, qv, c_vap, qs_mlt
):
    dtmp = pt1 - (constants.TICE + 0.1)
    dimqs = dim(qs_mlt, ql)
    if qs > 1e-7 and dtmp > 0.0:
        snowfac = (dtmp * 0.1) ** 2
        tmp = (
            qs if 1.0 < snowfac else snowfac * qs
        )  # no limiter on melting above 10 deg c
        sink = min(tmp, fac_smlt * dtmp / icp2)
        tmp = min(sink, dimqs)
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
    qi_gen,
    qi_lim,
):
    from __externals__ import t_sub

    src = 0.0
    if pt1 < t_sub:
        src = dim(qv, 1e-6)
    elif pt1 < constants.TICE0:
        dq = qv - iqs2
        sink = adj_fac * dq / (1.0 + tcp2 * dqsdt)
        if qi > 1.0e-8:
            pidep = (
                sdt
                * dq
                * 349138.78
                * expsubl
                / (
                    iqs2
                    * den
                    * constants.LAT2
                    / (0.0243 * constants.RVGAS * pt1 ** 2.0)
                    + 4.42478e4
                )
            )
        else:
            pidep = 0.0
        if dq > 0.0:
            tmp = constants.TICE - pt1
            qi_crt = (
                qi_gen * qi_lim / den
                if qi_lim < 0.1 * tmp
                else qi_gen * 0.1 * tmp / den
            )
            maxtmp = qi_crt - qi if qi_crt - qi > pidep else pidep
            src = sink if sink < maxtmp else maxtmp
            src = src if src < tmp / tcp2 else tmp / tcp2
        else:
            dimtmp = dim(pt1, t_sub)
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
    lhi = constants.LI00 + constants.DC_ICE * pt1
    icp2 = lhi / cvm
    return lhi, icp2


@gtscript.function
def update_latent_heat_coefficient_l(pt1, cvm, lv00, d0_vap):
    lhl = lv00 + d0_vap * pt1
    lcp2 = lhl / cvm
    return lhl, lcp2


@gtscript.function
def update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap):
    lhl, lcp2 = update_latent_heat_coefficient_l(pt1, cvm, lv00, d0_vap)
    lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
    return lhl, lhi, lcp2, icp2


@gtscript.function
def compute_dq0(qv, wqsat, dq2dt, tcp3):
    return (qv - wqsat) / (1.0 + tcp3 * dq2dt)


@gtscript.function
def get_factor(wqsat, qv, fac_l2v):
    factor = -min(1, fac_l2v * 10.0 * (1.0 - qv / wqsat))
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


@gtscript.function
def ap1_for_wqs2(ta):
    ap1 = 10.0 * dim(ta, constants.T_SAT_MIN) + 1.0
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


def compute_q_tables(
    index: FloatField,
    tablew: FloatField,
    table2: FloatField,
    table: FloatField,
    desw: FloatField,
    des2: FloatField,
):
    """
    Args:
        index (in):
        tablew (out):
        table2 (out):
        table (out):
        desw (out):
        des2 (out):
    """
    with computation(PARALLEL), interval(...):
        tablew = qs_tablew_fn(index)
        table2 = qs_table2_fn(index)
        table = qs_table_fn(index)
        desw = desw_table(index)
        des2 = des2_table(index)


def satadjust(
    peln: FloatField,
    qv: FloatField,
    ql: FloatField,
    qi: FloatField,
    qr: FloatField,
    qs: FloatField,
    cappa: FloatField,
    qg: FloatField,
    pt: FloatField,
    dp: FloatField,
    delz: FloatField,
    te0: FloatField,
    q_con: FloatField,
    qa: FloatField,
    area: FloatFieldIJ,
    hs: FloatFieldIJ,
    pkz: FloatField,
    sdt: float,
    zvir: float,
    fac_i2s: float,
    do_qa: bool,
    consv_te: bool,
    c_air: float,
    c_vap: float,
    mdt: float,
    fac_r2g: float,
    fac_smlt: float,
    fac_l2r: float,
    fac_imlt: float,
    d0_vap: float,
    lv00: float,
    fac_v2l: float,
    fac_l2v: float,
    last_step: bool,
):
    """
    Documented in Zhou, Harris and Chen (2022)
    https://repository.library.noaa.gov/view/noaa/44636.

    Args:
        peln (in):
        qv (inout):
        ql (inout):
        qi (inout):
        qr (inout):
        qs (inout):
        cappa (out):
        qg (inout):
        pt (inout):
        dp (in):
        delz (inout): If nonhydrostatic delz is only in, not out
        te0 (out):
        q_con (out):
        qa (out):
        area (in):
        hs (in):
        pkz (out):
        sdt (in):
        zvir (in): epsilon parameter in virtual temperature
        fac_i2s (in):
        do_qa (in): enables the cloud fraction scheme accounting for subgrid
            variability in cloud fraction
        consv_te (in):
        c_air (in):
        c_vap (in):
        mdt (in):
        fac_r2g (in):
        fac_smlt (in):
        fac_l2r (in):
        fac_imlt (in):
        d0_vap (in): Cvapor - Cliquid, used for Clausius-Clapeyron
        lv00 (in): latent heating of vaporization, HLV - d0_vap * TICE
        fac_v2l (in):
        fac_l2v (in):
        last_step (in):
    """
    from __externals__ import (
        cld_min,
        dw_land,
        dw_ocean,
        hydrostatic,
        icloud_f,
        qi0_max,
        qi_gen,
        qi_lim,
        ql0_max,
        ql_gen,
        qs_mlt,
        rad_graupel,
        rad_rain,
        rad_snow,
        sat_adj0,
        tintqs,
    )

    with computation(PARALLEL), interval(1, None):
        if __INLINED(hydrostatic):
            delz_0 = delz[0, 0, -1]
            delz = delz_0
    with computation(PARALLEL), interval(...):
        q_liq = ql + qr
        q_sol = qi + qs + qg
        qpz = q_liq + q_sol
        pt1 = pt / ((1.0 + zvir * qv) * (1.0 - qpz))
        t0 = pt1  # true temperature
        qpz = qpz + qv  # total_wat conserved in this routine
        # define air density based on hydrostatical property
        if __INLINED(hydrostatic):
            den = dp / ((peln[0, 0, 1] - peln) * constants.RDGAS * pt)
        else:
            den = -dp / (constants.GRAV * delz)
        # define heat capacity and latend heat coefficient
        mc_air = (1.0 - qpz) * c_air
        cvm = compute_cvm(mc_air, qv, c_vap, q_liq, q_sol)
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        #  fix energy conservation
        if consv_te:
            if __INLINED(hydrostatic):
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
        diff_ice = dim(constants.TICE, pt1) / 48.0
        dimmin = min(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin

        dq0 = compute_dq0(qv, wqsat, dq2dt, tcp3)
        # TODO Might be able to get rid of these temporary allocations when not used?
        if dq0 > 0:  # whole grid - box saturated
            src = min(
                sat_adj0 * dq0,
                max(ql_gen - ql, fac_v2l * dq0),
            )
        else:
            factor, src = ql_evaporation(wqsat, qv, ql, dq0, fac_l2v)

        qv, ql, q_liq, cvm, pt1 = wqsat_correct(
            src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap
        )
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        # TODO: Remove duplicate
        diff_ice = dim(constants.TICE, pt1) / 48.0
        dimmin = min(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin

        if last_step:
            wqsat, dq2dt = wqs2_fn_w(pt1, den)

            dq0 = compute_dq0(qv, wqsat, dq2dt, tcp3)
            if dq0 > 0:
                src = dq0
            else:
                factor, src = ql_evaporation(wqsat, qv, ql, dq0, fac_l2v)

            qv, ql, q_liq, cvm, pt1 = wqsat_correct(
                src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap
            )

            lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(
                pt1, cvm, lv00, d0_vap
            )

        # homogeneous freezing of cloud water to cloud ice
        ql, qi, q_liq, q_sol, cvm, pt1 = homogenous_freezing(
            qv, ql, qi, q_liq, q_sol, pt1, cvm, icp2, mc_air, lhi, c_vap
        )
        # update some of the latent heat coefficients
        lhi, icp2 = update_latent_heat_coefficient_i(pt1, cvm)
        exptc = exp(0.66 * (constants.TICE0 - pt1))
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
            qs_mlt,
        )
        #  autoconversion from cloud water to rain
        ql, qr = autoconversion_cloud_to_rain(ql, qr, fac_l2r, ql0_max)
        iqs2, dqsdt = wqs2_fn_2(pt1, den)
        expsubl = exp(0.875 * log(qi * den))
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        tcp2 = lcp2 + icp2

        if last_step:
            adj_fac = 1.0
        else:
            adj_fac = sat_adj0

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
            qi_gen,
            qi_lim,
        )
        # virtual temp updated
        q_con = q_liq + q_sol
        tmp = 1.0 + zvir * qv
        pt = pt1 * tmp * (1.0 - q_con)
        tmp *= constants.RDGAS
        cappa = tmp / (tmp + cvm)
        #  fix negative graupel with available cloud ice
        if qg < 0:
            maxtmp = max(0.0, qi)
            mintmp = min(-qg, maxtmp)
            qg = qg + mintmp
            qi = qi - mintmp
        else:
            qg = qg
        #  autoconversion from cloud ice to snow
        qim = qi0_max / den
        if qi > qim:
            sink = fac_i2s * (qi - qim)
            qi = qi - sink
            qs = qs + sink
        # fix energy conservation
        if consv_te:
            if __INLINED(hydrostatic):
                te0 = dp * (te0 + c_air * pt1)
            else:
                te0 = dp * (te0 + cvm * pt1)
        # update latent heat coefficient
        cvm = mc_air + (qv + q_liq + q_sol) * c_vap
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        # compute cloud fraction
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
            # use the "liquid - frozen water temperature" (tin) to compute
            # saturated specific humidity
            if tintqs:
                tin = pt1
            else:
                tin = pt1 - (lcp2 * q_cond + icp2 * q_sol)

            # CK : Additions from satadjust_part3_laststep_qa
            it, ap1 = ap1_and_index(tin)
            wqs1 = wqs1_fn_w(it, ap1, tin, den)
            iqs1 = wqs1_fn_2(it, ap1, tin, den)
            # Determine saturated specific humidity
            if tin < constants.T_WFR:
                # ice phase
                qstar = iqs1
            elif tin >= constants.TICE:
                qstar = wqs1
            else:
                # qsw = wqs1
                if q_cond > 1e-6:
                    rqi = q_sol / q_cond
                else:
                    rqi = (constants.TICE - tin) / (constants.TICE - constants.T_WFR)
                qstar = rqi * iqs1 + (1.0 - rqi) * wqs1
                # higher than 10 m is considered "land" and will have higher subgrid
                # variability
            mindw = min(1.0, abs(hs) / (10.0 * constants.GRAV))
            dw = dw_ocean + (dw_land - dw_ocean) * mindw
            # "scale - aware" subgrid variability: 100 - km as the base
            dbl_sqrt_area = dw * (area ** 0.5 / 100.0e3) ** 0.5
            maxtmp = max(0.01, dbl_sqrt_area)
            hvar = min(0.2, maxtmp)
            # partial cloudiness by pdf:
            # assuming subgrid linear distribution in horizontal; this is
            # effectively a smoother for the binary cloud scheme;
            # qa = 0.5 if qstar == qpz
            rh = qpz / qstar
            # icloud_f = 0: bug - fixed
            # icloud_f = 1: old fvgfs gfdl) mp implementation
            # icloud_f = 2: binary cloud scheme (0 / 1)
            if rh > 0.75 and qpz > 1.0e-8:
                dq = hvar * qpz
                q_plus = qpz + dq
                q_minus = qpz - dq
                if icloud_f == 2:  # TODO untested
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
                            if icloud_f == 0:
                                qa = (q_plus - qstar) / (dq + dq)
                            else:
                                qa = (q_plus - qstar) / (2.0 * dq * (1.0 - q_cond))
                        else:
                            qa = 0.0
                        # impose minimum cloudiness if substantial q_cond exist
                        if q_cond > 1.0e-8:
                            qa = max(cld_min, qa)
                        qa = min(1, qa)
            else:
                qa = 0.0

        if __INLINED(not hydrostatic):
            pkz = compute_pkz_func(dp, delz, pt, cappa)


class SatAdjust3d:
    def __init__(
        self, stencil_factory: StencilFactory, config: SatAdjustConfig, area_64, kmp
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._config = config
        self._area_64 = area_64

        self._satadjust_stencil = stencil_factory.from_origin_domain(
            func=satadjust,
            externals={
                "hydrostatic": self._config.hydrostatic,
                "rad_snow": self._config.rad_snow,
                "rad_rain": self._config.rad_rain,
                "rad_graupel": self._config.rad_graupel,
                "tintqs": self._config.tintqs,
                "sat_adj0": self._config.sat_adj0,
                "ql_gen": self._config.ql_gen,
                "qs_mlt": self._config.qs_mlt,
                "ql0_max": self._config.ql0_max,
                "t_sub": self._config.t_sub,
                "qi_gen": self._config.qi_gen,
                "qi_lim": self._config.qi_lim,
                "qi0_max": self._config.qi0_max,
                "dw_ocean": self._config.dw_ocean,
                "dw_land": self._config.dw_land,
                "icloud_f": self._config.icloud_f,
                "cld_min": self._config.cld_min,
            },
            origin=(grid_indexing.isc, grid_indexing.jsc, int(kmp)),
            domain=(
                grid_indexing.domain[0],
                grid_indexing.domain[1],
                int(grid_indexing.domain[2] - kmp),
            ),
        )

    def __call__(
        self,
        te: FloatField,
        qvapor: FloatField,
        qliquid: FloatField,
        qice: FloatField,
        qrain: FloatField,
        qsnow: FloatField,
        qgraupel: FloatField,
        qcld: FloatField,
        hs: FloatFieldIJ,
        peln: FloatField,
        delp: FloatField,
        delz: FloatField,
        q_con: FloatField,
        pt: FloatField,
        pkz: FloatField,
        cappa: FloatField,
        r_vir: float,
        mdt: float,
        fast_mp_consv: bool,
        last_step: bool,
        akap: float,
        kmp: int,
    ):
        """
        Fast phase change as part of GFDL microphysics.

        Grid-scale condensation.

        Args:
            te (out):
            qvapor (inout):
            qliquid (inout):
            qice (inout):
            qrain (inout):
            qsnow (inout):
            qgraupel (inout):
            qcld (out):
            hs (in):
            peln (in): only read if hydrostatic, otherwise unused
            delp (in):
            delz (inout): If nonhydrostatic delz is only in, not out
            q_con (out):
            pt (inout):
            pkz (out):
            cappa (out):
            r_vir (in):
            mdt (in):
            fast_mp_consv (in):
            last_step (in):
            akap (unused):
            kmp (unused):
        """
        # TODO: akap and kmp are not used and should be removed from the call
        # TODO: Maybe remove hydrostatic code
        sdt = 0.5 * mdt  # half remapping time step
        # define conversion scalar / factor
        # ice to snow
        fac_ice_to_snow = 1.0 - math.exp(-mdt / self._config.tau_i2s)
        # vapor to liquid
        fac_vapor_to_liquid = 1.0 - math.exp(-sdt / self._config.tau_v2l)
        # rain to graupel
        fac_rain_to_graupel = 1.0 - math.exp(-mdt / self._config.tau_r2g)
        # liquid to rain
        fac_liquid_to_rain = 1.0 - math.exp(-mdt / self._config.tau_l2r)

        # liquid to vapor
        fac_liquid_to_vapor = 1.0 - math.exp(-sdt / self._config.tau_l2v)
        fac_liquid_to_vapor = min(self._config.sat_adj0, fac_liquid_to_vapor)

        # time scale for ice melting (i2r)
        fac_ice_to_rain = 1.0 - math.exp(-sdt / self._config.tau_imlt)
        # snow melting (s2r)
        fac_snow_to_rain = 1.0 - math.exp(-mdt / self._config.tau_smlt)

        # define heat capacity of dry air and water vapor based on hydrostatical
        # property

        if self._config.hydrostatic:
            c_air = constants.CP_AIR
            c_vap = constants.CP_VAP
        else:
            c_air = constants.CV_AIR
            c_vap = constants.CV_VAP

        d0_vap = c_vap - constants.C_LIQ
        lv00 = constants.HLV - d0_vap * constants.TICE

        do_qa = True

        self._satadjust_stencil(
            peln,
            qvapor,
            qliquid,
            qice,
            qrain,
            qsnow,
            cappa,
            qgraupel,
            pt,
            delp,
            delz,
            te,
            q_con,
            qcld,
            self._area_64,
            hs,
            pkz,
            sdt,
            r_vir,
            fac_ice_to_snow,
            do_qa,
            fast_mp_consv,
            c_air,
            c_vap,
            mdt,
            fac_rain_to_graupel,
            fac_snow_to_rain,
            fac_liquid_to_rain,
            fac_ice_to_rain,
            d0_vap,
            lv00,
            fac_vapor_to_liquid,
            fac_liquid_to_vapor,
            last_step,
        )
