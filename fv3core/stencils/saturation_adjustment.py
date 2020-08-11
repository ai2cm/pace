#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3core._config as spec
import fv3core.stencils.moist_cv as moist_cv
from gt4py.gtscript import computation, interval, PARALLEL
import fv3core.utils.global_constants as constants
from fv3core.stencils.basic_operations import max_fn, min_fn, dim
import math
import numpy as np

# TODO, this code could be reduced greatly with abstraction, but first gt4py needs to support gtscript function calls of arbitrary depth embedded in conditionals
sd = utils.sd
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


def qs_init():
    length = QS_LENGTH
    for tablename in satmix.keys():
        if satmix[tablename] is None:
            satmix[tablename] = np.zeros(length)
        else:
            return  # already initialized any table means initializec them all
    qs_table(length)
    qs_table2(length)
    qs_tablew(length)
    for i in range(length - 1):
        satmix["des2"][i] = max(0.0, satmix["table2"][i + 1] - satmix["table2"][i])
        satmix["desw"][i] = max(0.0, satmix["tablew"][i + 1] - satmix["tablew"][i])
    satmix["des2"][length - 1] = satmix["des2"][length - 2]
    satmix["desw"][length - 1] = satmix["desw"][length - 2]


def get_fac0(tem):
    return (tem - TICE) / (tem * TICE)


def get_fac2(tem, fac1, d):
    return (d * math.log(tem / TICE) + fac1) / constants.RVGAS


# TODO refactor into streamlined array calcs
def qs_table(n):
    esupc = np.zeros(200)
    # compute es over ice between - 160 deg c and 0 deg c.
    for i in range(1600):
        tem = TMIN + DELT * i  # (i - 1)
        fac2 = get_fac2(tem, get_fac0(tem) * LI2, D2ICE)
        satmix["table"][i] = E00 * math.exp(fac2)

    # compute es over water between - 20 deg c and 102 deg c.
    for i in range(1221):
        tem = 253.16 + DELT * i  # real (i - 1)
        fac2 = get_fac2(tem, get_fac0(tem) * LV0, DC_VAP)
        esh20 = E00 * math.exp(fac2)
        if i < 200:
            esupc[i] = esh20
        else:
            satmix["table"][i + 1400] = esh20  # TODO

    #  derive blended es over ice and supercooled water between - 20 deg c and 0 deg c
    for i in range(200):
        tem = 253.16 + DELT * i  # real (i - 1)
        wice = 0.05 * (TICE - tem)
        wh2o = 0.05 * (tem - 253.16)
        satmix["table"][i + 1400] = wice * satmix["table"][i + 1400] + wh2o * esupc[i]


def qs_table2(n):
    for i in range(n):
        tem0 = TMIN + DELT * i
        fac0 = get_fac0(tem0)
        if i < 1600:
            # compute es over ice between - 160 deg c and 0 deg c.
            fac2 = get_fac2(tem0, fac0 * LI2, D2ICE)
        else:
            # compute es over water between 0 deg c and 102 deg c.
            fac2 = get_fac2(tem0, fac0 * LV0, DC_VAP)
        satmix["table2"][i] = E00 * math.exp(fac2)

    # smoother around 0 deg c
    smooth_indices = {1599: None, 1600: None}
    for i in smooth_indices.keys():
        smooth_indices[i] = 0.25 * (
            satmix["table2"][i - 1] + 2.0 * satmix["table"][i] + satmix["table2"][i + 1]
        )
    for i, v in smooth_indices.items():
        satmix["table2"][i] = v


def qs_tablew(n):
    for i in range(n):
        tem = TMIN + DELT * i
        fac2 = get_fac2(tem, get_fac0(tem) * LV0, DC_VAP)
        satmix["tablew"][i] = E00 * math.exp(fac2)


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
    factmp = 0.0  # TODO, if temporaries inside of if-statements become supported, remove this
    sink = 0.0  # TODO ditto factmp
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
        tmp = (
            qr if 1.0 < rainfac else rainfac * qr
        )  #  no limit on freezing below - 40 deg c
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
    factor = -1 if 1 < factor else -factor  # min_fn(1, factor) * -1
    return factor


@gtscript.function
def get_src(ql, factor, dq0):
    src = -ql if ql < factor * dq0 else -factor * dq0  # min_func(ql, factor * dq0) * -1
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


@utils.stencil()
def ap1_for_wqs2(ta: sd, ap1: sd):
    with computation(PARALLEL), interval(...):
        ap1 = 10.0 * dim(ta, TMIN) + 1.0
        ap1 = min_fn(ap1, QS_LENGTH) - 1


@utils.stencil()
def wqs2_stencil(
    ta: sd,
    den: sd,
    ap1: sd,
    it: sd,
    it2: sd,
    tablew_lookup: sd,
    desw_lookup: sd,
    desw2_lookup: sd,
    desw_p1_lookup: sd,
    wqsat: sd,
    dqdt: sd,
):
    with computation(PARALLEL), interval(...):
        es = tablew_lookup + (ap1 - it) * desw_lookup
        denom = constants.RVGAS * ta * den
        wqsat = es / denom
        dqdt = 10.0 * (desw2_lookup + (ap1 - it2) * (desw_p1_lookup - desw2_lookup))
        dqdt = dqdt / denom


@utils.stencil()
def wqs1_stencil(
    ta: sd, den: sd, ap1: sd, it: sd, tablew_lookup: sd, desw_lookup: sd, wqsat: sd
):
    with computation(PARALLEL), interval(...):
        es = tablew_lookup + (ap1 - it) * desw_lookup
        wqsat = es / (constants.RVGAS * ta * den)


# TODO put in gt4py (compute table values on the fly rather than using a lookup?)
# The function wqs2_vect computes the gradient of saturated specific humidity for table ii. with arguments tablename=tablew, desname=desw
# iqs2 computes the gradient of saturated specific humidity for table iii. with arguments tablename=table2, desname=des2
def wqs2_iqs2(ta, den, wqsat, dqdt, tablename="tablew", desname="desw"):
    ap1 = utils.make_storage_from_shape(ta.shape, utils.origin)
    ap1_for_wqs2(ta, ap1, origin=(0, 0, 0), domain=spec.grid.domain_shape_standard())
    it = ap1.data.astype(int)
    itgt = utils.make_storage_data(it, ta.shape)
    tablew_lookup = utils.make_storage_data(utils.index(satmix[tablename], it), ta.shape)
    desw_lookup = utils.make_storage_data(utils.index(satmix[desname], it), ta.shape)
    it2 = (ap1 - 0.5).data.astype(int)
    it2gt = utils.make_storage_data(it2, ta.shape)
    desw2_lookup = utils.make_storage_data(utils.index(satmix[desname], it2), ta.shape)
    desw2_p1_lookup = utils.make_storage_data(utils.index(satmix[desname], it2 + 1), ta.shape)
    wqs2_stencil(
        ta,
        den,
        ap1,
        itgt,
        it2gt,
        tablew_lookup,
        desw_lookup,
        desw2_lookup,
        desw2_p1_lookup,
        wqsat,
        dqdt,
        origin=(0, 0, 0),
        domain=spec.grid.domain_shape_standard(),
    )


def wqs1_iqs1(ta, den, wqsat, tablename="tablew", desname="desw"):
    ap1 = utils.make_storage_from_shape(ta.shape, utils.origin)
    ap1_for_wqs2(ta, ap1, origin=(0, 0, 0), domain=spec.grid.domain_shape_standard())
    it = ap1.data.astype(int)
    itgt = utils.make_storage_data(it, ta.shape)
    tablew_lookup = utils.make_storage_data(utils.index(satmix[tablename], it), ta.shape)
    desw_lookup = utils.make_storage_data(utils.index(satmix[desname], it), ta.shape)
    wqs1_stencil(
        ta,
        den,
        ap1,
        itgt,
        tablew_lookup,
        desw_lookup,
        wqsat,
        origin=(0, 0, 0),
        domain=spec.grid.domain_shape_standard(),
    )


@utils.stencil()
def satadjust_part1(
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
    lhi: sd,
    icp2: sd,
    zvir: float,
    hydrostatic: bool,
    consv_te: bool,
    c_air: float,
    c_vap: float,
    fac_imlt: float,
    d0_vap: float,
    lv00: float,
):
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


@utils.stencil()
def satadjust_part2(
    wqsat: sd,
    dq2dt: sd,
    pt1: sd,
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
    q_sol: sd,
    fac_v2l: float,
    fac_l2v: float,
    lv00: float,
    d0_vap: float,
    c_vap: float,
    adj_fac: float,
    ql_gen: float,
):
    with computation(PARALLEL), interval(...):
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = min_fn(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin

        dq0 = (qv - wqsat) / (
            1.0 + tcp3 * dq2dt
        )  # compute_dq0(qv, wqsat, dq2dt, tcp3)  #(qv - wqsat) / (1.0 + tcp3 * dq2dt)
        # TODO might be able to get rid of these temporary allocations when not used?
        tmpmax = 0.0
        src = 0.0
        factor = 0.0
        a = 0.0
        b = 0.0
        if dq0 > 0:  # whole grid - box saturated
            a = ql_gen - ql
            b = fac_v2l * dq0
            tmpmax = (
                a if a > b else b
            )  # max_fn(a, b) -- this yields an incorrect answer
            src = adj_fac * dq0 if adj_fac * dq0 < tmpmax else tmpmax
        else:
            # TODO -- we'd like to use this abstraction rather than duplicate code, but inside the if conditional complains 'not implemented'
            # factor, src = ql_evaporation(wqsat, qv, ql, dq0,fac_l2v)
            factor = fac_l2v * 10.0 * (1.0 - qv / wqsat)
            factor = -1.0 if 1.0 < factor else -factor
            src = -ql if ql < factor * dq0 else -factor * dq0
        qv, ql, q_liq, cvm, pt1 = wqsat_correct(
            src, pt1, lhl, qv, ql, q_liq, q_sol, mc_air, c_vap
        )
        # update latent heat coefficient
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        # TODO remove duplicate
        diff_ice = dim(TICE, pt1) / 48.0
        dimmin = min_fn(1.0, diff_ice)
        tcp3 = lcp2 + icp2 * dimmin


@utils.stencil()
def satadjust_part3(
    wqsat: sd,
    dq2dt: sd,
    pt1: sd,
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
    fac_v2l: float,
    fac_l2v: float,
    lv00: float,
    d0_vap: float,
    c_vap: float,
    last_step: bool,
):
    with computation(PARALLEL), interval(...):
        dq0 = 0.0
        src = 0.0
        factor = 0.0
        if last_step:
            dq0 = compute_dq0(qv, wqsat, dq2dt, tcp3)
            if dq0 > 0:
                src = dq0
            else:
                # TODO -- we'd like to use this abstraction rather than duplicate code, but inside the if conditional complains 'not implemented'
                # factor, src = ql_evaporation(wqsat, qv, ql, dq0,fac_l2v)
                factor = fac_l2v * 10.0 * (1.0 - qv / wqsat)
                factor = -1.0 if 1.0 < factor else -factor  # min_fn(1, factor) * -1
                src = (
                    -ql if ql < factor * dq0 else -factor * dq0
                )  # min_func(ql, factor * dq0) * -1
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


# TODO reading in ql0_max as a runtime argument causes problems for the if statement
@utils.stencil()
def satadjust_part4(
    den: sd,
    pt1: sd,
    cvm: sd,
    mc_air: sd,
    lhl: sd,
    lhi: sd,
    lcp2: sd,
    icp2: sd,
    exptc: sd,
    qv: sd,
    ql: sd,
    q_liq: sd,
    qi: sd,
    q_sol: sd,
    qr: sd,
    qg: sd,
    qs: sd,
    c_vap: float,
    mdt: float,
    fac_r2g: float,
    fac_smlt: float,
    fac_l2r: float,
    qs_mlt: float,
    ql0_max: float,
    last_step: bool,
):
    with computation(PARALLEL), interval(...):
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
        # TODO ql0_max is supposed to come from the namelist, but runtime floats cause an error while constants do not
        ql, qr = autoconversion_cloud_to_rain(ql, qr, fac_l2r, constants.ql0_max)


# TODO -- reading in t_sub, as runtime variables is causing weird issues with the if statements
@utils.stencil()
def satadjust_part5(
    pt: sd,
    cappa: sd,
    tin: sd,
    te0: sd,
    dp: sd,
    q_cond: sd,
    q_con: sd,
    expsubl: sd,
    iqs2: sd,
    dqsdt: sd,
    den: sd,
    pt1: sd,
    cvm: sd,
    mc_air: sd,
    lhl: sd,
    lhi: sd,
    lcp2: sd,
    icp2: sd,
    qv: sd,
    ql: sd,
    q_liq: sd,
    qi: sd,
    q_sol: sd,
    qr: sd,
    qg: sd,
    qs: sd,
    lv00: float,
    d0_vap: float,
    c_vap: float,
    sdt: float,
    adj_fac: float,
    zvir: float,
    fac_i2s: float,
    c_air: float,
    t_sub: float,
    qi_gen: float,
    qi_lim: float,
    qi0_max: float,
    consv_te: bool,
    hydrostatic: bool,
    do_qa: bool,
    rad_snow: bool,
    rad_rain: bool,
    rad_graupel: bool,
    tintqs: bool,
    last_step: bool,
):
    with computation(PARALLEL), interval(...):
        lhl, lhi, lcp2, icp2 = update_latent_heat_coefficient(pt1, cvm, lv00, d0_vap)
        tcp2 = lcp2 + icp2
        # TODO t_sub is supposed to come from the namelist, but currently gt4py breaks with a coniditional comparison of a runtime float (vs a constant)
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
            constants.t_sub,
            qi_gen,
            qi_lim,
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
        qim = qi0_max / den
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


@utils.stencil()
def satadjust_part6_laststep_qa(
    qa: sd,
    area: sd,
    qpz: sd,
    hs: sd,
    tin: sd,
    q_cond: sd,
    iqs1: sd,
    wqs1: sd,
    q_sol: sd,
    dw_ocean: float,
    dw_land: float,
    icloud_f: int,
    cld_min: float,
):
    with computation(PARALLEL), interval(...):
        qstar = 0.0
        rqi = 0.0
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
        abshs = hs if hs > 0 else -hs
        mindw = min_fn(1.0, abshs / (10.0 * constants.GRAV))
        dw = dw_ocean + (dw_land - dw_ocean) * mindw
        # "scale - aware" subgrid variability: 100 - km as the base
        dbl_sqrt_area = dw * (area ** 0.5 / 100.0e3) ** 0.5
        maxtmp = 0.01 if 0.01 > dbl_sqrt_area else dbl_sqrt_area
        hvar = min_fn(0.2, maxtmp)
        # partial cloudiness by pdf:
        # assuming subgrid linear distribution in horizontal; this is effectively a smoother for the
        # binary cloud scheme; qa = 0.5 if qstar == qpz
        rh = qpz / qstar
        # icloud_f = 0: bug - fixed
        # icloud_f = 1: old fvgfs gfdl) mp implementation
        # icloud_f = 2: binary cloud scheme (0 / 1)
        dq = 0.0
        q_plus = 0.0
        q_minus = 0.0
        if rh > 0.75 and qpz > 1.0e-8:
            dq = hvar * qpz
            q_plus = qpz + dq
            q_minus = qpz - dq
            # if (icloud_f == 2): # TODO this many if conditionals triggers an assertion error
            #    if (qpz > qstar):
            #        qa = 1.
            #    elif ((qstar < q_plus) and (q_cond > 1.e-8)):
            #        qa = ((q_plus - qstar) / dq)**2
            #        qa = 1. if 1. < qa else qa  # min_fn(1., qa)
            #    else:
            #        qa = 0.
            # else:
            qa = 0.0
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
                    qa = cld_min if cld_min > qa else qa  # max_fn(cld_min, qa)
                else:
                    qa = qa
                qa = 1.0 if 1.0 < qa else qa  # min_fn(1., qa)
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
    namelist = spec.namelist
    origin = (grid.is_, grid.js, kmp)
    domain = (grid.nic, grid.njc, (grid.npz - kmp))
    qs_init()
    hydrostatic = spec.namelist["hydrostatic"]
    sdt = 0.5 * mdt  # half remapping time step
    # define conversion scalar / factor
    fac_i2s = 1.0 - math.exp(-mdt / namelist["tau_i2s"])
    fac_v2l = 1.0 - math.exp(-sdt / namelist["tau_v2l"])
    fac_r2g = 1.0 - math.exp(-mdt / namelist["tau_r2g"])
    fac_l2r = 1.0 - math.exp(-mdt / namelist["tau_l2r"])

    fac_l2v = 1.0 - math.exp(-sdt / namelist["tau_l2v"])
    fac_l2v = min(namelist["sat_adj0"], fac_l2v)

    fac_imlt = 1.0 - math.exp(-sdt / namelist["tau_imlt"])
    fac_smlt = 1.0 - math.exp(-mdt / namelist["tau_smlt"])

    # define heat capacity of dry air and water vapor based on hydrostatical property

    if hydrostatic:
        c_air = constants.CP_AIR
        c_vap = constants.CP_VAP
        delz = utils.make_storage_data(
            np.squeeze(delz.data[:, :, 0]), te.shape, utils.origin()
        )
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
        lhi,
        icp2,
        r_vir,
        hydrostatic,
        fast_mp_consv,
        c_air,
        c_vap,
        fac_imlt,
        d0_vap,
        lv00,
        origin=origin,
        domain=domain,
    )
    wqs2_iqs2(pt1, den, wqsat, dq2dt)
    adj_fac = namelist["sat_adj0"]
    satadjust_part2(
        wqsat,
        dq2dt,
        pt1,
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
        q_sol,
        fac_v2l,
        fac_l2v,
        lv00,
        d0_vap,
        c_vap,
        adj_fac,
        namelist["ql_gen"],
        origin=origin,
        domain=domain,
    )
    if last_step:
        adj_fac = 1.0
        # condensation / evaporation between water vapor and cloud water, last time step
        #  enforce upper (no super_sat) & lower (critical rh) bounds
        # final iteration:
        wqs2_iqs2(pt1, den, wqsat, dq2dt)

    satadjust_part3(
        wqsat,
        dq2dt,
        pt1,
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
        fac_v2l,
        fac_l2v,
        lv00,
        d0_vap,
        c_vap,
        last_step,
        origin=origin,
        domain=domain,
    )
    exptc = np.exp(0.66 * (TICE0 - pt1))
    satadjust_part4(
        den,
        pt1,
        cvm,
        mc_air,
        lhl,
        lhi,
        lcp2,
        icp2,
        exptc,
        qvapor,
        qliquid,
        q_liq,
        qice,
        q_sol,
        qrain,
        qgraupel,
        qsnow,
        c_vap,
        mdt,
        fac_r2g,
        fac_smlt,
        fac_l2r,
        namelist["qs_mlt"],
        namelist["ql0_max"],
        last_step,
        origin=origin,
        domain=domain,
    )
    iqs2 = utils.make_storage_from_shape(peln.shape, utils.origin)
    dqsdt = utils.make_storage_from_shape(peln.shape, utils.origin)
    wqs2_iqs2(pt1, den, iqs2, dqsdt, tablename="table2", desname="des2")
    # TODO: this can be pushed into the stencil function as (qice * den)**0.875, but breaks validation
    expsubl = np.exp(0.875 * np.log(qice * den))
    do_qa = True  # TODO  -- this isn't a namelist option in Fortran, it is whether or not cld_amount is a tracer. If/when we support different sets of tracers, this will need to change
    satadjust_part5(
        pt,
        cappa,
        tin,
        te,
        delp,
        q_cond,
        q_con,
        expsubl,
        iqs2,
        dqsdt,
        den,
        pt1,
        cvm,
        mc_air,
        lhl,
        lhi,
        lcp2,
        icp2,
        qvapor,
        qliquid,
        q_liq,
        qice,
        q_sol,
        qrain,
        qgraupel,
        qsnow,
        lv00,
        d0_vap,
        c_vap,
        sdt,
        adj_fac,
        r_vir,
        fac_i2s,
        c_air,
        namelist["t_sub"],
        namelist["qi_gen"],
        namelist["qi_lim"],
        namelist["qi0_max"],
        fast_mp_consv,
        hydrostatic,
        do_qa,
        namelist["rad_snow"],
        namelist["rad_rain"],
        namelist["rad_graupel"],
        namelist["tintqs"],
        last_step,
        origin=origin,
        domain=domain,
    )
    if do_qa and last_step:
        iqs1 = utils.make_storage_from_shape(peln.shape, utils.origin)
        wqs1 = utils.make_storage_from_shape(peln.shape, utils.origin)
        wqs1_iqs1(tin, den, wqs1, tablename="tablew", desname="desw")
        wqs1_iqs1(tin, den, iqs1, tablename="table2", desname="des2")

        satadjust_part6_laststep_qa(
            qcld,
            grid.area_64,
            qpz,
            hs,
            tin,
            q_cond,
            iqs1,
            wqs1,
            q_sol,
            namelist["dw_ocean"],
            namelist["dw_land"],
            namelist["icloud_f"],
            namelist["cld_min"],
            origin=origin,
            domain=domain,
        )
    if not spec.namelist["hydrostatic"]:
        # moist_cv.compute_pkz_stencil_func(
        #    pkz, cappa, delp, delz, pt, origin=origin, domain=domain
        # )
        # TODO when we have math functions replace the below with the above
        tmpslice = (
            slice(grid.is_, grid.ie + 1),
            slice(grid.js, grid.je + 1),
            slice(kmp, grid.npz),
        )
        moist_cv.compute_pkz_slice(pkz, cappa, delp, delz, pt, tmpslice)
