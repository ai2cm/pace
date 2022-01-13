import numpy as np

import pace.util.constants as constants


def fpvs(t):
    """Compute saturation vapor pressure
       t: Temperature [K]
    fpvs: Vapor pressure [Pa]
    """

    con_psat = constants.PSAT
    con_ttp = constants.TTP
    con_cvap = constants.CVAP
    con_cliq = constants.C_LIQ
    con_hvap = constants.HLV
    con_rv = constants.RVGAS
    con_csol = constants.CSOL
    con_hfus = constants.HFUS

    tliq = con_ttp
    tice = con_ttp - 20.0
    dldtl = con_cvap - con_cliq
    heatl = con_hvap
    xponal = -dldtl / con_rv
    xponbl = -dldtl / con_rv + heatl / (con_rv * con_ttp)
    dldti = con_cvap - con_csol
    heati = con_hvap + con_hfus
    xponai = -dldti / con_rv
    xponbi = -dldti / con_rv + heati / (con_rv * con_ttp)

    convert_to_scalar = False
    if np.isscalar(t):
        t = np.array(t)
        convert_to_scalar = True

    fpvs = np.empty_like(t)
    tr = con_ttp / t

    ind1 = t >= tliq
    fpvs[ind1] = con_psat * (tr[ind1] ** xponal) * np.exp(xponbl * (1.0 - tr[ind1]))

    ind2 = t < tice
    fpvs[ind2] = con_psat * (tr[ind2] ** xponai) * np.exp(xponbi * (1.0 - tr[ind2]))

    ind3 = ~np.logical_or(ind1, ind2)
    w = (t[ind3] - tice) / (tliq - tice)
    pvl = con_psat * (tr[ind3] ** xponal) * np.exp(xponbl * (1.0 - tr[ind3]))
    pvi = con_psat * (tr[ind3] ** xponai) * np.exp(xponbi * (1.0 - tr[ind3]))
    fpvs[ind3] = w * pvl + (1.0 - w) * pvi

    if convert_to_scalar:
        fpvs = fpvs.item()

    return fpvs


def sfc_ocean(
    ps,
    t1,
    q1,
    tskin,
    cm,
    ch,
    prsl1,
    prslki,
    wet,
    wind,
    flag_iter,
    qsurf,
    cmm,
    chh,
    gflux,
    evap,
    hflx,
    ep,
):
    """Near-surface sea temperature scheme

    Args:
        ps: surface pressure (in)
        t1: surface layer mean temperature (in)
        q1: surface layer mean specific humidity (in)
        tskin: ground surface skin temperature (in)
        cm: surface exchange coefficient for momentum (in)
        ch: surface exchange coefficient for heat and moisture (in)
        prsl1: surface layer mean pressure (in)
        prslki: surface interface mean pressure (in)
        wet: ocean/lake mask (in)
        wind: wind speed (in)
        flag_iter: identifier for surface schemes iteration (in)
        qsurf: specific humidity at sruface (inout)
        cmm: surface momentum flux (inout)
        chh: surface heat and moisture flux (inout)
        gflux: ground heat flux, zero for ocean (inout)
        evap: evaporation from latent heat flux (inout)
        hflx: sensible heat flux (inout)
        ep: potential evaporation (inout)
    """
    flag = wet & flag_iter
    q0 = np.maximum(q1, 1.0e-8)
    rho = prsl1 / (constants.RDGAS * t1 * (1.0 + constants.ZVIR * q0))
    qss = fpvs(tskin)
    qss = constants.EPS * qss / (ps + constants.EPSM1 * qss)
    evap[flag] = 0.0
    hflx[flag] = 0.0
    ep[flag] = 0.0
    gflux[flag] = 0.0
    rch = rho * constants.CP_AIR * ch * wind
    cmm[flag] = cm[flag] * wind[flag]
    chh[flag] = rho[flag] * ch[flag] * wind[flag]
    hflx[flag] = rch[flag] * (tskin[flag] - t1[flag] * prslki[flag])
    evap[flag] = constants.ELOCP * rch[flag] * (qss[flag] - q0[flag])
    qsurf[flag] = qss[flag]
    tem = 1.0 / rho
    hflx[flag] = hflx[flag] * tem[flag] * 1.0 / constants.CP_AIR
    evap[flag] = evap[flag] * tem[flag] * 1.0 / constants.HLV
    return qsurf, cmm, chh, gflux, evap, hflx, ep
