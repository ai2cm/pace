import numpy as np

import pace.util.constants as constants


def fpvsx(t):
    tliq = constants.TTP
    tice = constants.TTP - 20.0
    dldtl = constants.CVAP - constants.C_LIQ
    heatl = constants.HLV
    xponal = -dldtl / constants.RVGAS
    xponbl = -dldtl / constants.RVGAS + heatl / (constants.RVGAS * constants.TTP)
    dldti = constants.CVAP - constants.CSOL
    heati = constants.HLV + constants.HFUS
    xponai = -dldti / constants.RVGAS
    xponbi = -dldti / constants.RVGAS + heati / (constants.RVGAS * constants.TTP)

    tr = constants.TTP / t

    fpvsx = np.zeros_like(tr)
    ind1 = t > tliq
    fpvsx[ind1] = (
        constants.PSAT * tr[ind1] ** xponal * np.exp(xponbl * (1.0 - tr[ind1]))
    )
    ind2 = t < tice
    fpvsx[ind2] = (
        constants.PSAT * tr[ind2] ** xponai * np.exp(xponbi * (1.0 - tr[ind2]))
    )
    ind3 = ~np.logical_or(ind1, ind2)
    w = (t - tice) / (tliq - tice)
    pvl = constants.PSAT * (tr ** xponal) * np.exp(xponbl * (1.0 - tr))
    pvi = constants.PSAT * (tr ** xponai) * np.exp(xponbi * (1.0 - tr))
    fpvsx[ind3] = w[ind3] * pvl[ind3] + (1.0 - w[ind3]) * pvi[ind3]

    return fpvsx


def fpvs(t):
    # gpvs function variables
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1.0 / xinc
    c1xpvs = 1.0 - (xmin * c2xpvs)

    xj = np.minimum(np.maximum(c1xpvs + c2xpvs * t, 1.0), nxpvs)
    jx = np.minimum(xj, nxpvs - 1.0)
    jx = np.floor(jx)

    # Convert jx to "x"
    x = xmin + (jx * xinc)
    xm = xmin + ((jx - 1) * xinc)

    fpvs = fpvsx(xm) + (xj - jx) * (fpvsx(x) - fpvsx(xm))

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
