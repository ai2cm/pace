import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, computation, exp, floor, interval

import pace.util.constants as constants
from pace.dsl.typing import FloatFieldIJ


@gtscript.function
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

    fpvsx = 0.0
    if t > tliq:
        fpvsx = constants.PSAT * tr ** xponal * exp(xponbl * (1.0 - tr))
    elif t < tice:
        fpvsx = constants.PSAT * tr ** xponai * exp(xponbi * (1.0 - tr))
    else:
        w = (t - tice) / (tliq - tice)
        pvl = constants.PSAT * (tr ** xponal) * exp(xponbl * (1.0 - tr))
        pvi = constants.PSAT * (tr ** xponai) * exp(xponbi * (1.0 - tr))
        fpvsx = w * pvl + (1.0 - w) * pvi

    return fpvsx


@gtscript.function
def fpvs(t):
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1.0 / xinc
    c1xpvs = 1.0 - (xmin * c2xpvs)

    xj = min(max(c1xpvs + c2xpvs * t, 1.0), nxpvs)
    jx = min(xj, nxpvs - 1.0)
    jx = floor(jx)

    x = xmin + (jx * xinc)
    xm = xmin + ((jx - 1) * xinc)

    fpvs = fpvsx(xm) + (xj - jx) * (fpvsx(x) - fpvsx(xm))

    return fpvs


def sfc_ocean(
    ps: FloatFieldIJ,
    t1: FloatFieldIJ,
    q1: FloatFieldIJ,
    tskin: FloatFieldIJ,
    cm: FloatFieldIJ,
    ch: FloatFieldIJ,
    prsl1: FloatFieldIJ,
    prslki: FloatFieldIJ,
    wet: FloatFieldIJ,
    wind: FloatFieldIJ,
    flag_iter: FloatFieldIJ,
    qsurf: FloatFieldIJ,
    cmm: FloatFieldIJ,
    chh: FloatFieldIJ,
    gflux: FloatFieldIJ,
    evap: FloatFieldIJ,
    hflx: FloatFieldIJ,
    ep: FloatFieldIJ,
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
    with computation(FORWARD), interval(...):
        flag = wet and flag_iter
        q0 = max(q1, 1.0e-8)
        rho = prsl1 / (constants.RDGAS * t1 * (1.0 + constants.ZVIR * q0))
        qss = fpvs(tskin)
        qss = constants.EPS * qss / (ps + constants.EPSM1 * qss)
        if flag:
            evap = 0.0
            hflx = 0.0
            ep = 0.0
            gflux = 0.0
            rch = rho * constants.CP_AIR * ch * wind
            cmm = cm * wind
            chh = rho * ch * wind
            hflx = rch * (tskin - t1 * prslki)
            evap = constants.ELOCP * rch * (qss - q0)
            qsurf = qss
            tem = 1.0 / rho
            hflx = hflx * tem * 1.0 / constants.CP_AIR
            evap = evap * tem * 1.0 / constants.HLV
