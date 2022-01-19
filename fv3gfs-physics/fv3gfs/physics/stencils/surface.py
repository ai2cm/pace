import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, computation, exp, floor, interval, log, sqrt

import pace.util.constants as constants
from pace.dsl.typing import FloatField, FloatFieldIJ


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


@gtscript.function
def stability(
    z1,
    snwdph,
    thv1,
    wind,
    z0max,
    ztmax,
    tvs,
    rb,
    fm,
    fh,
    fm10,
    fh2,
    cm,
    ch,
    stress,
    ustar,
):
    alpha = 5.0
    a0 = -3.975
    a1 = 12.32
    alpha4 = 4.0 * alpha
    b1 = -7.755
    b2 = 6.041
    alpha2 = alpha + alpha
    beta = 1.0
    a0p = -7.941
    a1p = 24.75
    b1p = -8.705
    b2p = 7.899
    ztmin1 = -999.0

    z1i = 1.0 / z1

    tem1 = z0max / z1
    if abs(1.0 - tem1) > 1.0e-6:
        ztmax1 = -beta * log(tem1) / (alpha2 * (1.0 - tem1))
    else:
        ztmax1 = 99.0

    if z0max < 0.05 and snwdph < 10.0:
        ztmax1 = 99.0

    dtv = thv1 - tvs
    adtv = max(abs(dtv), 0.001)
    if dtv >= 0.0:
        dtv = abs(adtv)
    else:
        dtv = -abs(adtv)
    rb = max(
        -5000.0,
        (constants.GRAV + constants.GRAV) * dtv * z1 / ((thv1 + tvs) * wind * wind),
    )
    tem1 = 1.0 / z0max
    tem2 = 1.0 / ztmax
    fm = log((z0max + z1) * tem1)
    fh = log((ztmax + z1) * tem2)
    fm10 = log((z0max + 10.0) * tem1)
    fh2 = log((ztmax + 2.0) * tem2)
    hlinf = rb * fm * fm / fh
    hlinf = min(max(hlinf, ztmin1), ztmax1)

    if dtv >= 0.0:
        hl1 = hlinf
        if hlinf > 0.25:
            tem1 = hlinf * z1i
            hl0inf = z0max * tem1
            hltinf = ztmax * tem1
            aa = sqrt(1.0 + alpha4 * hlinf)
            aa0 = sqrt(1.0 + alpha4 * hl0inf)
            bb = aa
            bb0 = sqrt(1.0 + alpha4 * hltinf)
            pm = aa0 - aa + log((aa + 1.0) / (aa0 + 1.0))
            ph = bb0 - bb + log((bb + 1.0) / (bb0 + 1.0))
            fms = fm - pm
            fhs = fh - ph
            hl1 = fms * fms * rb / fhs
            hl1 = min(max(hl1, ztmin1), ztmax1)

        tem1 = hl1 * z1i
        hl0 = z0max * tem1
        hlt = ztmax * tem1
        aa = sqrt(1.0 + alpha4 * hl1)
        aa0 = sqrt(1.0 + alpha4 * hl0)
        bb = aa
        bb0 = sqrt(1.0 + alpha4 * hlt)
        pm = aa0 - aa + log((1.0 + aa) / (1.0 + aa0))
        ph = bb0 - bb + log((1.0 + bb) / (1.0 + bb0))
        hl110 = hl1 * 10.0 * z1i
        hl110 = min(max(hl110, ztmin1), ztmax1)
        aa = sqrt(1.0 + alpha4 * hl110)
        pm10 = aa0 - aa + log((1.0 + aa) / (1.0 + aa0))
        hl12 = (hl1 + hl1) * z1i
        hl12 = min(max(hl12, ztmin1), ztmax1)
        bb = sqrt(1.0 + alpha4 * hl12)
        ph2 = bb0 - bb + log((1.0 + bb) / (1.0 + bb0))

    else:
        olinf = z1 / hlinf
        tem1 = 50.0 * z0max
        if abs(olinf) <= tem1:
            hlinf = -z1 / tem1
            hlinf = min(max(hlinf, ztmin1), ztmax1)
        if hlinf >= -0.5:
            hl1 = hlinf
            pm = (a0 + a1 * hl1) * hl1 / (1.0 + (b1 + b2 * hl1) * hl1)
            ph = (a0p + a1p * hl1) * hl1 / (1.0 + (b1p + b2p * hl1) * hl1)
            hl110 = hl1 * 10.0 * z1i
            hl110 = min(max(hl110, ztmin1), ztmax1)
            pm10 = (a0 + a1 * hl110) * hl110 / (1.0 + (b1 + b2 * hl110) * hl110)
            hl12 = (hl1 + hl1) * z1i
            hl12 = min(max(hl12, ztmin1), ztmax1)
            ph2 = (a0p + a1p * hl12) * hl12 / (1.0 + (b1p + b2p * hl12) * hl12)
        else:  # hlinf < 0.05
            hl1 = -hlinf
            tem1 = 1.0 / sqrt(hl1)
            pm = log(hl1) + 2.0 * sqrt(tem1) - 0.8776
            ph = log(hl1) + 0.5 * tem1 + 1.386
            hl110 = hl1 * 10.0 * z1i
            hl110 = min(max(hl110, ztmin1), ztmax1)
            pm10 = log(hl110) + 2.0 / sqrt(sqrt(hl110)) - 0.8776
            hl12 = (hl1 + hl1) * z1i
            hl12 = min(max(hl12, ztmin1), ztmax1)
            ph2 = log(hl12) + 0.5 / sqrt(hl12) + 1.386

    fm = fm - pm
    fh = fh - ph
    fm10 = fm10 - pm10
    fh2 = fh2 - ph2
    cm = constants.CA * constants.CA / (fm * fm)
    ch = constants.CA * constants.CA / (fm * fh)
    tem1 = 0.00001 / z1
    cm = max(cm, tem1)
    ch = max(ch, tem1)
    stress = cm * wind * wind
    ustar = sqrt(stress)

    return rb, fm, fh, fm10, fh2, cm, ch, stress, ustar


@gtscript.function
def znot_m_v6(uref):
    p13 = -1.296521881682694e-02
    p12 = 2.855780863283819e-01
    p11 = -1.597898515251717e00
    p10 = -8.396975715683501e00
    p25 = 3.790846746036765e-10
    p24 = 3.281964357650687e-09
    p23 = 1.962282433562894e-07
    p22 = -1.240239171056262e-06
    p21 = 1.739759082358234e-07
    p20 = 2.147264020369413e-05
    p35 = 1.840430200185075e-07
    p34 = -2.793849676757154e-05
    p33 = 1.735308193700643e-03
    p32 = -6.139315534216305e-02
    p31 = 1.255457892775006e00
    p30 = -1.663993561652530e01
    p40 = 4.579369142033410e-04

    if uref >= 0.0 and uref <= 6.5:
        znotm = exp(p10 + uref * (p11 + uref * (p12 + uref * p13)))
    elif uref > 6.5 and uref <= 15.7:
        znotm = p20 + uref * (
            p21 + uref * (p22 + uref * (p23 + uref * (p24 + uref * p25)))
        )
    elif uref > 15.7 and uref <= 53.0:
        znotm = exp(
            p30 + uref * (p31 + uref * (p32 + uref * (p33 + uref * (p34 + uref * p35))))
        )
    elif uref > 53.0:
        znotm = p40
    # else:
    #     print('Wrong input uref value:',uref)

    return znotm


@gtscript.function
def znot_t_v6(uref):
    p00 = 1.100000000000000e-04
    p15 = -9.144581627678278e-10
    p14 = 7.020346616456421e-08
    p13 = -2.155602086883837e-06
    p12 = 3.333848806567684e-05
    p11 = -2.628501274963990e-04
    p10 = 8.634221567969181e-04
    p25 = -8.654513012535990e-12
    p24 = 1.232380050058077e-09
    p23 = -6.837922749505057e-08
    p22 = 1.871407733439947e-06
    p21 = -2.552246987137160e-05
    p20 = 1.428968311457630e-04
    p35 = 3.207515102100162e-12
    p34 = -2.945761895342535e-10
    p33 = 8.788972147364181e-09
    p32 = -3.814457439412957e-08
    p31 = -2.448983648874671e-06
    p30 = 3.436721779020359e-05
    p45 = -3.530687797132211e-11
    p44 = 3.939867958963747e-09
    p43 = -1.227668406985956e-08
    p42 = -1.367469811838390e-05
    p41 = 5.988240863928883e-04
    p40 = -7.746288511324971e-03
    p56 = -1.187982453329086e-13
    p55 = 4.801984186231693e-11
    p54 = -8.049200462388188e-09
    p53 = 7.169872601310186e-07
    p52 = -3.581694433758150e-05
    p51 = 9.503919224192534e-04
    p50 = -1.036679430885215e-02
    p60 = 4.751256171799112e-05

    if uref >= 0.0 and uref < 5.9:
        znott = p00
    elif uref >= 5.9 and uref <= 15.4:
        znott = p10 + uref * (
            p11 + uref * (p12 + uref * (p13 + uref * (p14 + uref * p15)))
        )
    elif uref > 15.4 and uref <= 21.6:
        znott = p20 + uref * (
            p21 + uref * (p22 + uref * (p23 + uref * (p24 + uref * p25)))
        )
    elif uref > 21.6 and uref <= 42.2:
        znott = p30 + uref * (
            p31 + uref * (p32 + uref * (p33 + uref * (p34 + uref * p35)))
        )
    elif uref > 42.2 and uref <= 53.3:
        znott = p40 + uref * (
            p41 + uref * (p42 + uref * (p43 + uref * (p44 + uref * p45)))
        )
    elif uref > 53.3 and uref <= 80.0:
        znott = p50 + uref * (
            p51 + uref * (p52 + uref * (p53 + uref * (p54 + uref * (p55 + uref * p56))))
        )
    elif uref > 80.0:
        znott = p60
    # else:
    #     print("Wrong input uref value", uref)

    return znott


@gtscript.function
def znot_m_v7(uref):
    p13 = (-1.296521881682694e-02,)
    p12 = 2.855780863283819e-01
    p11 = -1.597898515251717e00
    p10 = -8.396975715683501e00
    p25 = 3.790846746036765e-10
    p24 = 3.281964357650687e-09
    p23 = 1.962282433562894e-07
    p22 = -1.240239171056262e-06
    p21 = 1.739759082358234e-07
    p20 = 2.147264020369413e-05
    p35 = 1.897534489606422e-07
    p34 = -3.019495980684978e-05
    p33 = 1.931392924987349e-03
    p32 = -6.797293095862357e-02
    p31 = 1.346757797103756e00
    p30 = -1.707846930193362e01
    p40 = 3.371427455376717e-04

    if uref >= 0.0 and uref <= 6.5:
        znotm = exp(p10 + uref * (p11 + uref * (p12 + uref * p13)))
    elif uref > 6.5 and uref <= 15.7:
        znotm = p20 + uref * (
            p21 + uref * (p22 + uref * (p23 + uref * (p24 + uref * p25)))
        )
    elif uref > 15.7 and uref <= 53.0:
        znotm = exp(
            p30 + uref * (p31 + uref * (p32 + uref * (p33 + uref * (p34 + uref * p35))))
        )
    elif uref > 53.0:
        znotm = p40
    # else:
    #     print('Wrong input uref value:',uref)

    return znotm


@gtscript.function
def znot_t_v7(uref):
    p00 = 1.100000000000000e-04
    p15 = -9.193764479895316e-10
    p14 = 7.052217518653943e-08
    p13 = -2.163419217747114e-06
    p12 = 3.342963077911962e-05
    p11 = -2.633566691328004e-04
    p10 = 8.644979973037803e-04
    p25 = -9.402722450219142e-12
    p24 = 1.325396583616614e-09
    p23 = -7.299148051141852e-08
    p22 = 1.982901461144764e-06
    p21 = -2.680293455916390e-05
    p20 = 1.484341646128200e-04
    p35 = 7.921446674311864e-12
    p34 = -1.019028029546602e-09
    p33 = 5.251986927351103e-08
    p32 = -1.337841892062716e-06
    p31 = 1.659454106237737e-05
    p30 = -7.558911792344770e-05
    p45 = -2.694370426850801e-10
    p44 = 5.817362913967911e-08
    p43 = -5.000813324746342e-06
    p42 = 2.143803523428029e-04
    p41 = -4.588070983722060e-03
    p40 = 3.924356617245624e-02
    p56 = -1.663918773476178e-13
    p55 = 6.724854483077447e-11
    p54 = -1.127030176632823e-08
    p53 = 1.003683177025925e-06
    p52 = -5.012618091180904e-05
    p51 = 1.329762020689302e-03
    p50 = -1.450062148367566e-02
    p60 = 6.840803042788488e-05

    if uref >= 0.0 and uref < 5.9:
        znott = p00
    elif uref >= 5.9 and uref <= 15.4:
        znott = p10 + uref * (
            p11 + uref * (p12 + uref * (p13 + uref * (p14 + uref * p15)))
        )
    elif uref > 15.4 and uref <= 21.6:
        znott = p20 + uref * (
            p21 + uref * (p22 + uref * (p23 + uref * (p24 + uref * p25)))
        )
    elif uref > 21.6 and uref <= 42.6:
        znott = p30 + uref * (
            p31 + uref * (p32 + uref * (p33 + uref * (p34 + uref * p35)))
        )
    elif uref > 42.6 and uref <= 53.0:
        znott = p40 + uref * (
            p41 + uref * (p42 + uref * (p43 + uref * (p44 + uref * p45)))
        )
    elif uref > 53.0 and uref <= 80.0:
        znott = p50 + uref * (
            p51 + uref * (p52 + uref * (p53 + uref * (p54 + uref * (p55 + uref * p56))))
        )
    elif uref > 80.0:
        znott = p60
    # else:
    #        print('Wrong input uref value:',uref)

    return znott


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


def sfc_diff(
    t1: FloatFieldIJ,
    q1: FloatFieldIJ,
    z1: FloatFieldIJ,
    wind: FloatFieldIJ,
    #  prsl1,
    prslki: FloatFieldIJ,
    sigmaf: FloatFieldIJ,
    vegtype: FloatFieldIJ,
    shdmax: FloatFieldIJ,
    ivegsrc: int,
    z0pert: FloatFieldIJ,
    ztpert: FloatFieldIJ,
    flag_iter: FloatFieldIJ,
    redrag: bool,
    u10m: FloatFieldIJ,
    v10m: FloatFieldIJ,
    sfc_z0_type: int,
    wet: FloatFieldIJ,
    dry: FloatFieldIJ,
    icy: FloatFieldIJ,
    tskin: FloatField,
    tsurf: FloatField,
    snwdph: FloatField,
    z0rl: FloatField,
    ustar: FloatField,
    cm: FloatField,
    ch: FloatField,
    rb: FloatField,
    stress: FloatField,
    fm: FloatField,
    fh: FloatField,
    fm10: FloatField,
    fh2: FloatField,
):

    with computation(FORWARD), interval(0, 1):

        if flag_iter[0, 0]:
            virtfac = 1.0 + constants.RVRDM1 * max(q1[0, 0], 1.0e-8)
            thv1 = t1[0, 0] * prslki[0, 0] * virtfac

            if dry[0, 0]:
                tvs = 0.5 * (tsurf[0, 0, 0] + tskin[0, 0, 0]) * virtfac
                z0max = max(1.0e-6, min(0.01 * z0rl[0, 0, 0], z1[0, 0]))

                tem1 = 1.0 - shdmax[0, 0]
                tem2 = tem1 * tem1
                tem1 = 1.0 - tem2

                if ivegsrc == 1:
                    if vegtype[0, 0] == 10:
                        z0max = exp(tem2 * log(0.01) + tem1 * log(0.07))
                    elif vegtype[0, 0] == 6:
                        z0max = exp(tem2 * log(0.01) + tem1 * log(0.05))
                    elif vegtype[0, 0] == 7:
                        z0max = 0.01
                    elif vegtype[0, 0] == 16:
                        z0max = 0.01
                    else:
                        z0max = exp(tem2 * log(0.01) + tem1 * log(z0max))
                elif ivegsrc == 2:
                    if vegtype[0, 0] == 7:
                        z0max = exp(tem2 * log(0.01) + tem1 * log(0.07))
                    elif vegtype[0, 0] == 8:
                        z0max = exp(tem2 * log(0.01) + tem1 * log(0.05))
                    elif vegtype[0, 0] == 9:
                        z0max = 0.01
                    elif vegtype[0, 0] == 11:
                        z0max = 0.01
                    else:
                        z0max = exp(tem2 * log(0.01) + tem1 + log(z0max))

                if z0pert[0, 0] != 0.0:
                    z0max = z0max * (10.0 ** z0pert[0, 0])

                z0max = max(z0max, 1.0e-6)

                czilc = 0.8

                tem1 = 1.0 - sigmaf[0, 0]
                ztmax = z0max * exp(
                    -tem1
                    * tem1
                    * czilc
                    * constants.CA
                    * sqrt(ustar[0, 0, 0] * (0.01 / 1.5e-5))
                )

                if ztpert[0, 0] != 0.0:
                    ztmax = ztmax * (10.0 ** ztpert[0, 0])

                ztmax = max(ztmax, 1.0e-6)

                (
                    rb[0, 0, 0],
                    fm[0, 0, 0],
                    fh[0, 0, 0],
                    fm10[0, 0, 0],
                    fh2[0, 0, 0],
                    cm[0, 0, 0],
                    ch[0, 0, 0],
                    stress[0, 0, 0],
                    ustar[0, 0, 0],
                ) = stability(
                    z1[0, 0],
                    snwdph[0, 0, 0],
                    thv1,
                    wind[0, 0],
                    z0max,
                    ztmax,
                    tvs,
                    rb[0, 0, 0],
                    fm[0, 0, 0],
                    fh[0, 0, 0],
                    fm10[0, 0, 0],
                    fh2[0, 0, 0],
                    cm[0, 0, 0],
                    ch[0, 0, 0],
                    stress[0, 0, 0],
                    ustar[0, 0, 0],
                )

    with computation(FORWARD), interval(1, 2):
        if flag_iter[0, 0]:
            virtfac = 1.0 + constants.RVRDM1 * max(q1[0, 0], 1.0e-8)
            thv1 = t1[0, 0] * prslki[0, 0] * virtfac
            if icy[0, 0]:
                tvs = 0.5 * (tsurf[0, 0, 0] + tskin[0, 0, 0]) * virtfac
                z0max = max(1.0e-6, min(0.01 * z0rl[0, 0, 0], z1[0, 0]))

                tem1 = 1.0 - shdmax[0, 0]
                tem2 = tem1 * tem1
                tem1 = 1.0 - tem2

                if ivegsrc == 1:
                    z0max = exp(tem2 * log(0.01) + tem1 * log(z0max))
                elif ivegsrc == 2:
                    z0max = exp(tem2 * log(0.01) + tem1 * log(z0max))

                z0max = max(z0max, 1.0e-6)

                czilc = 0.8

                tem1 = 1.0 - sigmaf[0, 0]
                ztmax = z0max * exp(
                    -tem1
                    * tem1
                    * czilc
                    * constants.CA
                    * sqrt(ustar[0, 0, 0] * (0.01 / 1.5e-5))
                )
                ztmax = max(ztmax, 1.0e-6)

                (
                    rb[0, 0, 0],
                    fm[0, 0, 0],
                    fh[0, 0, 0],
                    fm10[0, 0, 0],
                    fh2[0, 0, 0],
                    cm[0, 0, 0],
                    ch[0, 0, 0],
                    stress[0, 0, 0],
                    ustar[0, 0, 0],
                ) = stability(
                    z1[0, 0],
                    snwdph[0, 0, 0],
                    thv1,
                    wind[0, 0],
                    z0max,
                    ztmax,
                    tvs,
                    rb[0, 0, 0],
                    fm[0, 0, 0],
                    fh[0, 0, 0],
                    fm10[0, 0, 0],
                    fh2[0, 0, 0],
                    cm[0, 0, 0],
                    ch[0, 0, 0],
                    stress[0, 0, 0],
                    ustar[0, 0, 0],
                )

    with computation(FORWARD), interval(2, 3):
        if flag_iter[0, 0]:
            virtfac = 1.0 + constants.RVRDM1 * max(q1[0, 0], 1.0e-8)
            thv1 = t1[0, 0] * prslki[0, 0] * virtfac
            if wet[0, 0]:
                tvs = 0.5 * (tsurf[0, 0, 0] + tskin[0, 0, 0]) * virtfac
                z0 = 0.01 * z0rl[0, 0, 0]
                z0max = max(1.0e-6, min(z0, z1[0, 0]))
                ustar[0, 0, 0] = sqrt(constants.GRAV * z0 / constants.CHARNOCK)
                wind10m = sqrt(u10m[0, 0] * u10m[0, 0] + v10m[0, 0] * v10m[0, 0])

                restar = max(ustar[0, 0, 0] * z0max * constants.VISI, 0.000001)

                rat = min(7.0, 2.67 * sqrt(sqrt(restar)) - 2.57)
                ztmax = max(z0max * exp(-rat), 1.0e-6)

                if sfc_z0_type == 6:
                    ztmax = znot_t_v6(wind10m)
                elif sfc_z0_type == 7:
                    ztmax = znot_t_v7(wind10m)
                # elif sfc_z0_type != 0:
                # print("No option for zfc_zo_type=", sfc_z0_type)
                # exit(1)

                (
                    rb[0, 0, 0],
                    fm[0, 0, 0],
                    fh[0, 0, 0],
                    fm10[0, 0, 0],
                    fh2[0, 0, 0],
                    cm[0, 0, 0],
                    ch[0, 0, 0],
                    stress[0, 0, 0],
                    ustar[0, 0, 0],
                ) = stability(
                    z1[0, 0],
                    snwdph[0, 0, 0],
                    thv1,
                    wind[0, 0],
                    z0max,
                    ztmax,
                    tvs,
                    rb[0, 0, 0],
                    fm[0, 0, 0],
                    fh[0, 0, 0],
                    fm10[0, 0, 0],
                    fh2[0, 0, 0],
                    cm[0, 0, 0],
                    ch[0, 0, 0],
                    stress[0, 0, 0],
                    ustar[0, 0, 0],
                )

                if sfc_z0_type == 0:
                    z0 = (
                        (constants.CHARNOCK / constants.GRAV)
                        * ustar[0, 0, 0]
                        * ustar[0, 0, 0]
                    )

                    if redrag:
                        z0rl[0, 0, 0] = 100.0 * max(min(z0, constants.Z0S_MAX), 1.0e-7)
                    else:
                        z0rl[0, 0, 0] = 100.0 * max(min(z0, 0.1), 1.0e-7)

                elif sfc_z0_type == 6:
                    z0 = znot_m_v6(wind10m)
                    z0rl[0, 0, 0] = 100.0 * z0

                elif sfc_z0_type == 7:
                    z0 = znot_m_v7(wind10m)
                    z0rl[0, 0, 0] = 100.0 * z0

                else:
                    z0rl[0, 0, 0] = 1.0e-4
