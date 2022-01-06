import copy
import typing

import numpy as np
from gt4py.gtscript import BACKWARD, FORWARD, PARALLEL, computation, interval, sqrt, floor, exp

import fv3gfs.physics.functions.turbulence_funcs as functions
import pace.dsl.gt4py_utils as utils
import pace.util.constants as constants
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import Float, Int, FloatField, FloatFieldIJ, IntField, IntFieldIJ, BoolFieldIJ
from pace.util.grid import GridData

# @gtscript.stencil(backend=backend)
def mask_init(mask: IntField):
    with computation(FORWARD), interval(1, None):
        mask = mask[0, 0, -1] + 1

#@gtscript.stencil(backend=backend)
def init(
    zi: FloatField,
    zl: FloatField,
    zm: FloatField,
    phii: FloatField,
    phil: FloatField,
    chz: FloatField,
    ckz: FloatField,
    garea: FloatFieldIJ,
    gdx: FloatFieldIJ,
    tke: FloatField,
    q1: functions.FloatField_8,
    rdzt: FloatField,
    prn: FloatField,
    kx1: IntField,
    prsi: FloatField,
    mask: IntField,
    kinver: IntFieldIJ,
    tx1: FloatFieldIJ,
    tx2: FloatFieldIJ,
    xkzo: FloatField,
    xkzmo: FloatField,
    kpblx: IntFieldIJ,
    hpblx: FloatFieldIJ,
    pblflg: BoolFieldIJ,
    sfcflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    scuflg: BoolFieldIJ,
    zorl: FloatFieldIJ,
    dusfc: FloatFieldIJ,
    dvsfc: FloatFieldIJ,
    dtsfc: FloatFieldIJ,
    dqsfc: FloatFieldIJ,
    kpbl: IntFieldIJ,
    hpbl: FloatFieldIJ,
    rbsoil: FloatFieldIJ,
    radmin: FloatFieldIJ,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    lcld: IntFieldIJ,
    kcld: IntFieldIJ,
    theta: FloatField,
    prslk: FloatField,
    psk: FloatFieldIJ,
    t1: FloatField,
    pix: FloatField,
    qlx: FloatField,
    slx: FloatField,
    thvx: FloatField,
    qtx: FloatField,
    thlx: FloatField,
    thlvx: FloatField,
    svx: FloatField,
    thetae: FloatField,
    gotvx: FloatField,
    prsl: FloatField,
    plyr: FloatField,
    rhly: FloatField,
    qstl: FloatField,
    bf: FloatField,
    cfly: FloatField,
    crb: FloatFieldIJ,
    dtdz1: FloatField,
    evap: FloatFieldIJ,
    heat: FloatFieldIJ,
    hlw: FloatField,
    radx: FloatField,
    rbup: FloatFieldIJ,
    sflux: FloatFieldIJ,
    shr2: FloatField,
    stress: FloatFieldIJ,
    swh: FloatField,
    thermal: FloatFieldIJ,
    tsea: FloatFieldIJ,
    u10m: FloatFieldIJ,
    ustar: FloatFieldIJ,
    u1: FloatField,
    v1: FloatField,
    v10m: FloatFieldIJ,
    xmu: FloatFieldIJ,
    gravi: Float,
    dt2: Float,
    el2orc: Float,
    tkmin: Float,
    xkzm_h: Float,
    xkzm_m: Float,
    xkzm_s: Float,
    km1: Int,
    ntiw: Int,
    fv: Float,
    elocp: Float,
    g: Float,
    eps: Float,
    ntke: Int,
    ntcw: Int,
):

    with computation(FORWARD), interval(0,1):
        pcnvflg = 0
        scuflg = 1
        dusfc = 0.0
        dvsfc = 0.0
        dtsfc = 0.0
        dqsfc = 0.0
        kpbl = 1
        hpbl = 0.0
        kpblx = 1
        hpblx = 0.0
        pblflg = 1
        lcld = km1 - 1
        kcld = km1 - 1
        mrad = km1
        krad = 0
        radmin = 0.0
        sfcflg = 1
        if rbsoil[0, 0] > 0.0:
            sfcflg = 0
        gdx = sqrt(garea[0, 0])

    with computation(PARALLEL), interval(...):
        zi = phii[0, 0, 0] * gravi
        zl = phil[0, 0, 0] * gravi
        tke = max(q1[0, 0, 0][ntke], tkmin)
    with computation(PARALLEL), interval(0, -1):
        ckz = functions.CK1
        chz = functions.CH1
        prn = 1.0
        kx1 = 0.0
        zm = zi[0, 0, 1]
        rdzt = 1.0 / (zl[0, 0, 1] - zl[0, 0, 0])

        if gdx[0, 0] >= functions.XKGDX:
            xkzm_hx = xkzm_h
            xkzm_mx = xkzm_m
        else:
            xkzm_hx = 0.01 + ((xkzm_h - 0.01) * (1.0 / (functions.XKGDX - 5.0))) * (
                gdx[0, 0] - 5.0
            )
            xkzm_mx = 0.01 + ((xkzm_m - 0.01) * (1.0 / (functions.XKGDX - 5.0))) * (
                gdx[0, 0] - 5.0
            )

        if mask[0, 0, 0] < kinver[0, 0]:
            ptem = prsi[0, 0, 1] * tx1[0, 0]
            xkzo = xkzm_hx * min(
                1.0, exp(-((1.0 - ptem) * (1.0 - ptem) * 10.0))
            )

            if ptem >= xkzm_s:
                xkzmo = xkzm_mx
                kx1 = mask[0, 0, 0] + 1
            else:
                tem1 = min(
                    1.0,
                    exp(
                        -(
                            (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * 5.0
                        )
                    ),
                )
                xkzmo = xkzm_mx * tem1

        pix = psk[0, 0] / prslk[0, 0, 0]
        theta = t1[0, 0, 0] * pix[0, 0, 0]
        if (ntiw+1) > 0:
            tem = max(q1[0, 0, 0][ntcw], functions.QLMIN)
            tem1 = max(q1[0, 0, 0][ntiw], functions.QLMIN)
            ptem = functions.HVAP * tem + (functions.HVAP + functions.HFUS) * tem1
            qlx = tem + tem1
            slx = functions.CP * t1[0, 0, 0] + phil[0, 0, 0] - ptem
        else:
            qlx = max(q1[0, 0, 0][ntcw], functions.QLMIN)
            slx = functions.CP * t1[0, 0, 0] + phil[0, 0, 0] - functions.HVAP * qlx[0, 0, 0]

        tem = 1.0 + fv * max(q1[0, 0, 0][0], functions.QMIN) - qlx[0, 0, 0]
        thvx = theta[0, 0, 0] * tem
        qtx = max(q1[0, 0, 0][0], functions.QMIN) + qlx[0, 0, 0]
        thlx = theta[0, 0, 0] - pix[0, 0, 0] * elocp * qlx[0, 0, 0]
        thlvx = thlx[0, 0, 0] * (1.0 + fv * qtx[0, 0, 0])
        svx = functions.CP * t1[0, 0, 0] * tem
        thetae = theta[0, 0, 0] + elocp * pix[0, 0, 0] * max(q1[0, 0, 0][0], functions.QMIN)
        gotvx = g / (t1[0, 0, 0] * tem)

        tem = (t1[0, 0, 1] - t1[0, 0, 0]) * tem * rdzt[0, 0, 0]
        if tem > 1.0e-5:
            xkzo = min(xkzo[0, 0, 0], functions.XKZINV)
            xkzmo = min(xkzmo[0, 0, 0], functions.XKZINV)

        plyr = 0.01 * prsl[0, 0, 0]
        es = 0.01 * functions.fpvs(t1)
        qs = max(functions.QMIN, eps * es / (plyr[0, 0, 0] + (eps - 1) * es))
        rhly = max(0.0, min(1.0, max(functions.QMIN, q1[0, 0, 0][0]) / qs))
        qstl = qs

    with computation(FORWARD), interval(...):
        cfly = 0.0
        clwt = 1.0e-6 * (plyr[0, 0, 0] * 0.001)
        if qlx[0, 0, 0] > clwt:
            onemrh = max(1.0e-10, 1.0 - rhly[0, 0, 0])
            tem1 = functions.CQL / min(max((onemrh * qstl[0, 0, 0]) ** 0.49, 0.0001), 1.0)
            val = max(min(tem1 * qlx[0, 0, 0], 50.0), 0.0)
            cfly = min(max(sqrt(sqrt(rhly[0, 0, 0])) * (1.0 - exp(-val)), 0.0), 1.0)

    with computation(PARALLEL), interval(0, -2):
        tem1 = 0.5 * (t1[0, 0, 0] + t1[0, 0, 1])
        cfh = min(cfly[0, 0, 1], 0.5 * (cfly[0, 0, 0] + cfly[0, 0, 1]))
        alp = g / (0.5 * (svx[0, 0, 0] + svx[0, 0, 1]))
        gamma = el2orc * (0.5 * (qstl[0, 0, 0] + qstl[0, 0, 1])) / (tem1 ** 2)
        epsi = tem1 / elocp
        beta = (1.0 + gamma * epsi * (1.0 + fv)) / (1.0 + gamma)
        chx = cfh * alp * beta + (1.0 - cfh) * alp
        cqx = cfh * alp * functions.HVAP * (beta - epsi)
        cqx = cqx + (1.0 - cfh) * fv * g
        bf = chx * ((slx[0, 0, 1] - slx[0, 0, 0]) * rdzt[0, 0, 0]) + cqx * (
            (qtx[0, 0, 1] - qtx[0, 0, 0]) * rdzt[0, 0, 0]
        )
        radx = (zi[0, 0, 1] - zi[0, 0, 0]) * (
            swh[0, 0, 0] * xmu[0, 0] + hlw[0, 0, 0]
        )

    with computation(FORWARD):
        with interval(0, 1):
            sflux = heat[0, 0] + evap[0, 0] * fv * theta[0, 0, 0]

            if sfcflg[0, 0] == 0 or sflux[0, 0] <= 0.0:
                pblflg = 0

            if pblflg[0, 0]:
                thermal = thlvx[0, 0, 0]
                crb = functions.RBCR
            else:
                tem1 = 1e-7 * (
                    max(sqrt(u10m[0, 0] ** 2 + v10m[0, 0] ** 2), 1.0)
                    / (functions.F0 * 0.01 * zorl[0, 0])
                )
                thermal = tsea[0, 0] * (1.0 + fv * max(q1[0, 0, 0][0], functions.QMIN))
                crb = max(min(0.16 * (tem1 ** (-0.18)), functions.CRBMAX), functions.CRBMIN)

            dtdz1 = dt2 / (zi[0, 0, 1] - zi[0, 0, 0])
            ustar = sqrt(stress[0, 0])

    with computation(PARALLEL):
        with interval(0, -2):
            dw2 = (u1[0, 0, 0] - u1[0, 0, 1]) ** 2 + (v1[0, 0, 0] - v1[0, 0, 1]) ** 2
            shr2 = max(dw2, functions.DW2MIN) * rdzt[0, 0, 0] * rdzt[0, 0, 0]

    with computation(FORWARD):
        with interval(0,1):
            rbup = rbsoil[0, 0]

#@gtscript.stencil(backend=backend)
def mrf_pbl_scheme_part1(
    crb: FloatFieldIJ,
    flg: BoolFieldIJ,
    kpblx: IntFieldIJ,
    mask: IntField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    thermal: FloatFieldIJ,
    thlvx: FloatField,
    thlvx_0: FloatFieldIJ,
    u1: FloatField,
    v1: FloatField,
    zl: FloatField,
    g: Float,
):

    with computation(FORWARD):
        with interval(0, 1):
            thlvx_0 = thlvx[0, 0, 0]

            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(1, None):
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

#@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def mrf_pbl_2_thermal_1(
    crb: FloatFieldIJ,
    evap: FloatFieldIJ,
    fh: FloatFieldIJ,
    flg: BoolFieldIJ,
    fm: FloatFieldIJ,
    gotvx: FloatField,
    heat: FloatFieldIJ,
    hpbl: FloatFieldIJ,
    hpblx: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    mask: IntField,
    pblflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    phih: FloatFieldIJ,
    phim: FloatFieldIJ,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    rbsoil: FloatFieldIJ,
    sfcflg: BoolFieldIJ,
    sflux: FloatFieldIJ,
    thermal: FloatFieldIJ,
    theta: FloatField,
    ustar: FloatFieldIJ,
    vpert: FloatFieldIJ,
    zi: FloatField,
    zl: FloatField,
    zol: FloatFieldIJ,
    fv: Float,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == kpblx[0, 0]:
            if kpblx[0, 0] > 0:
                if rbdn[0, 0] >= crb[0, 0]:
                    rbint = 0.0
                elif rbup[0, 0] <= crb[0, 0]:
                    rbint = 1.0
                else:
                    rbint = (crb[0, 0] - rbdn[0, 0]) / (
                        rbup[0, 0] - rbdn[0, 0]
                    )
                hpblx = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

                if hpblx[0, 0] < zi[0, 0, 0]:
                    kpblx = kpblx[0, 0] - 1
            else:
                hpblx = zl[0, 0, 0]
                kpblx = 0

            hpbl = hpblx[0, 0]
            kpbl = kpblx[0, 0]

            if kpbl[0, 0] <= 0:
                pblflg = 0

    with computation(FORWARD), interval(0, 1):
        zol = max(rbsoil[0, 0] * fm[0, 0] * fm[0, 0] / fh[0, 0], functions.RIMIN)
        if sfcflg[0, 0]:
            zol = min(zol[0, 0], -functions.ZFMIN)
        else:
            zol = max(zol[0, 0], functions.ZFMIN)

        zol1 = zol[0, 0] * functions.SFCFRAC * hpbl[0, 0] / zl[0, 0, 0]

        if sfcflg[0, 0]:
            phih = sqrt(1.0 / (1.0 - functions.APHI16 * zol1))
            phim = sqrt(phih[0, 0])
        else:
            phim = 1.0 + functions.APHI5 * zol1
            phih = phim[0, 0]

        pcnvflg = pblflg[0, 0] and (zol[0, 0] < functions.ZOLCRU)

        wst3 = gotvx[0, 0, 0] * sflux[0, 0] * hpbl[0, 0]
        ust3 = ustar[0, 0] ** 3.0

        if pblflg[0, 0]:
            wscale = max(
                (ust3 + functions.WFAC * functions.VK * wst3 * functions.SFCFRAC) ** functions.H1, ustar[0, 0] / functions.APHI5
            )

        flg = 1

        if pcnvflg[0, 0]:
            hgamt = heat[0, 0] / wscale
            hgamq = evap[0, 0] / wscale
            vpert = max(hgamt + hgamq * fv * theta[0, 0, 0], 0.0)
            thermal = thermal[0, 0] + min(functions.CFAC * vpert[0, 0], functions.GAMCRT)
            flg = 0
            rbup = rbsoil[0, 0]

#@gtscript.stencil(backend=backend)
def thermal_2(
    crb: FloatFieldIJ,
    flg: BoolFieldIJ,
    kpbl: IntFieldIJ,
    mask: IntField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    thermal: FloatFieldIJ,
    thlvx: FloatField,
    thlvx_0: FloatFieldIJ,
    u1: FloatField,
    v1: FloatField,
    zl: FloatField,
    g: Float,
):

    with computation(FORWARD):
        with interval(1, 2):
            thlvx_0 = thlvx[0,0,-1]
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(2, None):
            if flg[0, 0] == 0:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (g * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

#@gtscript.stencil(backend=backend)
def pbl_height_enhance(
    crb: FloatFieldIJ,
    flg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    lcld: IntFieldIJ,
    mask: IntField,
    pblflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    scuflg: BoolFieldIJ,
    zi: FloatField,
    zl: FloatField,
):

    with computation(FORWARD), interval(...):
        if pcnvflg[0, 0] and kpbl[0, 0] == mask[0, 0, 0]:
            if rbdn[0, 0] >= crb[0, 0]:
                rbint = 0.0
            elif rbup[0, 0] <= crb[0, 0]:
                rbint = 1.0
            else:
                rbint = (crb[0, 0] - rbdn[0, 0]) / (rbup[0, 0] - rbdn[0, 0])

            hpbl[0, 0] = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

            if hpbl[0, 0] < zi[0, 0, 0]:
                kpbl[0, 0] = kpbl[0, 0] - 1

            if kpbl[0, 0] <= 0:
                pblflg[0, 0] = 0
                pcnvflg[0, 0] = 0

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]
            if flg[0, 0] and (zl[0, 0, 0] >= functions.ZSTBLMAX):
                lcld = mask[0, 0, 0]
                flg = 0
        with interval(1, -1):
            if flg[0, 0] and (zl[0, 0, 0] >= functions.ZSTBLMAX):
                lcld = mask[0, 0, 0]
                flg = 0

#@gtscript.stencil(backend=backend)
def stratocumulus(
    flg: BoolFieldIJ,
    kcld: IntFieldIJ,
    krad: IntFieldIJ,
    lcld: IntFieldIJ,
    mask: IntField,
    radmin: FloatFieldIJ,
    radx: FloatField,
    qlx: FloatField,
    scuflg: BoolFieldIJ,
    km1: Int,
):

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if (
                flg[0, 0]
                and (mask[0, 0, 0] <= lcld[0, 0])
                and (qlx[0, 0, 0] >= functions.QLCR)
            ):
                kcld = mask[0, 0, 0]
                flg = 0

        with interval(0, -1):
            if (
                flg[0, 0]
                and (mask[0, 0, 0] <= lcld[0, 0])
                and (qlx[0, 0, 0] >= functions.QLCR)
            ):
                kcld = mask[0, 0, 0]
                flg = 0

    with computation(FORWARD):
        with interval(0, 1):
            if scuflg[0, 0] and (kcld[0, 0] == (km1 - 1)):
                scuflg = 0
            flg = scuflg[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and (mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= functions.QLCR:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

        with interval(0, -1):
            if flg[0, 0] and (mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= functions.QLCR:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if scuflg[0, 0] and krad[0, 0] <= 0:
            scuflg = 0
        if scuflg[0, 0] and radmin[0, 0] >= 0.0:
            scuflg = 0

#@gtscript.stencil(backend=backend)
def mass_flux_comp(
    pcnvflg: BoolFieldIJ,
    q1: functions.FloatField_8,
    qcdo: functions.FloatField_8,
    qcko: functions.FloatField_8,
    scuflg: BoolFieldIJ,
    t1: FloatField,
    tcdo: FloatField,
    tcko: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    ucko: FloatField,
    v1: FloatField,
    vcdo: FloatField,
    vcko: FloatField,
):
    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            tcko = t1[0, 0, 0]
            ucko = u1[0, 0, 0]
            vcko = v1[0, 0, 0]
            for ii in range(8): 
                qcko[0, 0, 0][ii] = q1[0, 0, 0][ii]
        if scuflg[0, 0]:
            tcdo = t1[0, 0, 0]
            ucdo = u1[0, 0, 0]
            vcdo = v1[0, 0, 0]
            for i2 in range(8):
                qcdo[0, 0, 0][i2] = q1[0, 0, 0][i2]

#@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def prandtl_comp_exchg_coeff(
    chz: FloatField,
    ckz: FloatField,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    mask: IntField,
    pcnvflg: BoolFieldIJ,
    phih: FloatFieldIJ,
    phim: FloatFieldIJ,
    prn: FloatField,
    zi: FloatField,
):

    with computation(PARALLEL), interval(...):
        tem1 = max(zi[0, 0, 1] - functions.SFCFRAC * hpbl[0, 0], 0.0)
        ptem = -3.0 * (tem1 ** 2.0) / (hpbl[0, 0] ** 2.0)
        if mask[0, 0, 0] < kpbl[0, 0]:
            if pcnvflg[0, 0]:
                prn = 1.0 + ((phih[0, 0] / phim[0, 0]) - 1.0) * exp(ptem)
            else:
                prn = phih[0, 0] / phim[0, 0]

        if mask[0, 0, 0] < kpbl[0, 0]:
            prn = max(min(prn[0, 0, 0], functions.PRMAX), functions.PRMIN)
            ckz = max(min(functions.CK1 + (functions.CK0 - functions.CK1) * exp(ptem), functions.CK0), functions.CK1)
            chz = max(min(functions.CH1 + (functions.CH0 - functions.CH1) * exp(ptem), functions.CH0), functions.CH1)

#@gtscript.stencil(backend=backend)
def compute_eddy_buoy_shear(
    bf: FloatField,
    buod: FloatField,
    buou: FloatField,
    chz: FloatField,
    ckz: FloatField,
    dku: FloatField,
    dkt: FloatField,
    dkq: FloatField,
    ele: FloatField,
    elm: FloatField,
    gdx: FloatFieldIJ,
    gotvx: FloatField,
    kpbl: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    pblflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    phim: FloatFieldIJ,
    prn: FloatField,
    prod: FloatField,
    radj: FloatFieldIJ,
    rdzt: FloatField,
    rlam: FloatField,
    rle: FloatField,
    scuflg: BoolFieldIJ,
    sflux: FloatFieldIJ,
    shr2: FloatField,
    stress: FloatFieldIJ,
    tke: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    ucko: FloatField,
    ustar: FloatFieldIJ,
    v1: FloatField,
    vcdo: FloatField,
    vcko: FloatField,
    xkzo: FloatField,
    xkzmo: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
    zi: FloatField,
    zl: FloatField,
    zol: FloatFieldIJ,
):

    with computation(FORWARD):
        with interval(0, -1):
            if zol[0, 0] < 0.0:
                zk = functions.VK * zl[0, 0, 0] * (1.0 - 100.0 * zol[0, 0]) ** 0.2
            elif zol[0, 0] >= 1.0:
                zk = functions.VK * zl[0, 0, 0] / 3.7
            else:
                zk = functions.VK * zl[0, 0, 0] / (1.0 + 2.7 * zol[0, 0])

            elm = zk * rlam[0, 0, 0] / (rlam[0, 0, 0] + zk)
            dz = zi[0, 0, 1] - zi[0, 0, 0]
            tem = max(gdx[0, 0], dz)
            elm = min(elm[0, 0, 0], tem)
            ele = min(ele[0, 0, 0], tem)

        with interval(-1, None):
            elm = elm[0, 0, -1]
            ele = ele[0, 0, -1]

    with computation(PARALLEL), interval(0, -1):
        tem = (
            0.5
            * (elm[0, 0, 0] + elm[0, 0, 1])
            * sqrt(0.5 * (tke[0, 0, 0] + tke[0, 0, 1]))
        )
        ri = max(bf[0, 0, 0] / shr2[0, 0, 0], functions.RIMIN)

        if mask[0, 0, 0] < kpbl[0, 0]:
            if pblflg[0, 0]:
                dku = ckz[0, 0, 0] * tem
                dkt = dku[0, 0, 0] / prn[0, 0, 0]
            else:
                dkt = chz[0, 0, 0] * tem
                dku = dkt[0, 0, 0] * prn[0, 0, 0]
        else:
            if ri < 0.0:
                dku = functions.CK1 * tem
                dkt = functions.RCHCK * dku[0, 0, 0]
            else:
                dkt = functions.CH1 * tem
                dku = dkt[0, 0, 0] * min(1.0 + 2.1 * ri, functions.PRMAX)

        tem = ckz[0, 0, 0] * tem
        dku_tmp = max(dku[0, 0, 0], tem)
        dkt_tmp = max(dkt[0, 0, 0], tem / functions.PRSCU)

        if scuflg[0, 0]:
            if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                dku = dku_tmp
                dkt = dkt_tmp

        dkq = functions.PRTKE * dkt[0, 0, 0]

        dkt = max(min(dkt[0, 0, 0], functions.DKMAX), xkzo[0, 0, 0])

        dkq = max(min(dkq[0, 0, 0], functions.DKMAX), xkzo[0, 0, 0])

        dku = max(min(dku[0, 0, 0], functions.DKMAX), xkzmo[0, 0, 0])

    with computation(PARALLEL), interval(...):
        if mask[0, 0, 0] == krad[0, 0]:
            if scuflg[0, 0]:
                tem1 = bf[0, 0, 0] / gotvx[0, 0, 0]
                if tem1 < functions.TDZMIN:
                    tem1 = functions.TDZMIN
                ptem = radj[0, 0] / tem1
                dkt = dkt[0, 0, 0] + ptem
                dku = dku[0, 0, 0] + ptem
                dkq = dkq[0, 0, 0] + ptem

    with computation(PARALLEL):
        with interval(0, 1):
            if scuflg[0, 0] and mrad[0, 0] == 0:
                ptem = xmfd[0, 0, 0] * buod[0, 0, 0]
                ptem1 = (
                    0.5
                    * (u1[0, 0, 1] - u1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (ucdo[0, 0, 0] + ucdo[0, 0, 1] - u1[0, 0, 0] - u1[0, 0, 1])
                )
                ptem2 = (
                    0.5
                    * (v1[0, 0, 1] - v1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (vcdo[0, 0, 0] + vcdo[0, 0, 1] - v1[0, 0, 0] - v1[0, 0, 1])
                )
            else:
                ptem = 0.0
                ptem1 = 0.0
                ptem2 = 0.0

            buop = 0.5 * (
                gotvx[0, 0, 0] * sflux[0, 0] + (-dkt[0, 0, 0] * bf[0, 0, 0] + ptem)
            )

            shrp = 0.5 * (
                dku[0, 0, 0] * shr2[0, 0, 0]
                + ptem1
                + ptem2
                + (
                    stress[0, 0]
                    * ustar[0, 0]
                    * phim[0, 0]
                    / (functions.VK * zl[0, 0, 0])
                )
            )

            prod = buop + shrp

        with interval(1, -1):
            tem1_1 = (u1[0, 0, 1] - u1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2_1 = (u1[0, 0, 0] - u1[0, 0, -1]) * rdzt[0, 0, -1]
            tem1_2 = (v1[0, 0, 1] - v1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2_2 = (v1[0, 0, 0] - v1[0, 0, -1]) * rdzt[0, 0, -1]

            if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                ptem1_0 = 0.5 * (xmf[0, 0, -1] + xmf[0, 0, 0]) * buou[0, 0, 0]
                ptem1_1 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1_1 + xmf[0, 0, -1] * tem2_1)
                    * (u1[0, 0, 0] - ucko[0, 0, 0])
                )
                ptem1_2 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1_2 + xmf[0, 0, -1] * tem2_2)
                    * (v1[0, 0, 0] - vcko[0, 0, 0])
                )
            else:
                ptem1_0 = 0.0
                ptem1_1 = 0.0
                ptem1_2 = 0.0

            if scuflg[0, 0]:
                if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                    ptem2_0 = 0.5 * (xmfd[0, 0, -1] + xmfd[0, 0, 0]) * buod[0, 0, 0]
                    ptem2_1 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1_1 + xmfd[0, 0, -1] * tem2_1)
                        * (ucdo[0, 0, 0] - u1[0, 0, 0])
                    )
                    ptem2_2 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1_2 + xmfd[0, 0, -1] * tem2_2)
                        * (vcdo[0, 0, 0] - v1[0, 0, 0])
                    )
                else:
                    ptem2_0 = 0.0
                    ptem2_1 = 0.0
                    ptem2_2 = 0.0
            else:
                ptem2_0 = 0.0
                ptem2_1 = 0.0
                ptem2_2 = 0.0

            buop = (
                0.5 * ((-dkt[0, 0, -1] * bf[0, 0, -1]) + (-dkt[0, 0, 0] * bf[0, 0, 0]))
                + ptem1_0
                + ptem2_0
            )

            shrp = (
                0.5
                * ((dku[0, 0, -1] * shr2[0, 0, -1]) + (dku[0, 0, 0] * shr2[0, 0, 0]))
                + ptem1_1
                + ptem2_1
            ) + ptem1_2 + ptem2_2

            prod = buop + shrp

    with computation(PARALLEL), interval(0, -1):
        rle = functions.CE0 / ele[0, 0, 0]

#@gtscript.stencil(backend=backend)
def predict_tke(
    diss: FloatField, 
    prod: FloatField, 
    rle: FloatField, 
    tke: FloatField, 
    dtn: Float, 
    kk : Int,
):
    with computation(PARALLEL), interval(...):
        for n in range(kk):
            diss = max(
                min(
                    rle[0, 0, 0] * tke[0, 0, 0] * sqrt(tke[0, 0, 0]),
                    prod[0, 0, 0] + tke[0, 0, 0] / dtn,
                ),
                0.0,
            )
            tke = max(tke[0, 0, 0] + dtn * (prod[0, 0, 0] - diss[0, 0, 0]), functions.TKMIN)
    
#@gtscript.stencil(backend=backend)
def tke_up_down_prop(
    pcnvflg: BoolFieldIJ,
    qcdo: functions.FloatField_8,
    qcko: functions.FloatField_8,
    scuflg: BoolFieldIJ,
    tke: FloatField,
    kpbl: IntFieldIJ,
    mask: IntField,
    xlamue: FloatField,
    zl: FloatField,
    ad: FloatField,
    f1: FloatField,
    krad: IntFieldIJ,
    mrad: IntFieldIJ,
    xlamde: FloatField,
    kmpbl: Int,
    kmscu: Int,
):

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            qcko[0,0,0][7] = tke[0, 0, 0]
        if scuflg[0, 0]:
            qcdo[0,0,0][7] = tke[0, 0, 0]

    with computation(FORWARD), interval(1, None):
        if mask[0,0,0] < kmpbl:
            tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
            if pcnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                qcko[0,0,0][7] = (
                    (1.0 - tem) * qcko[0, 0, -1][7] + tem * (tke[0, 0, 0] + tke[0, 0, -1])
                ) / (1.0 + tem)

    with computation(BACKWARD), interval(...):
        if mask[0,0,0] < kmscu:
            tem = 0.5 * xlamde[0, 0, 0] * (zl[0, 0, 1] - zl[0, 0, 0])
            if (
                scuflg[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
            ):
                qcdo[0,0,0][7] = (
                    (1.0 - tem) * qcdo[0, 0, 1][7] + tem * (tke[0, 0, 0] + tke[0, 0, 1])
                ) / (1.0 + tem)

    with computation(PARALLEL), interval(0, 1):
        if mask[0,0,0] < kmscu:
            ad = 1.0
            f1 = tke[0, 0, 0]

#@gtscript.stencil(backend=backend)
def tke_tridiag_matrix_ele_comp(
    ad: FloatField,
    ad_p1: FloatFieldIJ,
    al: FloatField,
    au: FloatField,
    del_: FloatField,
    dkq: FloatField,
    f1: FloatField,
    f1_p1: FloatFieldIJ,
    kpbl: IntFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    pcnvflg: BoolFieldIJ,
    prsl: FloatField,
    qcdo: functions.FloatField_8,
    qcko: functions.FloatField_8,
    rdzt: FloatField,
    scuflg: BoolFieldIJ,
    tke: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
    dt2: Float,
):

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]
            tem2 = dsig * rdz

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7]
                    + qcko[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * tem2 * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * tem2 * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7]
                    + qcdo[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * tem2 * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * tem2 * xmfd[0, 0, 0]
        with interval(1, -1):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7]
                    + qcko[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * dsig * rdz * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * dsig * rdz * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7]
                    + qcdo[0, 0, 1][7]
                    - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * dsig * rdz * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * dsig * rdz * xmfd[0, 0, 0]

        with interval(-1, None):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]

#@gtscript.stencil(backend=backend)
def part12a(
    rtg : functions.FloatField_8,
    f1 : FloatField,
    q1 : functions.FloatField_8,
    ad : FloatField,
    f2 : functions.FloatField_8,
    dtdz1 : FloatField,
    evap : FloatFieldIJ,
    heat : FloatFieldIJ,
    t1 : FloatField,
    rdt : Float,
    ntrac1 : Int,
    ntke : Int,
):
    with computation(PARALLEL), interval(...):
        rtg[0,0,0][ntke-1] = (rtg[0,0,0][ntke-1] + (f1[0,0,0] - q1[0,0,0][ntke-1]) * rdt)

    with computation(FORWARD), interval(0,1):
        ad = 1.0
        f1 = t1[0,0,0] + dtdz1[0,0,0] * heat[0,0]
        f2[0,0,0][0] = q1[0,0,0][0] + dtdz1[0,0,0] * evap[0,0]
    
        if ntrac1 >= 2:
            for kk in range(1,ntrac1):
                f2[0,0,0][kk] = q1[0,0,0][kk]

#@gtscript.stencil(backend=backend)
def heat_moist_tridiag_mat_ele_comp(
    ad: FloatField,
    ad_p1: FloatFieldIJ,
    al: FloatField,
    au: FloatField,
    del_: FloatField,
    dkt: FloatField,
    f1: FloatField,
    f1_p1: FloatFieldIJ,
    f2: functions.FloatField_7,
    f2_p1: FloatFieldIJ,
    kpbl: IntFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    pcnvflg: BoolFieldIJ,
    prsl: FloatField,
    q1: functions.FloatField_8,
    qcdo: functions.FloatField_8,
    qcko: functions.FloatField_8,
    rdzt: FloatField,
    scuflg: BoolFieldIJ,
    tcdo: FloatField,
    tcko: FloatField,
    t1: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
    dt2: Float,
    gocp: Float,
):

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * gocp
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = qcko[0, 0, 0][0] + qcko[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = qcdo[0, 0, 0][0] + qcdo[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            f1 = f1_p1[0, 0]
            f2[0,0,0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * gocp
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = qcko[0, 0, 0][0] + qcko[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = qcdo[0, 0, 0][0] + qcdo[0, 0, 1][0] - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                f2[0,0,0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0,0,0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

#@gtscript.stencil(backend=backend)
def part13a(
    pcnvflg : BoolFieldIJ,
    mask : IntField,
    kpbl : IntFieldIJ,
    del_ : FloatField,
    prsl : FloatField,
    rdzt : FloatField,
    xmf : FloatField,
    qcko : functions.FloatField_8,
    q1 : functions.FloatField_8,
    f2 : functions.FloatField_8,
    scuflg : BoolFieldIJ,
    mrad : IntFieldIJ,
    krad : IntFieldIJ,
    xmfd : FloatField,
    qcdo : functions.FloatField_8,
    ntrac1 : Int,
    dt2 : Float,
):
    with computation(FORWARD), interval(0,-1):
        for kk in range(1, ntrac1):
            if mask[0,0,0] > 0:
                if pcnvflg[0, 0] and mask[0,0,-1] < kpbl[0,0]:
                    dtodsu = dt2 / del_[0,0,0]
                    dsig = prsl[0,0,-1] - prsl[0,0,0]
                    tem = dsig * rdzt[0,0,-1]
                    ptem = 0.5 * tem * xmf[0,0,-1]
                    ptem2 = dtodsu * ptem
                    tem1 = qcko[0,0,-1][kk] + qcko[0,0,0][kk]
                    tem2 = q1[0,0,-1][kk] + q1[0,0,0][kk]
                    f2[0,0,0][kk] = q1[0,0,0][kk] + (tem1 - tem2) * ptem2
                else:
                    f2[0,0,0][kk] = q1[0,0,0][kk]

                if scuflg[0,0] and mask[0,0,-1] >= mrad[0,0] and mask[0,0,-1] < krad[0,0]:
                    dtodsu = dt2 / del_[0,0,0]
                    dsig = prsl[0,0,-1] - prsl[0,0,0]
                    tem = dsig * rdzt[0,0,-1]
                    ptem = 0.5 * tem * xmfd[0,0,-1]
                    ptem2 = dtodsu * ptem
                    tem1 = qcdo[0,0,-1][kk] + qcdo[0,0,0][kk]
                    tem2 = q1[0,0,-1][kk] + q1[0,0,0][kk]
                    f2[0,0,0][kk] = f2[0,0,0][kk] - (tem1 - tem2) * ptem2


            if pcnvflg[0,0] and mask[0,0,0] < kpbl[0,0]:
                dtodsd = dt2 / del_[0,0,0]
                dtodsu = dt2 / del_[0,0,1]
                dsig = prsl[0,0,0] - prsl[0,0,1]
                tem = dsig * rdzt[0,0,0]
                ptem = 0.5 * tem * xmf[0,0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1 = qcko[0,0,0][kk] + qcko[0,0,1][kk]
                tem2 = q1[0,0,0][kk] + q1[0,0,1][kk]
                f2[0,0,0][kk] = f2[0,0,0][kk] - (tem1 - tem2) * ptem1

            
            if scuflg[0,0] and mask[0,0,0] >= mrad[0,0] and mask[0,0,0] < krad[0,0]:
                dtodsd = dt2 / del_[0,0,0]
                dtodsu = dt2 / del_[0,0,1]
                dsig = prsl[0,0,0] - prsl[0,0,1]
                tem = dsig * rdzt[0,0,0]
                ptem = 0.5 * tem * xmfd[0,0,0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem1 = qcdo[0,0,0][kk] + qcdo[0,0,1][kk]
                tem2 = q1[0,0,0][kk] + q1[0,0,1][kk]
                f2[0,0,0][kk] = f2[0,0,0][kk] + (tem1 - tem2) * ptem1
    
    with computation(FORWARD), interval(-1,None):
        for kk2 in range(1, ntrac1):
            if pcnvflg[0, 0] and mask[0,0,-1] < kpbl[0,0]:
                dtodsu = dt2 / del_[0,0,0]
                dsig = prsl[0,0,-1] - prsl[0,0,0]
                tem = dsig * rdzt[0,0,-1]
                ptem = 0.5 * tem * xmf[0,0,-1]
                ptem2 = dtodsu * ptem
                tem1 = qcko[0,0,-1][kk2] + qcko[0,0,0][kk2]
                tem2 = q1[0,0,-1][kk2] + q1[0,0,0][kk2]
                f2[0,0,0][kk2] = q1[0,0,0][kk2] + (tem1 - tem2) * ptem2
            else:
                f2[0,0,0][kk2] = q1[0,0,0][kk2]

#@gtscript.stencil(backend=backend)
def part13b(
    f1 : FloatField,
    t1 : FloatField,
    f2 : functions.FloatField_8,
    q1 : functions.FloatField_8,
    tdt : FloatField,
    rtg : functions.FloatField_8,
    dtsfc : FloatFieldIJ,
    del_ : FloatField,
    dqsfc : FloatFieldIJ,
    conq : Float,
    cont : Float,
    rdt : Float,
    ntrac1 : Int,
):
    with computation(PARALLEL), interval(...):
        tdt = tdt[0,0,0] + (f1[0,0,0] - t1[0,0,0]) * rdt
        rtg[0,0,0][0] = rtg[0,0,0][0] + (f2[0,0,0][0] - q1[0,0,0][0]) * rdt

        if ntrac1 >= 2:
            for kk in range(1, ntrac1):
                rtg[0,0,0][kk] = rtg[0,0,0][kk] + ((f2[0,0,0][kk] - q1[0,0,0][kk]) * rdt)

    with computation(FORWARD), interval(...):
        dtsfc = dtsfc[0,0] + cont * del_[0,0,0] * ((f1[0,0,0] - t1[0,0,0]) * rdt)
        dqsfc = dqsfc[0,0] + conq * del_[0,0,0] * ((f2[0,0,0][0] - q1[0,0,0][0]) * rdt)

#@gtscript.stencil(backend=backend)
def moment_tridiag_mat_ele_comp(
    ad: FloatField,
    ad_p1: FloatFieldIJ,
    al: FloatField,
    au: FloatField,
    del_: FloatField,
    diss: FloatField,
    dku: FloatField,
    dtdz1: FloatField,
    f1: FloatField,
    f1_p1: FloatFieldIJ,
    f2: functions.FloatField_7,
    f2_p1: FloatFieldIJ,
    kpbl: IntFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    pcnvflg: BoolFieldIJ,
    prsl: FloatField,
    rdzt: FloatField,
    scuflg: BoolFieldIJ,
    spd1: FloatFieldIJ,
    stress: FloatFieldIJ,
    tdt: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    ucko: FloatField,
    v1: FloatField,
    vcdo: FloatField,
    vcko: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
    dspheat: bool,
    dt2: Float,
):

    with computation(PARALLEL), interval(0, -1):
            if dspheat:
                tdt = tdt[0, 0, 0] + functions.DSPFAC * (diss[0, 0, 0] / functions.CP)

    with computation(PARALLEL), interval(0, 1):
            ad = 1.0 + dtdz1[0, 0, 0] * stress[0, 0] / spd1[0, 0]
            f1 = u1[0, 0, 0]
            f2[0,0,0][0] = v1[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0]- tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            f1 = f1_p1[0, 0]
            f2[0,0,0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / del_[0, 0, 0]
            dtodsu = dt2 / del_[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and mask[0, 0, 0] >= mrad[0, 0]
                and mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2

        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

#@gtscript.stencil(backend=backend)
def moment_recover(
    del_: FloatField,
    du: FloatField,
    dusfc: FloatFieldIJ,
    dv: FloatField,
    dvsfc: FloatFieldIJ,
    f1: FloatField,
    f2: functions.FloatField_7,
    hpbl: FloatFieldIJ,
    hpblx: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    mask: IntField,
    u1: FloatField,
    v1: FloatField,
    conw: Float,
    rdt: Float,
):

    with computation(FORWARD), interval(...):
        if mask[0,0,0] < 1:
            hpbl = hpblx[0, 0]
            kpbl = kpblx[0, 0]
        utend = (f1[0, 0, 0] - u1[0, 0, 0]) * rdt
        vtend = (f2[0, 0, 0][0] - v1[0, 0, 0]) * rdt
        du = du[0, 0, 0] + utend
        dv = dv[0, 0, 0] + vtend
        dusfc = dusfc[0, 0] + conw * del_[0, 0, 0] * utend
        dvsfc = dvsfc[0, 0] + conw * del_[0, 0, 0] * vtend

def mfpblt(
    im,
    ix,
    km,
    kmpbl,
    ntcw,
    ntrac1,
    delt,
    cnvflg,
    zl,
    zm,
    q1_gt,
    t1,
    u1,
    v1,
    plyr,
    pix,
    thlx,
    thvx,
    gdx,
    hpbl,
    kpbl,
    vpert,
    buo,
    xmf,
    tcko,
    qcko,
    ucko,
    vcko,
    xlamue,
    g,
    gocp,
    elocp,
    el2orc,
    mask,
    qtx,
    wu2,
    qtu,
    xlamuem,
    thlu,
    kpblx,
    kpbly,
    rbup,
    rbdn,
    flg,
    hpblx,
    xlamavg,
    sumx,
    scaldfunc,
    ce0,
    cm,
    qmin,
    qlmin,
    alp,
    pgcon,
    a1,
    b1,
    f1,
    fv,
    eps,
    epsm1,
):
    totflag = True

    for i in range(im):
        totflag = totflag and ~cnvflg[i, 0]

    if totflag:
        return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

    mfpblt_s0(
        alp=alp,
        buo=buo,
        cnvflg=cnvflg,
        g=g,
        hpbl=hpbl,
        kpbl=kpbl,
        q1=q1_gt,
        qtu=qtu,
        qtx=qtx,
        thlu=thlu,
        thlx=thlx,
        thvx=thvx,
        vpert=vpert,
        wu2=wu2,
        kpblx=kpblx,
        kpbly=kpbly,
        rbup=rbup,
        rbdn=rbdn,
        hpblx=hpblx,
        xlamavg=xlamavg,
        sumx=sumx,
        ntcw=ntcw-1,
    )

    mfpblt_s1(
        buo=buo,
        ce0=ce0,
        cm=cm,
        cnvflg=cnvflg,
        elocp=elocp,
        el2orc=el2orc,
        eps=eps,
        epsm1=epsm1,
        flg=flg,
        fv=fv,
        g=g,
        hpbl=hpbl,
        kpbl=kpbl,
        kpblx=kpblx,
        kpbly=kpbly,
        mask=mask,
        pix=pix,
        plyr=plyr,
        qtu=qtu,
        qtx=qtx,
        rbdn=rbdn,
        rbup=rbup,
        thlu=thlu,
        thlx=thlx,
        thvx=thvx,
        wu2=wu2,
        xlamue=xlamue,
        xlamuem=xlamuem,
        zl=zl,
        zm=zm,
        domain=(im, 1, kmpbl),
    )

    mfpblt_s1a(
        cnvflg=cnvflg,
        hpblx=hpblx,
        kpblx=kpblx,
        mask=mask,
        rbdn=rbdn,
        rbup=rbup,
        zm=zm,
        domain=(im, 1, km),
    )

    mfpblt_s2(
        a1=a1,
        ce0=ce0,
        cm=cm,
        cnvflg=cnvflg,
        dt2=delt,
        el2orc=el2orc,
        elocp=elocp,
        eps=eps,
        epsm1=epsm1,
        gdx=gdx,
        hpbl=hpbl,
        hpblx=hpblx,
        kpbl=kpbl,
        kpblx=kpblx,
        kpbly=kpbly,
        mask=mask,
        pgcon=pgcon,
        pix=pix,
        plyr=plyr,
        qcko=qcko,
        qtu=qtu,
        qtx=qtx,
        scaldfunc=scaldfunc,
        sumx=sumx,
        tcko=tcko,
        thlu=thlu,
        thlx=thlx,
        u1=u1,
        ucko=ucko,
        v1=v1,
        vcko=vcko,
        xlamue=xlamue,
        xlamuem=xlamuem,
        xlamavg=xlamavg,
        xmf=xmf,
        wu2=wu2,
        zl=zl,
        zm=zm,
        domain=(im, 1, kmpbl),
    )

    mfpblt_s3(
        cnvflg = cnvflg,
        kpbl = kpbl,
        mask = mask,
        xlamue = xlamue,
        qcko = qcko,
        q1_gt = q1_gt,
        zl = zl,
        ntcw = ntcw,
        ntrac1 = ntrac1,
        domain=(im, 1, kmpbl)
    )

    return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

#@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions","GreedyMerging"])
def mfpblt_s3(
    cnvflg : BoolFieldIJ,
    kpbl : IntFieldIJ,
    mask : IntField,
    xlamue : FloatField,
    qcko : functions.FloatField_8,
    q1_gt : functions.FloatField_8,
    zl : FloatField,
    ntcw  : Int,
    ntrac1 : Int,
):
    with computation(FORWARD), interval(1,None):
        if ntcw > 2:
            for n in range(1, ntcw):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0,0,0][n] = (
                        (1.0 - tem) * qcko[0,0,-1][n]
                        + tem * (q1_gt[0,0,0][n] + q1_gt[0,0,-1][n])
                    ) / factor

        ndc = ntrac1 - ntcw
        if ndc > 0:
            for n2 in range(ntcw, ntrac1):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0,0,0][n2] = (
                        (1.0 - tem) * qcko[0,0,-1][n2]
                        + tem * (q1_gt[0,0,0][n2] + q1_gt[0,0,-1][n2])
                     ) / factor

#@gtscript.stencil(backend=backend)
def mfpblt_s0(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    q1: functions.FloatField_8,
    qtu: FloatField,
    qtx: FloatField,
    thlu: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    vpert: FloatFieldIJ,
    wu2: FloatField,
    kpblx : IntFieldIJ,
    kpbly : IntFieldIJ,
    rbup : FloatFieldIJ,
    rbdn : FloatFieldIJ,
    hpblx : FloatFieldIJ,
    xlamavg : FloatFieldIJ,
    sumx : FloatFieldIJ,
    alp: Float,
    g: Float,
    ntcw: Int,
):

    with computation(PARALLEL), interval(0, -1):
        if cnvflg[0, 0]:
            buo = 0.0
            wu2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw]

    with computation(FORWARD), interval(0, 1):
        kpblx = 0
        kpbly = 0
        rbup = 0.0
        rbdn = 0.0
        hpblx = 0.0
        xlamavg = 0.0
        sumx = 0.0
        if cnvflg[0, 0]:
            ptem = min(alp * vpert[0, 0], 3.0)
            thlu = thlx[0, 0, 0] + ptem
            qtu = qtx[0, 0, 0]
            buo = g * ptem / thvx[0, 0, 0]

#@gtscript.stencil(backend=backend)
def mfpblt_s1(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    kpbly: IntFieldIJ,
    mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    qtu: FloatField,
    qtx: FloatField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    thlu: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    wu2: FloatField,
    xlamue: FloatField,
    xlamuem: FloatField,
    zl: FloatField,
    zm: FloatField,
    ce0: Float,
    cm: Float,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    fv: Float,
    g: Float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                xlamue = ce0 * (
                    1.0 / (zm[0, 0, 0] + dz)
                    + 1.0 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                )
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(1, None):
            if cnvflg[0, 0]:
                tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
                factor = 1.0 + tem
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

                tlu = thlu[0, 0, 0] / pix[0, 0, 0]
                es = 0.01 * functions.fpvs(tlu)
                qs = max(functions.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
                dq = qtu[0, 0, 0] - qs

                if dq > 0.0:
                    gamma = el2orc * qs / (tlu ** 2)
                    qlu = dq / (1.0 + gamma)
                    qtu = qs + qlu
                    thvu = (thlu[0, 0, 0] + pix[0, 0, 0] * elocp * qlu) * (
                        1.0 + fv * qs - qlu
                    )
                else:
                    thvu = thlu[0, 0, 0] * (1.0 + fv * qtu[0, 0, 0])
                buo = g * (thvu / thvx[0, 0, 0] - 1.0)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                wu2 = (4.0 * buo[0, 0, 0] * zm[0, 0, 0]) / (
                    1.0 + (0.5 * 2.0 * xlamue[0, 0, 0] * zm[0, 0, 0])
                )
        with interval(1, None):
            if cnvflg[0, 0]:
                dz = zm[0, 0, 0] - zm[0, 0, -1]
                tem = 0.25 * 2.0 * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
                wu2 = (((1.0 - tem) * wu2[0, 0, -1]) + (4.0 * buo[0, 0, 0] * dz)) / (
                    1.0 + tem
                )

    with computation(FORWARD):
        with interval(0, 1):
            flg = True
            kpbly = kpbl[0, 0]
            if cnvflg[0, 0]:
                flg = False
                rbup = wu2[0, 0, 0]

        with interval(1, None):
            if flg[0, 0] == False:
                rbdn = rbup[0, 0]
                rbup = wu2[0, 0, 0]
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] <= 0.0


#@gtscript.stencil(backend=backend)
def mfpblt_s1a(
    cnvflg: BoolFieldIJ,
    hpblx: FloatFieldIJ,
    kpblx: IntFieldIJ,
    mask: IntField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    zm: FloatField,
):

    with computation(FORWARD), interval(...):
        rbint = 0.0

        if mask[0, 0, 0] == kpblx[0, 0]:
            if cnvflg[0, 0]:
                if rbdn[0, 0] <= 0.0:
                    rbint = 0.0
                elif rbup[0, 0] >= 0.0:
                    rbint = 1.0
                else:
                    rbint = rbdn[0, 0] / (rbdn[0, 0] - rbup[0, 0])

                hpblx = zm[0, 0, -1] + rbint * (zm[0, 0, 0] - zm[0, 0, -1])


#@gtscript.stencil(backend=backend)
def mfpblt_s2(
    cnvflg: BoolFieldIJ,
    gdx: FloatFieldIJ,
    hpbl: FloatFieldIJ,
    hpblx: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    kpbly: IntFieldIJ,
    mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    qcko: functions.FloatField_8,
    qtu: FloatField,
    qtx: FloatField,
    scaldfunc: FloatFieldIJ,
    sumx: FloatFieldIJ,
    tcko: FloatField,
    thlu: FloatField,
    thlx: FloatField,
    u1: FloatField,
    ucko: FloatField,
    v1: FloatField,
    vcko: FloatField,
    xmf: FloatField,
    xlamavg: FloatFieldIJ,
    xlamue: FloatField,
    xlamuem: FloatField,
    wu2: FloatField,
    zl: FloatField,
    zm: FloatField,
    a1: Float,
    dt2: Float,
    ce0: Float,
    cm: Float,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    pgcon: Float,
):

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                if kpbl[0, 0] > kpblx[0, 0]:
                    kpbl = kpblx[0, 0]
                    hpbl = hpblx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (kpbly[0, 0] > kpblx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 1 / (zm[0, 0, 0] + dz)
                ptem1 = 1 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                xlamue = ce0 * (ptem + ptem1)
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz
        with interval(1, None):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            if wu2[0, 0, 0] > 0.0:
                xmf = a1 * sqrt(wu2[0, 0, 0])
            else:
                xmf = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem = 0.2 / xlamavg[0, 0]
                sigma = min(
                    max((3.14 * tem * tem) / (gdx[0, 0] * gdx[0, 0]), 0.001),
                    0.999,
                )

                if sigma > a1:
                    scaldfunc = max(
                        min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0
                    )
                else:
                    scaldfunc = 1.0

    with computation(PARALLEL), interval(...):
        xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            xmf = min(scaldfunc[0, 0] * xmf[0, 0, 0], xmmx)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                thlu = thlx[0, 0, 0]
        with interval(1, None):
            dz = zl[0, 0, 0] - zl[0, 0, -1]
            tem = 0.5 * xlamue[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

            tlu = thlu[0, 0, 0] / pix[0, 0, 0]
            es = 0.01 * functions.fpvs(tlu)
            qs = max(functions.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
            dq = qtu[0, 0, 0] - qs
            qlu = dq / (1.0 + (el2orc * qs / (tlu ** 2)))

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                if dq > 0.0:
                    qtu = qs + qlu
                    qcko[0,0,0][0] = qs
                    qcko[0,0,0][1] = qlu
                    tcko = tlu + elocp * qlu
                else:
                    qcko[0,0,0][0] = qtu[0, 0, 0]
                    qcko[0,0,0][1] = 0.0
                    qcko_track = 1
                    tcko = tlu

            tem = 0.5 * xlamuem[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                ucko = (
                    (1.0 - tem) * ucko[0, 0, -1]
                    + (tem + pgcon) * u1[0, 0, 0]
                    + (tem - pgcon) * u1[0, 0, -1]
                ) / factor
                vcko = (
                    (1.0 - tem) * vcko[0, 0, -1]
                    + (tem + pgcon) * v1[0, 0, 0]
                    + (tem - pgcon) * v1[0, 0, -1]
                ) / factor


def mfscu(
    im,
    ix,
    km,
    kmscu,
    ntcw,
    ntrac1,
    delt,
    cnvflg,
    zl,
    zm,
    q1,
    t1,
    u1,
    v1,
    plyr,
    pix,
    thlx,
    thvx,
    thlvx,
    gdx,
    thetae,
    radj,
    krad,
    mrad,
    radmin,
    buo,
    xmfd,
    tcdo,
    qcdo,
    ucdo,
    vcdo,
    xlamde,
    g,
    gocp,
    elocp,
    el2orc,
    mask,
    qtx,
    wd2,
    hrad,
    krad1,
    thld,
    qtd,
    thlvd,
    ra1,
    ra2,
    flg,
    xlamdem,
    mradx,
    mrady,
    sumx,
    xlamavg,
    scaldfunc,
    zm_mrad,
    ce0,
    cm,
    pgcon,
    qmin,
    qlmin,
    b1,
    f1,
    a1,
    a2,
    a11,
    a22,
    cldtime,
    actei,
    hvap,
    cp,
    eps,
    epsm1,
    fv,
):

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    mfscu_s0(
        buo=buo,
        cnvflg=cnvflg,
        flg=flg,
        hrad=hrad,
        krad=krad,
        krad1=krad1,
        mask=mask,
        mrad=mrad,
        q1=q1,
        qtd=qtd,
        qtx=qtx,
        ra1=ra1,
        ra2=ra2,
        radmin=radmin,
        radj=radj,
        thetae=thetae,
        thld=thld,
        thlvd=thlvd,
        thlvx=thlvx,
        thlx=thlx,
        thvx=thvx,
        wd2=wd2,
        zm=zm,
        a1=a1,
        a11=a11,
        a2=a2,
        a22=a22,
        actei=actei,
        cldtime=cldtime,
        cp=cp,
        hvap=hvap,
        g=g,
        ntcw=ntcw,
        domain=(im, 1, km),
    )

    mfscu_s1(
        cnvflg=cnvflg,
        flg=flg,
        krad=krad,
        mask=mask,
        mrad=mrad,
        thlvd=thlvd,
        thlvx=thlvx,
        domain=(im, 1, kmscu),
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    for i in range(im):
        zm_mrad[i,0] = zm[i, 0, mrad[i,0]-1]

    mfscu_s2(
        zl=zl,
        mask=mask,
        mrad=mrad,
        krad=krad,
        zm=zm,
        zm_mrad=zm_mrad,
        xlamde=xlamde,
        xlamdem=xlamdem,
        hrad=hrad,
        cnvflg=cnvflg,
        ce0=ce0,
        cm=cm,
        domain=(im,1,kmscu),
    )

    mfscu_s3(
        buo=buo,
        cnvflg=cnvflg,
        el2orc=el2orc,
        elocp=elocp,
        eps=eps,
        epsm1=epsm1,
        fv=fv,
        g=g,
        krad=krad,
        mask=mask,
        pix=pix,
        plyr=plyr,
        thld=thld,
        thlx=thlx,
        thvx=thvx,
        qtd=qtd,
        qtx=qtx,
        xlamde=xlamde,
        zl=zl,
        domain=(im, 1, kmscu),
    )

    bb1 = 2.0
    bb2 = 4.0

    mfscu_s4(
        buo=buo,
        cnvflg=cnvflg,
        krad1=krad1,
        mask=mask,
        wd2=wd2,
        xlamde=xlamde,
        zm=zm,
        bb1=bb1,
        bb2=bb2,
        domain=(im, 1, km),
    )

    mfscu_s5(
        buo=buo,
        cnvflg=cnvflg,
        flg=flg,
        krad=krad,
        krad1=krad1,
        mask=mask,
        mrad=mrad,
        mradx=mradx,
        mrady=mrady,
        xlamde=xlamde,
        wd2=wd2,
        zm=zm,
        domain=(im, 1, kmscu),
    )

    totflg = True

    for i in range(im):
        totflg = totflg and ~cnvflg[i, 0]

    if totflg:
        return

    for i in range(im):
        zm_mrad[i,0] = zm[i, 0, mrad[i,0]-1]

    mfscu_s6(
        zl=zl,
        mask=mask,
        mrad=mrad,
        krad=krad,
        zm=zm,
        zm_mrad=zm_mrad,
        xlamde=xlamde,
        xlamdem=xlamdem,
        hrad=hrad,
        cnvflg=cnvflg,
        mrady=mrady,
        mradx=mradx,
        ce0=ce0,
        cm=cm,
        domain=(im,1,kmscu),
    )

    mfscu_s7(
        cnvflg=cnvflg,
        dt2=delt,
        gdx=gdx,
        krad=krad,
        mask=mask,
        mrad=mrad,
        ra1=ra1,
        scaldfunc=scaldfunc,
        sumx=sumx,
        wd2=wd2,
        xlamde=xlamde,
        xlamavg=xlamavg,
        xmfd=xmfd,
        zl=zl,
        domain=(im, 1, kmscu),
    )

    mfscu_s8(
        cnvflg=cnvflg, 
        krad=krad,
        mask=mask,
        thld=thld,
        thlx=thlx,
        domain=(im, 1, km)
    )

    mfscu_s9(
        cnvflg=cnvflg,
        el2orc=el2orc,
        elocp=elocp,
        eps=eps,
        epsm1=epsm1,
        krad=krad,
        mask=mask,
        mrad=mrad,
        pgcon=pgcon,
        pix=pix,
        plyr=plyr,
        qcdo=qcdo,
        qtd=qtd,
        qtx=qtx,
        tcdo=tcdo,
        thld=thld,
        thlx=thlx,
        u1=u1,
        ucdo=ucdo,
        v1=v1,
        vcdo=vcdo,
        xlamde=xlamde,
        xlamdem=xlamdem,
        zl=zl,
        ntcw=ntcw,
        domain=(im, 1, kmscu),
    )

    mfscu_10(
        cnvflg = cnvflg,
        krad = krad,
        mrad = mrad,
        mask = mask,
        zl = zl,
        xlamde = xlamde,
        qcdo = qcdo,
        q1 = q1,
        ntcw = ntcw,
        ntrac1 = ntrac1,
        domain = (im, 1, kmscu),
    )

    return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde

#@gtscript.stencil(backend=backend)
def mfscu_s2(
    zl : FloatField,
    mask : IntField,
    mrad : IntFieldIJ,
    krad : IntFieldIJ,
    zm : FloatField,
    zm_mrad : FloatFieldIJ,
    xlamde : FloatField,
    xlamdem : FloatField,
    hrad : FloatFieldIJ,
    cnvflg: BoolFieldIJ,
    ce0 : Float,
    cm : Float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0,0]:
            dz = zl[0,0,1] - zl[0,0,0]
            if mask[0,0,0] >= mrad[0,0] and mask[0,0,0] < krad[0,0]:
                if mrad[0,0] == 0:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] - zm_mrad[0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0,0,0]

#@gtscript.stencil(backend=backend)
def mfscu_s6(
    zl : FloatField,
    mask : IntField,
    mrad : IntFieldIJ,
    krad : IntFieldIJ,
    zm : FloatField,
    zm_mrad : FloatFieldIJ,
    xlamde : FloatField,
    xlamdem : FloatField,
    hrad : FloatFieldIJ,
    cnvflg: BoolFieldIJ,
    mrady : IntFieldIJ,
    mradx : IntFieldIJ,
    ce0 : Float,
    cm : Float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0,0] and (mrady[0,0] < mradx[0,0]):
            dz = zl[0,0,1] - zl[0,0,0]
            if mask[0,0,0] >= mrad[0,0] and mask[0,0,0] < krad[0,0]:
                if mrad[0,0] == 0:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0/(zm[0,0,0] - zm_mrad[0,0] + dz)) + 1.0 / max(hrad[0,0] - zm[0,0,0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0,0,0]

#@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions","GreedyMerging"])
def mfscu_10(
    cnvflg : BoolFieldIJ,
    krad   : IntFieldIJ,
    mrad   : IntFieldIJ,
    mask   : IntField,
    zl     : FloatField,
    xlamde : FloatField,
    qcdo   : functions.FloatField_8,
    q1     : functions.FloatField_8,
    ntcw   : Int,
    ntrac1 : Int,
):
    with computation(BACKWARD), interval(...):
        if ntcw > 2:
            for n in range(1, ntcw-1):
                if cnvflg[0,0] and mask[0,0,0] < krad[0,0] and mask[0,0,0] >= mrad[0,0]:
                    dz = zl[0,0,1] - zl[0,0,0]
                    tem = 0.5 * xlamde[0,0,0] * dz
                    factor = 1.0 + tem
                    qcdo[0,0,0][n] = (
                        (1.0 - tem) * qcdo[0,0,1][n]
                        + tem * (q1[0,0,0][n] + q1[0,0,1][n])
                    ) / factor
            
        ndc = ntrac1 - ntcw
        if ndc > 0:
            for n1 in range(ntcw, ntrac1):
                if cnvflg[0,0] and mask[0,0,0] < krad[0,0] and mask[0,0,0] >= mrad[0,0]:
                    dz = zl[0,0,1] - zl[0,0,0]
                    tem = 0.5 * xlamde[0,0,0] * dz
                    factor = 1.0 + tem
                    qcdo[0,0,0][n1] = (
                        (1.0 - tem) * qcdo[0,0,1][n1]
                        + tem * (q1[0,0,0][n1] + q1[0,0,1][n1])
                    ) / factor

#@gtscript.stencil(backend=backend)
def mfscu_s0(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    hrad: FloatFieldIJ,
    krad: IntFieldIJ,
    krad1: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    q1: functions.FloatField_8,
    qtd: FloatField,
    qtx: FloatField,
    ra1: FloatFieldIJ,
    ra2: FloatFieldIJ,
    radmin: FloatFieldIJ,
    radj: FloatFieldIJ,
    thetae: FloatField,
    thld: FloatField,
    thlvd: FloatFieldIJ,
    thlvx: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    wd2: FloatField,
    zm: FloatField,
    a1: Float,
    a11: Float,
    a2: Float,
    a22: Float,
    actei: Float,
    cldtime: Float,
    cp: Float,
    hvap: Float,
    g: Float,
    ntcw: Int,
):

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw-1]

    with computation(FORWARD), interval(...):
        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                hrad = zm[0, 0, 0]
                krad1 = mask[0, 0, 0] - 1
                tem1 = max(
                    cldtime * radmin[0, 0] / (zm[0, 0, 1] - zm[0, 0, 0]), -3.0
                )
                thld = thlx[0, 0, 0] + tem1
                qtd = qtx[0, 0, 0]
                thlvd = thlvx[0, 0, 0] + tem1
                buo = -g * tem1 / thvx[0, 0, 0]

                ra1 = a1
                ra2 = a11

                tem = thetae[0, 0, 0] - thetae[0, 0, 1]
                tem1 = qtx[0, 0, 0] - qtx[0, 0, 1]
                if (tem > 0.0) and (tem1 > 0.0):
                    cteit = cp * tem / (hvap * tem1)
                    if cteit > actei:
                        ra1 = a2
                        ra2 = a22

                radj = -ra2[0, 0] * radmin[0, 0]

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrad = krad[0, 0]


#@gtscript.stencil(backend=backend)
def mfscu_s1(
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    thlvd: FloatFieldIJ,
    thlvx: FloatField,
):

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0
        with interval(0, -1):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0

    with computation(FORWARD), interval(0, 1):
        kk = krad[0, 0] - mrad[0, 0]
        if cnvflg[0, 0]:
            if kk < 1:
                cnvflg[0, 0] = 0


#@gtscript.stencil(backend=backend)
def mfscu_s3(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    thld: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    qtd: FloatField,
    qtx: FloatField,
    xlamde: FloatField,
    zl: FloatField,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    fv: Float,
    g: Float,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        tem = 0.5 * xlamde[0, 0, 0] * dz
        factor = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * functions.fpvs(tld)
        qs = max(functions.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                tem1 = 1.0 + fv * qs - qld
                thvd = (thld[0, 0, 0] + pix[0, 0, 0] * elocp * qld) * tem1
            else:
                tem1 = 1.0 + fv * qtd[0, 0, 0]
                thvd = thld[0, 0, 0] * tem1
            buo = g * (1.0 - thvd / thvx[0, 0, 0])


#@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def mfscu_s4(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    krad1: IntFieldIJ,
    mask: IntField,
    wd2: FloatField,
    xlamde: FloatField,
    zm: FloatField,
    bb1: Float,
    bb2: Float,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == krad1[0, 0]:
            if cnvflg[0, 0]:
                dz = zm[0, 0, 1] - zm[0, 0, 0]
                wd2 = (bb2 * buo[0, 0, 1] * dz) / (
                    1.0 + (0.5 * bb1 * xlamde[0, 0, 0] * dz)
                )


#@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions"])
def mfscu_s5(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    krad: IntFieldIJ,
    krad1: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    mradx: IntFieldIJ,
    mrady: IntFieldIJ,
    xlamde: FloatField,
    wd2: FloatField,
    zm: FloatField,
):
    with computation(BACKWARD), interval(...):
        dz = zm[0, 0, 1] - zm[0, 0, 0]
        tem = 0.25 * 2.0 * (xlamde[0, 0, 0] + xlamde[0, 0, 1]) * dz
        ptem1 = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad1[0, 0]:
            wd2 = (((1.0 - tem) * wd2[0, 0, 1]) + (4.0 * buo[0, 0, 1] * dz)) / ptem1

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrady = mrad[0, 0]
        if flg[0, 0]:
            mradx = krad[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0
        with interval(0, -1):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            if mrad[0, 0] < mradx[0, 0]:
                mrad = mradx[0, 0]
            if (krad[0, 0] - mrad[0, 0]) < 1:
                cnvflg = 0


#@gtscript.stencil(backend=backend)
def mfscu_s7(
    cnvflg: BoolFieldIJ,
    gdx: FloatFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    ra1: FloatFieldIJ,
    scaldfunc: FloatFieldIJ,
    sumx: FloatFieldIJ,
    wd2: FloatField,
    xlamde: FloatField,
    xlamavg: FloatFieldIJ,
    xmfd: FloatField,
    zl: FloatField,
    dt2: Float,
):
    with computation(FORWARD), interval(0, 1):
        xlamavg = 0.0
        sumx = 0.0

    with computation(BACKWARD), interval(-1, None):
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            if wd2[0, 0, 0] > 0:
                xmfd = ra1[0, 0] * sqrt(wd2[0, 0, 0])
            else:
                xmfd = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem1 = (3.14 * (0.2 / xlamavg[0, 0]) * (0.2 / xlamavg[0, 0])) / (
                    gdx[0, 0] * gdx[0, 0]
                )
                sigma = min(max(tem1, 0.001), 0.999)

            if cnvflg[0, 0]:
                if sigma > ra1[0, 0]:
                    scaldfunc = max(
                        min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0
                    )
                else:
                    scaldfunc = 1.0

    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
            xmfd = min(scaldfunc[0, 0] * xmfd[0, 0, 0], xmmx)


#@gtscript.stencil(backend=backend)
def mfscu_s8(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    thld: FloatField,
    thlx: FloatField,
):

    with computation(PARALLEL), interval(...):
        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                thld = thlx[0, 0, 0]


#@gtscript.stencil(backend=backend)
def mfscu_s9(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    pix: FloatField,
    plyr: FloatField,
    qcdo: functions.FloatField_8,
    qtd: FloatField,
    qtx: FloatField,
    tcdo: FloatField,
    thld: FloatField,
    thlx: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    v1: FloatField,
    vcdo: FloatField,
    xlamde: FloatField,
    xlamdem: FloatField,
    zl: FloatField,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    pgcon: Float,
    ntcw: Int,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            tem = 0.5 * xlamde[0, 0, 0] * dz
            factor = 1.0 + tem
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * functions.fpvs(tld)
        qs = max(functions.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)

        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
        ):
            if dq > 0.0:
                qtd = qs + qld
                qcdo[0,0,0][0] = qs
                qcdo[0,0,0][ntcw-1] = qld
                tcdo = tld + elocp * qld
            else:
                qcdo[0,0,0] = qtd[0, 0, 0]
                qcdo[0,0,0][ntcw-1] = 0.0
                tcdo = tld

        if (
            cnvflg[0, 0]
            and mask[0, 0, 0] < krad[0, 0]
            and mask[0, 0, 0] >= mrad[0, 0]
        ):
            tem = 0.5 * xlamdem[0, 0, 0] * dz
            factor = 1.0 + tem
            ptem = tem - pgcon
            ptem1 = tem + pgcon
            ucdo = (
                (1.0 - tem) * ucdo[0, 0, 1] + ptem * u1[0, 0, 1] + ptem1 * u1[0, 0, 0]
            ) / factor
            vcdo = (
                (1.0 - tem) * vcdo[0, 0, 1] + ptem * v1[0, 0, 1] + ptem1 * v1[0, 0, 0]
            ) / factor


#@gtscript.stencil(backend=backend)
def tridit(
    au: FloatField,
    cm: FloatField,
    cl: FloatField,
    f1: FloatField,
):
    with computation(FORWARD):
        with interval(0, 1):
            fk = 1.0 / cm[0, 0, 0]
            au = fk * au[0, 0, 0]
            f1 = fk * f1[0, 0, 0]
        with interval(1, -1):
            fkk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fkk * au[0, 0, 0]
            f1 = fkk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])

    with computation(BACKWARD):
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            f1 = fk * (f1[0, 0, 0] - cl[0, 0, -1] * f1[0, 0, -1])
        with interval(0, -1):
            f1 = f1[0, 0, 0] - au[0, 0, 0] * f1[0, 0, 1]


#@gtscript.stencil(backend=backend,skip_passes=["graph_merge_horizontal_executions","GreedyMerging"])
def tridin(
    cl : FloatField, 
    cm : FloatField,
    cu : FloatField, 
    r1 : FloatField, 
    r2 : functions.FloatField_7, 
    au : FloatField, 
    a1 : FloatField, 
    a2 : functions.FloatField_7,
    nt : Int,
):
    with computation(FORWARD):
        with interval(0,1):
            fk = 1.0 / cm[0,0,0]
            au = fk * cu[0,0,0]
            a1 = fk * r1[0,0,0]
            for n0 in range(nt):
                a2[0,0,0][n0] = fk * r2[0,0,0][n0]

        with interval(1,-1):
            fkk = 1.0 / (cm[0,0,0] - cl[0,0,-1] * au[0,0,-1])
            au = fkk * cu[0,0,0]
            a1 = fkk * (r1[0,0,0] - cl[0,0,-1] * a1[0,0,-1])

            for n1 in range(nt):
                a2[0,0,0][n1] = fkk * (r2[0,0,0][n1] - cl[0,0,-1] * a2[0,0,-1][n1])

        with interval(-1,None):
            fk = 1.0 / (cm[0,0,0] - cl[0,0,-1] * au[0,0,-1])
            a1 = fk * (r1[0,0,0] - cl[0,0,-1] * a1[0,0,-1])

            for n2 in range(nt):
                a2[0,0,0][n2] = fk * (r2[0,0,0][n2] - cl[0,0,-1] * a2[0,0,-1][n2])

    with computation(BACKWARD):
        with interval(0,-1):
            a1 = a1[0,0,0] - au[0,0,0] * a1[0,0,1]
            for n3 in range(nt):
                a2[0,0,0][n3] = a2[0,0,0][n3] - au[0,0,0] * a2[0,0,1][n3]

#@gtscript.stencil(backend=backend)
def tridi2(
    a1: FloatField,
    a2: functions.FloatField_7,
    au: FloatField,
    cl: FloatField,
    cm: FloatField,
    cu: FloatField,
    r1: FloatField,
    r2: functions.FloatField_7,
):

    with computation(PARALLEL), interval(0, 1):
        fk = 1 / cm[0, 0, 0]
        au = fk * cu[0, 0, 0]
        a1 = fk * r1[0, 0, 0]
        a2[0, 0, 0][0] = fk * r2[0, 0, 0][0]

    with computation(FORWARD):
        with interval(1, -1):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            au = fk * cu[0, 0, 0]
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])
        with interval(-1, None):
            fk = 1.0 / (cm[0, 0, 0] - cl[0, 0, -1] * au[0, 0, -1])
            a1 = fk * (r1[0, 0, 0] - cl[0, 0, -1] * a1[0, 0, -1])
            a2[0, 0, 0][0] = fk * (r2[0, 0, 0][0] - cl[0, 0, -1] * a2[0, 0, -1][0])

    with computation(BACKWARD), interval(0, -1):
        a1 = a1[0, 0, 0] - au[0, 0, 0] * a1[0, 0, 1]
        a2[0, 0, 0][0] = a2[0, 0, 0][0] - au[0, 0, 0] * a2[0, 0, 1][0]

#@gtscript.stencil(backend=backend)
def comp_asym_mix_up(
    mask : IntField,
    mlenflg : BoolFieldIJ,
    bsum : FloatFieldIJ,
    zlup : FloatFieldIJ,
    thvx_k : FloatFieldIJ,
    tke_k : FloatFieldIJ,
    thvx : FloatField,
    tke : FloatField,
    gotvx : FloatField,
    zl : FloatField,
    zfmin : Float,
    k : Int,
):  
    with computation(FORWARD), interval(...):
        if mask[0,0,0] == k:
            mlenflg = True
            zlup = 0.0
            bsum = 0.0
            thvx_k = thvx[0,0,0]
            tke_k = tke[0,0,0]
        if mlenflg[0,0] == True:
            dz = zl[0,0,1] - zl[0,0,0]
            ptem = gotvx[0,0,0] * (thvx[0,0,1] - thvx_k[0,0]) * dz
            bsum = bsum[0,0] + ptem
            zlup = zlup[0,0] + dz
            if bsum[0,0] >= tke_k[0,0]:
                if ptem >= 0.0:
                    tem2 = max(ptem, zfmin)
                else:
                    tem2 = min(ptem, -zfmin)
                ptem1 = (bsum[0,0] - tke_k[0,0]) / tem2
                zlup = zlup[0,0] - ptem1 * dz
                zlup = max(zlup[0,0], 0.0)
                mlenflg = False

#@gtscript.stencil(backend=backend)
def comp_asym_mix_dn(
    mask : IntField,
    mlenflg : BoolFieldIJ,
    bsum : FloatFieldIJ,
    zldn : FloatFieldIJ,
    thvx_k : FloatFieldIJ,
    tke_k : FloatFieldIJ,
    thvx : FloatField,
    tke : FloatField,
    gotvx : FloatField,
    zl : FloatField,
    tsea : FloatFieldIJ,
    q1_gt : functions.FloatField_8,
    zfmin : Float,
    fv : Float,
    k : Int,
):
    with computation(BACKWARD), interval(...):
        if mask[0,0,0] == k:
            mlenflg = True
            bsum = 0.0
            zldn = 0.0
            thvx_k = thvx[0, 0 ,0]
            tke_k = tke[0, 0, 0]
        if mlenflg[0,0] == True:
            if mask[0,0,0] == 0:
                dz = zl[0,0,0]
                tem1 = tsea[0,0] * (1.0 + fv * max(q1_gt[0,0,0][0], functions.QMIN))
            else:
                dz = zl[0,0,0] - zl[0,0,-1]
                tem1 = thvx[0,0,-1]
            ptem = gotvx[0,0,0] * (thvx_k[0,0] - tem1) * dz
            bsum = bsum[0,0] + ptem
            zldn = zldn[0,0] + dz
            if bsum[0,0] >= tke_k[0,0]:
                if ptem >= 0.0:
                    tem2 = max(ptem,zfmin)
                else:
                    tem2 = min(ptem, -zfmin)
                ptem1 = (bsum[0,0] - tke_k[0,0]) / tem2
                zldn = zldn[0,0] - ptem1 * dz
                zldn = max(zldn[0,0], 0.0)
                mlenflg[0,0] = False

#@gtscript.stencil(backend=backend)
def comp_asym_rlam_ele(
    zi : FloatField,
    rlam : FloatField,
    ele : FloatField,
    zlup : FloatFieldIJ,
    zldn : FloatFieldIJ,
    rlmn : Float,
    rlmx : Float,
    elmfac : Float,
    elmx : Float,
):
    with computation(FORWARD), interval(...):
        tem = 0.5 * (zi[0,0,1] - zi[0,0,0])
        tem1 = min(tem, rlmn)

        ptem2 = min(zlup[0,0], zldn[0,0])
        rlam = min(max(elmfac*ptem2,tem1),rlmx)

        ptem2 = sqrt(zlup[0,0] * zldn[0,0])
        ele = min(max(functions.ELEFAC*ptem2,tem1),elmx)