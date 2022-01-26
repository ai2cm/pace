from gt4py import gtscript
from gt4py.gtscript import exp, log, sqrt, floor

import pace.util.constants as constants
import pace.dsl.typing as typing

FloatField_8 = typing.Field[gtscript.IJK, (typing.Float,(8,))]
FloatField_7 = typing.Field[gtscript.IJK, (typing.Float,(7,))]

# Turbulence Constants

# RD = 2.87050e2    This is under pace-util/constants.py as RDGAS
# CP = 1.00460e3    This is under pace-util/constants.py as CP_AIR
# RV = 4.61500e2    This is under pace-util/constants.py as RVGAS
# HVAP = 2.50000e6  This is under pace-util/constants.py as HLV
# HFUS = 3.33580e5  This is under pace-util/constants.py as HLF
WFAC = 7.0
CFAC = 4.5
GAMCRT = 3.0
SFCFRAC = 0.1
VK = 0.4
RIMIN = -100.0
RBCR = 0.25
ZOLCRU = -0.02
TDZMIN = 1.0e-3
RLMN = 30.0
RLMX = 500.0
ELMX = 500.0
PRMIN = 0.25
PRMAX = 4.0
PRTKE = 1.0
PRSCU = 0.67
F0 = 1.0e-4
CRBMIN = 0.15
CRBMAX = 0.35
TKMIN = 1.0e-9
DSPFAC = 0.5
QMIN = 1.0e-8
QLMIN = 1.0e-12
ZFMIN = 1.0e-8
APHI5 = 5.0
APHI16 = 16.0
ELMFAC = 1.0
ELEFAC = 1.0
CQL = 100.0
DW2MIN = 1.0e-4
DKMAX = 1000.0
XKGDX = 25000.0
QLCR = 3.5e-5
ZSTBLMAX = 2500.0
XKZINV = 0.15
H1 = 0.33333333
CK0 = 0.4
CK1 = 0.15
CH0 = 0.4
CH1 = 0.15
CE0 = 0.4
RCHCK = 1.5
CDTN = 25.0
XMIN = 180.0
XMAX = 330.0

CON_TTP = 2.7316e2          # Maybe constants.TFREEZE?
CON_CVAP = constants.CP_VAP # 1.8460e3
CON_CLIQ = constants.C_LIQ  # 4.1855e3
CON_HVAP = 2.5000e6
CON_RV = constants.RVGAS    # 4.6150e2
CON_CSOL = 2.1060e3
CON_HFUS = constants.HLF    # 3.3358e5
CON_PSAT = 6.1078e2

@gtscript.function
def fpvsx(t):
    tliq = CON_TTP
    tice = CON_TTP - 20.0
    dldtl = CON_CVAP - CON_CLIQ
    heatl = CON_HVAP
    xponal = -dldtl / CON_RV
    xponbl = -dldtl / CON_RV + heatl / (CON_RV * CON_TTP)
    dldti = CON_CVAP - CON_CSOL
    heati = CON_HVAP + CON_HFUS
    xponai = -dldti / CON_RV
    xponbi = -dldti / CON_RV + heati / (CON_RV * CON_TTP)

    tr = CON_TTP / t

    fpvsx = 0.0
    if t > tliq:
        fpvsx = CON_PSAT * tr ** xponal * exp(xponbl * (1.0 - tr))
    elif t < tice:
        fpvsx = CON_PSAT * tr ** xponai * exp(xponbi * (1.0 - tr))
    else:
        w = (t - tice) / (tliq - tice)
        pvl = CON_PSAT * (tr ** xponal) * exp(xponbl * (1.0 - tr))
        pvi = CON_PSAT * (tr ** xponai) * exp(xponbi * (1.0 - tr))
        fpvsx = w * pvl + (1.0 - w) * pvi

    return fpvsx

@gtscript.function
def fpvs(t):
    # gpvs function variables
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1.0 / xinc
    c1xpvs = 1.0 - (xmin * c2xpvs)

    xj = min(max(c1xpvs + c2xpvs * t[0, 0, 0], 1.0), nxpvs)
    jx = min(xj, nxpvs - 1.0)
    jx = floor(jx)

    # Convert jx to "x"
    x = xmin + (jx * xinc)
    xm = xmin + ((jx - 1) * xinc)

    fpvs = fpvsx(xm) + (xj - jx) * (fpvsx(x) - fpvsx(xm))

    return fpvs

