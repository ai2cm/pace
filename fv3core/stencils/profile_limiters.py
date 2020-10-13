import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


def grid():
    return spec.grid


@gtstencil()
def posdef_constraint_iv0(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, r12: float):
    with computation(PARALLEL), interval(...):
        a32 = a4_3 - a4_2
        absa32 = abs(a32)
        if a4_1 <= 0.0:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if absa32 < -a4_4:
                if (a4_1 + 0.25 * (a4_3 - a4_2) ** 2 / a4_4 + a4_4 * r12) < 0.0:
                    if (a4_1 < a4_3) and (a4_1 < a4_2):
                        a4_3 = a4_1
                        a4_2 = a4_1
                        a4_4 = 0.0
                    elif a4_3 > a4_2:
                        a4_4 = 3.0 * (a4_2 - a4_1)
                        a4_3 = a4_2 - a4_4
                    else:
                        a4_4 = 3.0 * (a4_3 - a4_1)
                        a4_2 = a4_3 - a4_4


@gtstencil()
def posdef_constraint_iv1(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd):
    with computation(PARALLEL), interval(...):
        da1 = a4_3 - a4_2
        da2 = da1 ** 2
        a6da = a4_4 * da1
        if ((a4_1 - a4_2) * (a4_1 - a4_3)) >= 0.0:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if a6da < -1.0 * da2:
                a4_4 = 3.0 * (a4_2 - a4_1)
                a4_3 = a4_2 - a4_4
            elif a6da > da2:
                a4_4 = 3.0 * (a4_3 - a4_1)
                a4_2 = a4_3 - a4_4


@gtstencil()
def ppm_constraint(a4_1: sd, a4_2: sd, a4_3: sd, a4_4: sd, extm: sd):
    with computation(PARALLEL), interval(...):
        da1 = a4_3 - a4_2
        da2 = da1 ** 2
        a6da = a4_4 * da1
        if extm == 1:
            a4_2 = a4_1
            a4_3 = a4_1
            a4_4 = 0.0
        else:
            if a6da < -da2:
                a4_4 = 3.0 * (a4_2 - a4_1)
                a4_3 = a4_2 - a4_4
            elif a6da > da2:
                a4_4 = 3.0 * (a4_3 - a4_1)
                a4_2 = a4_3 - a4_4


def compute(a4_1, a4_2, a4_3, a4_4, extm, iv, i1, i_extent, kstart, nk, js, j_extent):
    r12 = 1.0 / 12.0
    if iv == 0:
        posdef_constraint_iv0(
            a4_1,
            a4_2,
            a4_3,
            a4_4,
            r12,
            origin=(i1, js, kstart),
            domain=(i_extent, j_extent, nk),
        )
    elif iv == 1:
        posdef_constraint_iv1(
            a4_1,
            a4_2,
            a4_3,
            a4_4,
            origin=(i1, js, kstart),
            domain=(i_extent, j_extent, nk),
        )
    else:
        ppm_constraint(
            a4_1,
            a4_2,
            a4_3,
            a4_4,
            extm,
            origin=(i1, js, kstart),
            domain=(i_extent, j_extent, nk),
        )
    return a4_1, a4_2, a4_3, a4_4
