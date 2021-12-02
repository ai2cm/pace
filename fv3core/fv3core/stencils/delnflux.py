from typing import Optional

import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import (
    __INLINED,
    FORWARD,
    PARALLEL,
    computation,
    horizontal,
    interval,
    region,
)

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import get_stencils_with_varied_bounds
from fv3core.utils.grid import DampingCoefficients, axis_offsets
from fv3core.utils.stencil import StencilFactory
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK
from fv3gfs.util import X_DIM, Y_DIM, Z_DIM
from fv3gfs.util.constants import X_INTERFACE_DIM, Y_INTERFACE_DIM


def calc_damp(damp4: FloatField, nord: FloatFieldK, damp_c: FloatFieldK, da_min: float):
    with computation(FORWARD), interval(...):
        damp4 = (damp_c * da_min) ** (nord + 1)


def fx_calc_stencil_nord(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)


def fy_calc_stencil_nord(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)


def fx_calc_stencil_column(q: FloatField, del6_v: FloatFieldIJ, fx: FloatField):
    from __externals__ import nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            fx = fx_calculation_neg(q, del6_v)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            fx = fx_calculation_neg(q, del6_v)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            fx = fx_calculation_neg(q, del6_v)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            fx = fx_calculation_neg(q, del6_v)


def fy_calc_stencil_column(q: FloatField, del6_u: FloatFieldIJ, fy: FloatField):
    from __externals__ import nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            fy = fy_calculation_neg(q, del6_u)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            fy = fy_calculation_neg(q, del6_u)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            fy = fy_calculation_neg(q, del6_u)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            fy = fy_calculation_neg(q, del6_u)


@gtscript.function
def fx_calculation(q: FloatField, del6_v: FloatField):
    return del6_v * (q[-1, 0, 0] - q)


@gtscript.function
def fx_calculation_neg(q: FloatField, del6_v: FloatField):
    return -del6_v * (q[-1, 0, 0] - q)


@gtscript.function
def fy_calculation(q: FloatField, del6_u: FloatField):
    return del6_u * (q[0, -1, 0] - q)


@gtscript.function
def fy_calculation_neg(q: FloatField, del6_u: FloatField):
    return -del6_u * (q[0, -1, 0] - q)


def d2_highorder_stencil(
    fx: FloatField, fy: FloatField, rarea: FloatFieldIJ, d2: FloatField
):
    from __externals__ import nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            d2 = d2_highorder(fx, fy, rarea)


@gtscript.function
def d2_highorder(fx: FloatField, fy: FloatField, rarea: FloatField):
    d2 = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea
    return d2


def d2_damp_interval(q: FloatField, d2: FloatField, damp: FloatFieldK):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2 = damp * q
        else:
            d2 = damp * q


def copy_stencil_interval(q_in: FloatField, q_out: FloatField):
    from __externals__ import (
        local_ie,
        local_is,
        local_je,
        local_js,
        nord0,
        nord1,
        nord2,
        nord3,
    )

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 == 0):
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in


def add_diffusive_component(
    fx: FloatField, fx2: FloatField, fy: FloatField, fy2: FloatField
):
    with computation(PARALLEL), interval(...):
        fx = fx + fx2
        fy = fy + fy2


def diffusive_damp(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mass: FloatField,
    damp: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        fx = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        fy = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


def copy_corners_y_nord(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q_out = q_in[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q_out = q_in[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q_out = q_in[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q_out = q_in[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q_out = q_in[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q_out = q_in[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q_out = q_in[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q_out = q_in[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q_out = q_in[1, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-3, 2, 0]
            with horizontal(
                region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-4, 1, 0]
            with horizontal(
                region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[-5, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-2, 1, 0]
            with horizontal(
                region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[-3, 0, 0]
            with horizontal(
                region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[-4, -1, 0]
            with horizontal(
                region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[-1, 0, 0]
            with horizontal(
                region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[-2, -1, 0]
            with horizontal(
                region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[-3, -2, 0]
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q_out = q_in[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q_out = q_in[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q_out = q_in[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q_out = q_in[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q_out = q_in[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q_out = q_in[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q_out = q_in[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q_out = q_in[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q_out = q_in[1, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-3, 2, 0]
            with horizontal(
                region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-4, 1, 0]
            with horizontal(
                region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[-5, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-2, 1, 0]
            with horizontal(
                region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[-3, 0, 0]
            with horizontal(
                region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[-4, -1, 0]
            with horizontal(
                region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[-1, 0, 0]
            with horizontal(
                region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[-2, -1, 0]
            with horizontal(
                region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[-3, -2, 0]

    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q_out = q_in[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q_out = q_in[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q_out = q_in[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q_out = q_in[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q_out = q_in[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q_out = q_in[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q_out = q_in[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q_out = q_in[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q_out = q_in[1, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-3, 2, 0]
            with horizontal(
                region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-4, 1, 0]
            with horizontal(
                region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[-5, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-2, 1, 0]
            with horizontal(
                region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[-3, 0, 0]
            with horizontal(
                region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[-4, -1, 0]
            with horizontal(
                region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[-1, 0, 0]
            with horizontal(
                region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[-2, -1, 0]
            with horizontal(
                region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[-3, -2, 0]
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
            ):
                q_out = q_in[5, 0, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
            ):
                q_out = q_in[4, 1, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
            ):
                q_out = q_in[3, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
            ):
                q_out = q_in[4, -1, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
            ):
                q_out = q_in[3, 0, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
            ):
                q_out = q_in[2, 1, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
            ):
                q_out = q_in[3, -2, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
            ):
                q_out = q_in[2, -1, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
            ):
                q_out = q_in[1, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-3, 2, 0]
            with horizontal(
                region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-4, 1, 0]
            with horizontal(
                region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[-5, 0, 0]
            with horizontal(
                region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-2, 1, 0]
            with horizontal(
                region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[-3, 0, 0]
            with horizontal(
                region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[-4, -1, 0]
            with horizontal(
                region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[-1, 0, 0]
            with horizontal(
                region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[-2, -1, 0]
            with horizontal(
                region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[-3, -2, 0]


def copy_corners_x_nord(q_in: FloatField, q_out: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start, nord0, nord1, nord2, nord3

    with computation(PARALLEL), interval(0, 1):
        if __INLINED(nord0 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q_out = q_in[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q_out = q_in[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q_out = q_in[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q_out = q_in[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q_out = q_in[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q_out = q_in[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q_out = q_in[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q_out = q_in[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q_out = q_in[0, 1, 0]
            with horizontal(
                region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[2, -3, 0]
            with horizontal(
                region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[1, -2, 0]
            with horizontal(
                region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[0, -1, 0]
            with horizontal(
                region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[1, -4, 0]
            with horizontal(
                region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[0, -3, 0]
            with horizontal(
                region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-1, -2, 0]
            with horizontal(
                region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[0, -5, 0]
            with horizontal(
                region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-1, -4, 0]
            with horizontal(
                region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-2, -3, 0]
    with computation(PARALLEL), interval(1, 2):
        if __INLINED(nord1 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q_out = q_in[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q_out = q_in[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q_out = q_in[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q_out = q_in[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q_out = q_in[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q_out = q_in[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q_out = q_in[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q_out = q_in[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q_out = q_in[0, 1, 0]
            with horizontal(
                region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[2, -3, 0]
            with horizontal(
                region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[1, -2, 0]
            with horizontal(
                region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[0, -1, 0]
            with horizontal(
                region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[1, -4, 0]
            with horizontal(
                region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[0, -3, 0]
            with horizontal(
                region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-1, -2, 0]
            with horizontal(
                region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[0, -5, 0]
            with horizontal(
                region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-1, -4, 0]
            with horizontal(
                region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-2, -3, 0]

    with computation(PARALLEL), interval(2, 3):
        if __INLINED(nord2 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q_out = q_in[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q_out = q_in[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q_out = q_in[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q_out = q_in[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q_out = q_in[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q_out = q_in[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q_out = q_in[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q_out = q_in[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q_out = q_in[0, 1, 0]
            with horizontal(
                region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[2, -3, 0]
            with horizontal(
                region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[1, -2, 0]
            with horizontal(
                region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[0, -1, 0]
            with horizontal(
                region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[1, -4, 0]
            with horizontal(
                region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[0, -3, 0]
            with horizontal(
                region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-1, -2, 0]
            with horizontal(
                region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[0, -5, 0]
            with horizontal(
                region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-1, -4, 0]
            with horizontal(
                region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-2, -3, 0]
    with computation(PARALLEL), interval(3, None):
        if __INLINED(nord3 > 0):
            with horizontal(
                region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
            ):
                q_out = q_in[0, 5, 0]
            with horizontal(
                region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
            ):
                q_out = q_in[-1, 4, 0]
            with horizontal(
                region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
            ):
                q_out = q_in[-2, 3, 0]
            with horizontal(
                region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
            ):
                q_out = q_in[1, 4, 0]
            with horizontal(
                region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
            ):
                q_out = q_in[0, 3, 0]
            with horizontal(
                region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
            ):
                q_out = q_in[-1, 2, 0]
            with horizontal(
                region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
            ):
                q_out = q_in[2, 3, 0]
            with horizontal(
                region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
            ):
                q_out = q_in[1, 2, 0]
            with horizontal(
                region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
            ):
                q_out = q_in[0, 1, 0]
            with horizontal(
                region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]
            ):
                q_out = q_in[2, -3, 0]
            with horizontal(
                region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]
            ):
                q_out = q_in[1, -2, 0]
            with horizontal(
                region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]
            ):
                q_out = q_in[0, -1, 0]
            with horizontal(
                region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]
            ):
                q_out = q_in[1, -4, 0]
            with horizontal(
                region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]
            ):
                q_out = q_in[0, -3, 0]
            with horizontal(
                region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]
            ):
                q_out = q_in[-1, -2, 0]
            with horizontal(
                region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]
            ):
                q_out = q_in[0, -5, 0]
            with horizontal(
                region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]
            ):
                q_out = q_in[-1, -4, 0]
            with horizontal(
                region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]
            ):
                q_out = q_in[-2, -3, 0]


class DelnFlux:
    """
    Fortran name is deln_flux
    The test class is DelnFlux
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        damping_coefficients: DampingCoefficients,
        rarea,
        nord: FloatFieldK,
        damp_c: FloatFieldK,
    ):
        """
        nord sets the order of damping to apply:
        nord = 0:   del-2
        nord = 1:   del-4
        nord = 2:   del-6

        nord and damp_c define the damping coefficient used in DelnFluxNoSG
        """
        self._no_compute = False
        if (damp_c <= 1e-4).all():
            self._no_compute = True
        elif (damp_c[:-1] <= 1e-4).any():
            raise NotImplementedError(
                "damp_c currently must be always greater than 10^-4 for delnflux"
            )
        grid_indexing = stencil_factory.grid_indexing
        nk = grid_indexing.domain[2]
        self._origin = grid_indexing.origin_full()

        shape = grid_indexing.max_shape
        k_shape = (1, 1, nk)

        self._damp_3d = utils.make_storage_from_shape(
            k_shape, backend=stencil_factory.backend
        )
        # fields must be 3d to assign to them
        self._fx2 = utils.make_storage_from_shape(
            shape, backend=stencil_factory.backend
        )
        self._fy2 = utils.make_storage_from_shape(
            shape, backend=stencil_factory.backend
        )
        self._d2 = utils.make_storage_from_shape(
            grid_indexing.domain_full(), backend=stencil_factory.backend
        )

        damping_factor_calculation = stencil_factory.from_origin_domain(
            calc_damp, origin=(0, 0, 0), domain=k_shape
        )
        self._add_diffusive_stencil = stencil_factory.from_dims_halo(
            func=add_diffusive_component,
            compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM],
        )
        self._diffusive_damp_stencil = stencil_factory.from_dims_halo(
            func=diffusive_damp, compute_dims=[X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_DIM]
        )

        damping_factor_calculation(
            self._damp_3d, nord, damp_c, damping_coefficients.da_min
        )
        self._damp = utils.make_storage_data(
            self._damp_3d[0, 0, :], (nk,), (0,), backend=stencil_factory.backend
        )

        self.delnflux_nosg = DelnFluxNoSG(
            stencil_factory, damping_coefficients, rarea, nord, nk=nk
        )

    def __call__(
        self,
        q: FloatField,
        fx: FloatField,
        fy: FloatField,
        d2: Optional["FloatField"] = None,
        mass: Optional["FloatField"] = None,
    ):
        """
        Del-n damping for fluxes, where n = 2 * nord + 2
        Args:
            q: Field for which to calculate damped fluxes (in)
            fx: x-flux on A-grid (inout)
            fy: y-flux on A-grid (inout)
            d2: A damped copy of the q field (in)
            mass: Mass to weight the diffusive flux by (in)
        """
        if self._no_compute is True:
            return fx, fy

        if d2 is None:
            d2 = self._d2

        self.delnflux_nosg(q, self._fx2, self._fy2, self._damp, d2, mass)

        if mass is None:
            self._add_diffusive_stencil(fx, self._fx2, fy, self._fy2)
        else:
            # TODO: To join these stencils you need to overcompute, making the edges
            # 'wrong', but not actually used, separating now for comparison sanity.

            # diffusive_damp(fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin,
            # domain=(grid.nic + 1, grid.njc + 1, nk))
            self._diffusive_damp_stencil(fx, self._fx2, fy, self._fy2, mass, self._damp)

        return fx, fy


class DelnFluxNoSG:
    """
    This contains the mechanics of del6_vt and some of deln_flux from
    the Fortran code, since they are very similar routines. The test class
    is Del6VtFlux
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        damping_coefficients: DampingCoefficients,
        rarea,
        nord,
        nk: Optional[int] = None,
    ):
        """
        nord sets the order of damping to apply:
        nord = 0:   del-2
        nord = 1:   del-4
        nord = 2:   del-6
        """
        grid_indexing = stencil_factory.grid_indexing
        self._del6_u = damping_coefficients.del6_u
        self._del6_v = damping_coefficients.del6_v
        self._rarea = rarea
        if max(nord[:]) > 3:
            raise ValueError("nord must be less than 3")
        if not np.all(n in [0, 2, 3] for n in nord[:]):
            raise NotImplementedError("nord must have values 0, 2, or 3")
        self._nmax = int(max(nord[:]))
        i1 = grid_indexing.isc - 1 - self._nmax
        i2 = grid_indexing.iec + 1 + self._nmax
        j1 = grid_indexing.jsc - 1 - self._nmax
        j2 = grid_indexing.jec + 1 + self._nmax
        if nk is None:
            nk = grid_indexing.domain[2]
        nk = nk
        origin_d2 = (i1, j1, 0)
        domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
        f1_ny = grid_indexing.jec - grid_indexing.jsc + 1 + 2 * self._nmax
        f1_nx = grid_indexing.iec - grid_indexing.isc + 2 + 2 * self._nmax
        fx_origin = (grid_indexing.isc - self._nmax, grid_indexing.jsc - self._nmax, 0)
        self._nord = nord

        if nk <= 3:
            raise NotImplementedError("nk must be more than 3 for DelnFluxNoSG")

        preamble_ax_offsets = axis_offsets(grid_indexing, origin_d2, domain_d2)
        fx_ax_offsets = axis_offsets(grid_indexing, fx_origin, (f1_nx, f1_ny, nk))
        fy_ax_offsets = axis_offsets(
            grid_indexing, fx_origin, (f1_nx - 1, f1_ny + 1, nk)
        )

        origins_d2 = []
        domains_d2 = []
        origins_flux = []
        domains_fx = []
        domains_fy = []

        for n in range(self._nmax):
            nt = self._nmax - 1 - n
            nt_ny = grid_indexing.jec - grid_indexing.jsc + 3 + 2 * nt
            nt_nx = grid_indexing.iec - grid_indexing.isc + 3 + 2 * nt
            origins_d2.append(
                (grid_indexing.isc - nt - 1, grid_indexing.jsc - nt - 1, 0)
            )
            domains_d2.append((nt_nx, nt_ny, nk))
            origins_flux.append((grid_indexing.isc - nt, grid_indexing.jsc - nt, 0))
            domains_fx.append((nt_nx - 1, nt_ny - 2, nk))
            domains_fy.append((nt_nx - 2, nt_ny - 1, nk))

        nord_dictionary = {
            "nord0": nord[0],
            "nord1": nord[1],
            "nord2": nord[2],
            "nord3": nord[3],
        }

        self._d2_damp = stencil_factory.from_origin_domain(
            d2_damp_interval,
            externals={
                **nord_dictionary,
                **preamble_ax_offsets,
            },
            origin=origin_d2,
            domain=domain_d2,
        )

        self._copy_stencil_interval = stencil_factory.from_origin_domain(
            copy_stencil_interval,
            externals={
                **nord_dictionary,
                **preamble_ax_offsets,
            },
            origin=origin_d2,
            domain=domain_d2,
        )

        self._d2_stencil = get_stencils_with_varied_bounds(
            d2_highorder_stencil,
            origins_d2,
            domains_d2,
            stencil_factory=stencil_factory,
            externals={**nord_dictionary},
        )
        self._column_conditional_fx_calculation = get_stencils_with_varied_bounds(
            fx_calc_stencil_column,
            origins_flux,
            domains_fx,
            stencil_factory=stencil_factory,
            externals={**nord_dictionary},
        )
        self._column_conditional_fy_calculation = get_stencils_with_varied_bounds(
            fy_calc_stencil_column,
            origins_flux,
            domains_fy,
            stencil_factory=stencil_factory,
            externals={**nord_dictionary},
        )
        self._fx_calc_stencil = stencil_factory.from_origin_domain(
            fx_calc_stencil_nord,
            externals={**fx_ax_offsets, **nord_dictionary},
            origin=fx_origin,
            domain=(f1_nx, f1_ny, nk),
        )
        self._fy_calc_stencil = stencil_factory.from_origin_domain(
            fy_calc_stencil_nord,
            externals={**fy_ax_offsets, **nord_dictionary},
            origin=fx_origin,
            domain=(f1_nx - 1, f1_ny + 1, nk),
        )
        corner_origin, corner_domain = grid_indexing.get_origin_domain(
            dims=[X_DIM, Y_DIM, Z_DIM],
            halos=(grid_indexing.n_halo, grid_indexing.n_halo),
        )
        corner_domain = corner_domain[:2] + (nk,)
        corner_axis_offsets = axis_offsets(grid_indexing, corner_origin, corner_domain)

        self._corner_tmp = utils.make_storage_from_shape(
            corner_domain, origin=corner_origin
        )
        self._copy_corners_x_nord = stencil_factory.from_origin_domain(
            copy_corners_x_nord,
            externals={**corner_axis_offsets, **nord_dictionary},
            origin=corner_origin,
            domain=corner_domain,
        )
        self._copy_corners_y_nord = stencil_factory.from_origin_domain(
            copy_corners_y_nord,
            externals={**corner_axis_offsets, **nord_dictionary},
            origin=corner_origin,
            domain=corner_domain,
        )

    def __call__(self, q, fx2, fy2, damp_c, d2, mass=None):
        """
        Computes flux fields which would apply del-n damping to q,
        where n is set by nord.

        TODO: confirm what these args are.

        Args:
            q: Field for which to calculate damped fluxes (in)
            fx2: ? x-flux on A grid to be damped (out)
            fy2: ? y-flux on A grid to be damped (out)
            damp_c: damping coefficient for q (in)
            d2: ? a damped copy of the q field (in)
            mass: Mass to weight the diffusive flux by (in)
        """

        if mass is None:
            self._d2_damp(q, d2, damp_c)
        else:
            self._copy_stencil_interval(q, d2)

        self._copy_corners_x_nord(d2, d2)

        self._fx_calc_stencil(d2, self._del6_v, fx2)

        self._copy_corners_y_nord(d2, d2)

        self._fy_calc_stencil(d2, self._del6_u, fy2)

        for n in range(self._nmax):
            self._d2_stencil[n](
                fx2,
                fy2,
                self._rarea,
                d2,
            )

            self._copy_corners_x_nord(d2, d2)

            self._column_conditional_fx_calculation[n](
                d2,
                self._del6_v,
                fx2,
            )

            self._copy_corners_y_nord(d2, d2)

            self._column_conditional_fy_calculation[n](
                d2,
                self._del6_u,
                fy2,
            )
