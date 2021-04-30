from typing import Optional

import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import FORWARD, PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import StencilWrapper, gtstencil
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ, FloatFieldK


def copy_corners_x_nord0(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(0, 1):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
        ):
            q = q[0, 5, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
        ):
            q = q[-1, 4, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
        ):
            q = q[-2, 3, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
        ):
            q = q[1, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
        ):
            q = q[0, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
        ):
            q = q[-1, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
        ):
            q = q[2, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
        ):
            q = q[1, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
        ):
            q = q[0, 1, 0]
        with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
            q = q[2, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
            q = q[1, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
            q = q[0, -1, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
            q = q[1, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
            q = q[0, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
            q = q[-1, -2, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
            q = q[0, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
            q = q[-1, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
            q = q[-2, -3, 0]


def copy_corners_x_nord1(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(1, 2):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
        ):
            q = q[0, 5, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
        ):
            q = q[-1, 4, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
        ):
            q = q[-2, 3, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
        ):
            q = q[1, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
        ):
            q = q[0, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
        ):
            q = q[-1, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
        ):
            q = q[2, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
        ):
            q = q[1, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
        ):
            q = q[0, 1, 0]
        with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
            q = q[2, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
            q = q[1, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
            q = q[0, -1, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
            q = q[1, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
            q = q[0, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
            q = q[-1, -2, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
            q = q[0, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
            q = q[-1, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
            q = q[-2, -3, 0]


def copy_corners_x_nord2(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(2, 3):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
        ):
            q = q[0, 5, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
        ):
            q = q[-1, 4, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
        ):
            q = q[-2, 3, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
        ):
            q = q[1, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
        ):
            q = q[0, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
        ):
            q = q[-1, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
        ):
            q = q[2, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
        ):
            q = q[1, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
        ):
            q = q[0, 1, 0]
        with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
            q = q[2, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
            q = q[1, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
            q = q[0, -1, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
            q = q[1, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
            q = q[0, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
            q = q[-1, -2, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
            q = q[0, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
            q = q[-1, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
            q = q[-2, -3, 0]


def copy_corners_x_nord3(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(3, None):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_end + 3, j_start - 3]
        ):
            q = q[0, 5, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_end + 3, j_start - 2]
        ):
            q = q[-1, 4, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_end + 3, j_start - 1]
        ):
            q = q[-2, 3, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_end + 2, j_start - 3]
        ):
            q = q[1, 4, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_end + 2, j_start - 2]
        ):
            q = q[0, 3, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_end + 2, j_start - 1]
        ):
            q = q[-1, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_end + 1, j_start - 3]
        ):
            q = q[2, 3, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_end + 1, j_start - 2]
        ):
            q = q[1, 2, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_end + 1, j_start - 1]
        ):
            q = q[0, 1, 0]
        with horizontal(region[i_start - 3, j_end + 1], region[i_end + 1, j_end + 3]):
            q = q[2, -3, 0]
        with horizontal(region[i_start - 2, j_end + 1], region[i_end + 1, j_end + 2]):
            q = q[1, -2, 0]
        with horizontal(region[i_start - 1, j_end + 1], region[i_end + 1, j_end + 1]):
            q = q[0, -1, 0]
        with horizontal(region[i_start - 3, j_end + 2], region[i_end + 2, j_end + 3]):
            q = q[1, -4, 0]
        with horizontal(region[i_start - 2, j_end + 2], region[i_end + 2, j_end + 2]):
            q = q[0, -3, 0]
        with horizontal(region[i_start - 1, j_end + 2], region[i_end + 2, j_end + 1]):
            q = q[-1, -2, 0]
        with horizontal(region[i_start - 3, j_end + 3], region[i_end + 3, j_end + 3]):
            q = q[0, -5, 0]
        with horizontal(region[i_start - 2, j_end + 3], region[i_end + 3, j_end + 2]):
            q = q[-1, -4, 0]
        with horizontal(region[i_start - 1, j_end + 3], region[i_end + 3, j_end + 1]):
            q = q[-2, -3, 0]


def copy_corners_y_nord0(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(0, 1):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q = q[5, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q = q[4, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
        ):
            q = q[3, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q = q[4, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q = q[3, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
        ):
            q = q[2, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q = q[3, -2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q = q[2, -1, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
        ):
            q = q[1, 0, 0]
        with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
            q = q[-3, 2, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
            q = q[-4, 1, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
            q = q[-5, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
            q = q[-2, 1, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
            q = q[-3, 0, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
            q = q[-4, -1, 0]
        with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
            q = q[-1, 0, 0]
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
            q = q[-2, -1, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
            q = q[-3, -2, 0]


def copy_corners_y_nord1(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(1, 2):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q = q[5, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q = q[4, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
        ):
            q = q[3, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q = q[4, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q = q[3, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
        ):
            q = q[2, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q = q[3, -2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q = q[2, -1, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
        ):
            q = q[1, 0, 0]
        with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
            q = q[-3, 2, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
            q = q[-4, 1, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
            q = q[-5, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
            q = q[-2, 1, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
            q = q[-3, 0, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
            q = q[-4, -1, 0]
        with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
            q = q[-1, 0, 0]
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
            q = q[-2, -1, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
            q = q[-3, -2, 0]


def copy_corners_y_nord2(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(2, 3):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q = q[5, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q = q[4, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
        ):
            q = q[3, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q = q[4, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q = q[3, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
        ):
            q = q[2, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q = q[3, -2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q = q[2, -1, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
        ):
            q = q[1, 0, 0]
        with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
            q = q[-3, 2, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
            q = q[-4, 1, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
            q = q[-5, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
            q = q[-2, 1, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
            q = q[-3, 0, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
            q = q[-4, -1, 0]
        with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
            q = q[-1, 0, 0]
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
            q = q[-2, -1, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
            q = q[-3, -2, 0]


def copy_corners_y_nord3(q: FloatField):
    from __externals__ import i_end, i_start, j_end, j_start

    with computation(PARALLEL), interval(3, None):
        with horizontal(
            region[i_start - 3, j_start - 3], region[i_start - 3, j_end + 3]
        ):
            q = q[5, 0, 0]
        with horizontal(
            region[i_start - 2, j_start - 3], region[i_start - 3, j_end + 2]
        ):
            q = q[4, 1, 0]
        with horizontal(
            region[i_start - 1, j_start - 3], region[i_start - 3, j_end + 1]
        ):
            q = q[3, 2, 0]
        with horizontal(
            region[i_start - 3, j_start - 2], region[i_start - 2, j_end + 3]
        ):
            q = q[4, -1, 0]
        with horizontal(
            region[i_start - 2, j_start - 2], region[i_start - 2, j_end + 2]
        ):
            q = q[3, 0, 0]
        with horizontal(
            region[i_start - 1, j_start - 2], region[i_start - 2, j_end + 1]
        ):
            q = q[2, 1, 0]
        with horizontal(
            region[i_start - 3, j_start - 1], region[i_start - 1, j_end + 3]
        ):
            q = q[3, -2, 0]
        with horizontal(
            region[i_start - 2, j_start - 1], region[i_start - 1, j_end + 2]
        ):
            q = q[2, -1, 0]
        with horizontal(
            region[i_start - 1, j_start - 1], region[i_start - 1, j_end + 1]
        ):
            q = q[1, 0, 0]
        with horizontal(region[i_end + 1, j_start - 3], region[i_end + 3, j_end + 1]):
            q = q[-3, 2, 0]
        with horizontal(region[i_end + 2, j_start - 3], region[i_end + 3, j_end + 2]):
            q = q[-4, 1, 0]
        with horizontal(region[i_end + 3, j_start - 3], region[i_end + 3, j_end + 3]):
            q = q[-5, 0, 0]
        with horizontal(region[i_end + 1, j_start - 2], region[i_end + 2, j_end + 1]):
            q = q[-2, 1, 0]
        with horizontal(region[i_end + 2, j_start - 2], region[i_end + 2, j_end + 2]):
            q = q[-3, 0, 0]
        with horizontal(region[i_end + 3, j_start - 2], region[i_end + 2, j_end + 3]):
            q = q[-4, -1, 0]
        with horizontal(region[i_end + 1, j_start - 1], region[i_end + 1, j_end + 1]):
            q = q[-1, 0, 0]
        with horizontal(region[i_end + 2, j_start - 1], region[i_end + 1, j_end + 2]):
            q = q[-2, -1, 0]
        with horizontal(region[i_end + 3, j_start - 1], region[i_end + 1, j_end + 3]):
            q = q[-3, -2, 0]


def calc_damp(damp4: FloatField, nord: FloatFieldK, damp_c: FloatFieldK, da_min: float):
    with computation(FORWARD), interval(...):
        damp4 = (damp_c * da_min) ** (nord + 1)


def fx_calc_stencil_region(
    q: FloatField,
    del6_v: FloatFieldIJ,
    fx: FloatField,
    nord0: float,
    nord1: float,
    nord2: float,
    nord3: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(0, 1):
        if nord0 == 0:
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(1, 2):
        if nord1 == 0:
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(2, 3):
        if nord2 == 0:
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)
    with computation(PARALLEL), interval(3, None):
        if nord3 == 0:
            with horizontal(region[local_is : local_ie + 2, local_js : local_je + 1]):
                fx = fx_calculation(q, del6_v)
        else:
            fx = fx_calculation(q, del6_v)


def fy_calc_stencil_region(
    q: FloatField,
    del6_u: FloatFieldIJ,
    fy: FloatField,
    nord0: float,
    nord1: float,
    nord2: float,
    nord3: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(0, 1):
        if nord0 == 0:
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(1, 2):
        if nord1 == 0:
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(2, 3):
        if nord2 == 0:
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)
    with computation(PARALLEL), interval(3, None):
        if nord3 == 0:
            with horizontal(region[local_is : local_ie + 1, local_js : local_je + 2]):
                fy = fy_calculation(q, del6_u)
        else:
            fy = fy_calculation(q, del6_u)


def fx_calc_stencil_column(
    q: FloatField, del6_v: FloatFieldIJ, fx: FloatField, nord: FloatFieldK
):
    with computation(PARALLEL), interval(...):
        if nord > 0:
            fx = fx_calculation_neg(q, del6_v)


def fy_calc_stencil_column(
    q: FloatField, del6_u: FloatFieldIJ, fy: FloatField, nord: FloatFieldK
):
    with computation(PARALLEL), interval(...):
        if nord > 0:
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


# WARNING: untested
@gtstencil()
def fx_firstorder_use_sg(
    q: FloatField,
    sin_sg1: FloatField,
    sin_sg3: FloatField,
    dy: FloatField,
    rdxc: FloatField,
    fx: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = (
            0.5 * (sin_sg3[-1, 0, 0] + sin_sg1) * dy * (q[-1, 0, 0] - q) * rdxc
        )


# WARNING: untested
@gtstencil()
def fy_firstorder_use_sg(
    q: FloatField,
    sin_sg2: FloatField,
    sin_sg4: FloatField,
    dx: FloatField,
    rdyc: FloatField,
    fy: FloatField,
):
    with computation(PARALLEL), interval(...):
        fy[0, 0, 0] = (
            0.5 * (sin_sg4[0, -1, 0] + sin_sg2) * dx * (q[0, -1, 0] - q) * rdyc
        )


def d2_highorder_stencil(
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    d2: FloatField,
    nord0: float,
    nord1: float,
    nord2: float,
    nord3: float,
):
    with computation(PARALLEL), interval(0, 1):
        if nord0 > 0:
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(1, 2):
        if nord1 > 0:
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(2, 3):
        if nord2 > 0:
            d2 = d2_highorder(fx, fy, rarea)
    with computation(PARALLEL), interval(3, None):
        if nord3 > 0:
            d2 = d2_highorder(fx, fy, rarea)


@gtscript.function
def d2_highorder(fx: FloatField, fy: FloatField, rarea: FloatField):
    d2 = (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea
    return d2


def d2_damp_interval(
    q: FloatField,
    d2: FloatField,
    damp: FloatFieldK,
    nord0: float,
    nord1: float,
    nord2: float,
    nord3: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(0, 1):
        if nord0 == 0:
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(1, 2):
        if nord1 == 0:
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(2, 3):
        if nord2 == 0:
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q
    with computation(PARALLEL), interval(3, None):
        if nord3 == 0:
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                d2[0, 0, 0] = damp * q
        else:
            d2[0, 0, 0] = damp * q


def copy_stencil_interval(
    q_in: FloatField,
    q_out: FloatField,
    nord0: float,
    nord1: float,
    nord2: float,
    nord3: float,
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(PARALLEL), interval(0, 1):
        if nord0 == 0:
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(1, 2):
        if nord1 == 0:
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(2, 3):
        if nord2 == 0:
            with horizontal(
                region[local_is - 1 : local_ie + 2, local_js - 1 : local_je + 2]
            ):
                q_out = q_in
        else:
            q_out = q_in
    with computation(PARALLEL), interval(3, None):
        if nord3 == 0:
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
        fx[0, 0, 0] = fx + fx2
        fy[0, 0, 0] = fy + fy2


def diffusive_damp(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mass: FloatField,
    damp: FloatFieldK,
):
    with computation(PARALLEL), interval(...):
        fx[0, 0, 0] = fx + 0.5 * damp * (mass[-1, 0, 0] + mass) * fx2
        fy[0, 0, 0] = fy + 0.5 * damp * (mass[0, -1, 0] + mass) * fy2


class DelnFlux:
    """
    Fortran name is delnflux
    """

    def __init__(self):
        grid = spec.grid
        nk = grid.npz
        self._origin = (grid.isd, grid.jsd, 0)
        self._grid = grid

        shape = grid.domain_shape_full(add=(1, 1, 1))
        k_shape = (1, 1, nk)

        self._damp_3d = utils.make_storage_from_shape(k_shape)
        # fields must be 3d to assign to them
        self._fx2 = utils.make_storage_from_shape(shape, self._origin)
        self._fy2 = utils.make_storage_from_shape(shape, self._origin)

        diffuse_origin = (grid.is_, grid.js, 0)
        extended_domain = (grid.nic + 1, grid.njc + 1, nk)

        self._damping_factor_calculation = StencilWrapper(
            calc_damp, origin=(0, 0, 0), domain=k_shape
        )
        self._add_diffusive_stencil = StencilWrapper(
            add_diffusive_component, origin=diffuse_origin, domain=extended_domain
        )
        self._diffusive_damp_stencil = StencilWrapper(
            diffusive_damp, origin=diffuse_origin, domain=extended_domain
        )
        self.delnflux_nosg = DelnFluxNoSG()

    def __call__(
        self,
        q: FloatField,
        fx: FloatField,
        fy: FloatField,
        nord: FloatFieldK,
        damp_c: FloatFieldK,
        d2: Optional["FloatField"] = None,
        mass: Optional["FloatField"] = None,
    ):
        """
        Del-n damping for fluxes, where n = 2 * nord + 2
        Args:
            q: Field for which to calculate damped fluxes (in)
            fx: x-flux on A-grid (inout)
            fy: y-flux on A-grid (inout)
            nord: Order of divergence damping (in)
            damp_c: damping coefficient (in)
            d2: A damped copy of the q field (in)
            mass: Mass to weight the diffusive flux by (in)
        """

        if d2 is None:
            d2 = utils.make_storage_from_shape(
                q.shape, self._origin, cache_key="delnflux_d2"
            )
        if (damp_c <= 1e-4).all():
            return fx, fy
        elif (damp_c[:-1] <= 1e-4).any():
            raise NotImplementedError(
                "damp_c currently must be always greater than 10^-4 for delnflux"
            )

        self._damping_factor_calculation(self._damp_3d, nord, damp_c, self._grid.da_min)
        nk = self._grid.npz
        damp = utils.make_storage_data(self._damp_3d[0, 0, :], (nk,), (0,))

        self.delnflux_nosg(q, self._fx2, self._fy2, nord, damp, d2, mass)

        if mass is None:
            self._add_diffusive_stencil(fx, self._fx2, fy, self._fy2)
        else:
            # TODO: To join these stencils you need to overcompute, making the edges
            # 'wrong', but not actually used, separating now for comparison sanity.

            # diffusive_damp(fx, fx2, fy, fy2, mass, damp, origin=diffuse_origin,
            # domain=(grid.nic + 1, grid.njc + 1, nk))
            self._diffusive_damp_stencil(fx, self._fx2, fy, self._fy2, mass, damp)

        return fx, fy


class DelnFluxNoSG:
    """
    Fortran name is delnflux
    """

    def __init__(self, nmax: int = 2, nk: Optional[int] = None):
        grid = spec.grid
        i1 = grid.is_ - 1 - nmax
        i2 = grid.ie + 1 + nmax
        j1 = grid.js - 1 - nmax
        j2 = grid.je + 1 + nmax
        if nk is None:
            nk = grid.npz + 1
        origin_d2 = (i1, j1, 0)
        domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
        f1_ny = grid.je - grid.js + 1 + 2 * nmax
        f1_nx = grid.ie - grid.is_ + 2 + 2 * nmax
        fx_origin = (grid.is_ - nmax, grid.js - nmax, 0)

        preamble_ax_offsets = axis_offsets(grid, origin_d2, domain_d2)
        full_ax_offsets = axis_offsets(
            spec.grid, (grid.isd, grid.jsd, 0), (grid.nid, grid.njd, nk)
        )
        fx_ax_offsets = axis_offsets(grid, fx_origin, (f1_nx, f1_ny, nk))
        fy_ax_offsets = axis_offsets(grid, fx_origin, (f1_nx - 1, f1_ny + 1, nk))

        self._d2_damp = StencilWrapper(
            d2_damp_interval,
            externals=preamble_ax_offsets,
        )

        self._copy_stencil_interval = StencilWrapper(
            copy_stencil_interval,
            externals=preamble_ax_offsets,
        )

        conditional_corner_copy_x_functions = [
            copy_corners_x_nord0,
            copy_corners_x_nord1,
            copy_corners_x_nord2,
            copy_corners_x_nord3,
        ]

        copy_origin = (grid.isd, grid.jsd, 0)
        copy_domain = (grid.nid, grid.njd, nk)

        self._conditional_corner_copy_x_stencils = [
            StencilWrapper(
                conditional_corner_copy_x_func,
                externals=full_ax_offsets,
                origin=(grid.isd, grid.jsd, 0),
                domain=copy_domain,
            )
            for conditional_corner_copy_x_func in conditional_corner_copy_x_functions
        ]

        conditional_corner_copy_y_functions = [
            copy_corners_y_nord0,
            copy_corners_y_nord1,
            copy_corners_y_nord2,
            copy_corners_y_nord3,
        ]

        self._conditional_corner_copy_y_stencils = [
            StencilWrapper(
                conditional_corner_copy_y_func,
                externals=full_ax_offsets,
                origin=copy_origin,
                domain=copy_domain,
            )
            for conditional_corner_copy_y_func in conditional_corner_copy_y_functions
        ]

        self._d2_stencil = StencilWrapper(d2_highorder_stencil)
        self._column_conditional_fx_calculation = StencilWrapper(fx_calc_stencil_column)
        self._column_conditional_fy_calculation = StencilWrapper(fy_calc_stencil_column)
        self._fx_calc_stencil = StencilWrapper(
            fx_calc_stencil_region, externals=fx_ax_offsets
        )
        self._fy_calc_stencil = StencilWrapper(
            fy_calc_stencil_region, externals=fy_ax_offsets
        )

    def __call__(self, q, fx2, fy2, nord, damp_c, d2, mass=None, nk=None):
        if max(nord[:]) > 3:
            raise Exception("nord must be less than 3")
        if not np.all(n in [0, 2, 3] for n in nord[:]):
            raise NotImplementedError("nord must have values 0, 2, or 3")
        nmax = int(max(nord[:]))
        grid = spec.grid
        i1 = grid.is_ - 1 - nmax
        i2 = grid.ie + 1 + nmax
        j1 = grid.js - 1 - nmax
        j2 = grid.je + 1 + nmax
        if nk is None:
            nk = grid.npz
        origin_d2 = (i1, j1, 0)
        domain_d2 = (i2 - i1 + 1, j2 - j1 + 1, nk)
        f1_ny = grid.je - grid.js + 1 + 2 * nmax
        f1_nx = grid.ie - grid.is_ + 2 + 2 * nmax
        fx_origin = (grid.is_ - nmax, grid.js - nmax, 0)

        if mass is None:
            self._d2_damp(
                q,
                d2,
                damp_c,
                nord[0],
                nord[1],
                nord[2],
                nord[3],
                origin=origin_d2,
                domain=domain_d2,
            )
        else:
            self._copy_stencil_interval(
                q,
                d2,
                nord[0],
                nord[1],
                nord[2],
                nord[3],
                origin=origin_d2,
                domain=domain_d2,
            )

        for i, conditional_corner_copy_x in enumerate(
            self._conditional_corner_copy_x_stencils
        ):
            if nord[i] > 0:
                conditional_corner_copy_x(d2)

        self._fx_calc_stencil(
            d2,
            grid.del6_v,
            fx2,
            nord[0],
            nord[1],
            nord[2],
            nord[3],
            origin=fx_origin,
            domain=(f1_nx, f1_ny, nk),
        )

        for j, conditional_corner_copy_y in enumerate(
            self._conditional_corner_copy_y_stencils
        ):
            if nord[j] > 0:
                conditional_corner_copy_y(d2)

        self._fy_calc_stencil(
            d2,
            grid.del6_u,
            fy2,
            nord[0],
            nord[1],
            nord[2],
            nord[3],
            origin=fx_origin,
            domain=(f1_nx - 1, f1_ny + 1, nk),
        )

        for n in range(nmax):
            nt = nmax - 1 - n
            nt_origin = (grid.is_ - nt - 1, grid.js - nt - 1, 0)
            nt_ny = grid.je - grid.js + 3 + 2 * nt
            nt_nx = grid.ie - grid.is_ + 3 + 2 * nt
            self._d2_stencil(
                fx2,
                fy2,
                grid.rarea,
                d2,
                nord[0],
                nord[1],
                nord[2],
                nord[3],
                origin=nt_origin,
                domain=(nt_nx, nt_ny, nk),
            )

            for i, conditional_corner_copy_x in enumerate(
                self._conditional_corner_copy_x_stencils
            ):
                if nord[i] > 0:
                    conditional_corner_copy_x(d2)

            nt_origin = (grid.is_ - nt, grid.js - nt, 0)
            self._column_conditional_fx_calculation(
                d2,
                grid.del6_v,
                fx2,
                nord,
                origin=nt_origin,
                domain=(nt_nx - 1, nt_ny - 2, nk),
            )

            for i, conditional_corner_copy_y in enumerate(
                self._conditional_corner_copy_y_stencils
            ):
                if nord[i] > 0:
                    conditional_corner_copy_y(d2)

            self._column_conditional_fy_calculation(
                d2,
                grid.del6_u,
                fy2,
                nord,
                origin=nt_origin,
                domain=(nt_nx - 2, nt_ny - 1, nk),
            )
