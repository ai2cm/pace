import gt4py.gtscript as gtscript
from gt4py.gtscript import FORWARD, PARALLEL, computation, cos, interval, sin

from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.util.constants import OMEGA


def copy_defn(q_in: FloatField, q_out: FloatField):
    """Copy q_in to q_out.

    Args:
        q_in: input field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_in


def adjustmentfactor_stencil_defn(adjustment: FloatFieldIJ, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_out * adjustment


def set_value_defn(q_out: FloatField, value: float):
    with computation(PARALLEL), interval(...):
        q_out = value


def adjust_divide_stencil(adjustment: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_out / adjustment


def compute_coriolis_parameter_defn(
    f: FloatFieldIJ, lon: FloatFieldIJ, lat: FloatFieldIJ, alpha: float
):
    with computation(FORWARD), interval(0, 1):
        f = (
            2.0
            * OMEGA
            * (-1.0 * cos(lon) * cos(lat) * sin(alpha) + sin(lat) * cos(alpha))
        )


@gtscript.function
def sign(a, b):
    asignb = abs(a)
    if b > 0:
        asignb = asignb
    else:
        asignb = -asignb
    return asignb


@gtscript.function
def dim(a, b):
    diff = a - b if a - b > 0 else 0
    return diff
