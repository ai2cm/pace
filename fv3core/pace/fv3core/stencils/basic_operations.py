import gt4py.cartesian.gtscript as gtscript
from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from pace.dsl.typing import FloatField, FloatFieldIJ


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
