import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def copy_stencil(q_in: sd, q_out: sd):
    """Copy q_in to q_out.

    Args:
        q_in: input field
        q_out: output field
    """
    with computation(PARALLEL), interval(...):
        q_out = q_in


def copy(q_in, origin=(0, 0, 0), domain=None):
    """Copy q_in inside the origin and domain, and zero outside.

    Args:
        q_in: input field
        origin: Origin of the copy and new field
        domain: Extent to copy

    Returns:
        gtscript.Field[float]: Copied field
    """
    q_out = utils.make_storage_from_shape(q_in.shape, origin, init=True)
    copy_stencil(q_in, q_out, origin=origin, domain=domain)
    return q_out


@gtstencil()
def adjustmentfactor_stencil(adjustment: sd, q_out: sd):
    with computation(PARALLEL), interval(...):
        q_out[0, 0, 0] = q_out * adjustment


@gtstencil()
def adjust_divide_stencil(adjustment: sd, q_out: sd):
    with computation(PARALLEL), interval(...):
        q_out[0, 0, 0] = q_out / adjustment


@gtstencil()
def multiply_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 * in2


@gtstencil()
def divide_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 / in2


@gtstencil()
def addition_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 + in2


@gtstencil()
def add_term_stencil(in1: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = out + in1


@gtstencil()
def add_term_two_vars(in1: sd, out1: sd, in2: sd, out2: sd):
    with computation(PARALLEL), interval(...):
        out1[0, 0, 0] = out1 + in1
        out2[0, 0, 0] = out2 + in2


@gtstencil()
def subtract_term_stencil(in1: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = out - in1


@gtstencil()
def multiply_constant(in1: sd, in2: float, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 * in2


@gtstencil()
def multiply_constant_inout(inout: sd, in_float: float):
    with computation(PARALLEL), interval(...):
        inout[0, 0, 0] = in_float * inout


@gtstencil()
def floor_cap(var: sd, floor_value: float):
    with computation(PARALLEL), interval(0, None):
        var[0, 0, 0] = var if var > floor_value else floor_value


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
