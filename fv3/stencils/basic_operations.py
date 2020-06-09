import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd


@utils.stencil()
def adjustmentfactor_stencil(adjustment: sd, q_out: sd):
    with computation(PARALLEL), interval(...):
        q_out[0, 0, 0] = q_out * adjustment


@utils.stencil()
def adjust_divide_stencil(adjustment: sd, q_out: sd):
    with computation(PARALLEL), interval(...):
        q_out[0, 0, 0] = q_out / adjustment


@utils.stencil()
def multiply_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 * in2


@utils.stencil()
def divide_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 / in2


@utils.stencil()
def addition_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = in1 + in2


@utils.stencil()
def add_term_stencil(in1: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out[0, 0, 0] = out + in1


@utils.stencil()
def add_term_two_vars(in1: sd, out1: sd, in2: sd, out2: sd):
    with computation(PARALLEL), interval(...):
        out1[0, 0, 0] = out1 + in1
        out2[0, 0, 0] = out2 + in2


@gtscript.function
def absolute_value(in_array):
    abs_value = in_array if in_array > 0 else -in_array
    return abs_value


@utils.stencil()
def floor_cap(var: sd, floor_value: float):
    with computation(PARALLEL), interval(0, None):
        var[0, 0, 0] = var if var > floor_value else floor_value
