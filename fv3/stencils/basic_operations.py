import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript

sd = utils.sd

@gtscript.stencil(backend=utils.exec_backend)
def adjustmentfactor_stencil(adjustment: sd, q_out: sd):
    with computation(PARALLEL), interval(...):
        q_out = q_out * adjustment

@gtscript.stencil(backend=utils.exec_backend)
def adjust_divide_stencil(adjustment: sd, q_out: sd):
    with computation(PARALLEL), interval(...):
        q_out = q_out / adjustment

@gtscript.stencil(backend=utils.exec_backend)
def multiply_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out = in1 * in2

@gtscript.stencil(backend=utils.exec_backend)
def divide_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out = in1 / in2

@gtscript.stencil(backend=utils.exec_backend)
def addition_stencil(in1: sd, in2: sd, out: sd):
    with computation(PARALLEL), interval(...):
        out = in1 + in2

@gtscript.stencil(backend=utils.exec_backend)
def add_term_stencil(in1: sd,  out: sd):
    with computation(PARALLEL), interval(...):
        out = out + in1

@gtscript.stencil(backend=utils.exec_backend)
def add_term_two_vars(in1: sd,  out1: sd, in2: sd, out2: sd):
    with computation(PARALLEL), interval(...):
        out1 = out1 + in1
        out2 = out2 + in2
