import timeit

import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
import numpy as np
from gt4py.gtscript import __INLINED, BACKWARD, FORWARD, PARALLEL, computation, interval

from fv3gfs.physics.global_config import *


# GTC backends require a to skip DaCe merge optimisation
# TODO: check with GT4Py version > 36
if BACKEND.startswith("gtc"):
    stencil_decorator = gtscript.stencil(
        backend=BACKEND, skip_passes=("graph_merge_horizontal_executions",)
    )
else:
    stencil_decorator = gtscript.stencil(backend=BACKEND)


@stencil_decorator
def get_phi_fv3_stencil(
    gt0: FIELD_FLT, gq0: FIELD_FLT, del_gz: FIELD_FLT, phii: FIELD_FLT, phil: FIELD_FLT
):
    with computation(PARALLEL), interval(0, -1):
        del_gz = (
            del_gz[0, 0, 0] * gt0[0, 0, 0] * (1.0 + con_fvirt * max(0.0, gq0[0, 0, 0]))
        )

    with computation(BACKWARD):
        with interval(-1, None):
            phii = 0.0
        with interval(-2, -1):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]
        with interval(0, -2):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]


def get_phi_fv3(
    gt0: FIELD_FLT, gq0: FIELD_FLT, del_gz: FIELD_FLT, phii: FIELD_FLT, phil: FIELD_FLT
):
    with computation(PARALLEL), interval(0, -1):
        del_gz = (
            del_gz[0, 0, 0] * gt0[0, 0, 0] * (1.0 + con_fvirt * max(0.0, gq0[0, 0, 0]))
        )

    with computation(BACKWARD):
        with interval(-1, None):
            phii = 0.0
        with interval(-2, -1):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]
        with interval(0, -2):
            phil = 0.5 * (phii[0, 0, 1] + phii[0, 0, 1] + del_gz[0, 0, 0])
            phii = phii[0, 0, 1] + del_gz[0, 0, 0]
