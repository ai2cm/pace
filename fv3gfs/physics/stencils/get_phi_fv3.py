import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
import timeit
from fv3gfs.physics.global_config import *

from gt4py.gtscript import (
    PARALLEL,
    __INLINED,
    FORWARD,
    BACKWARD,
    computation,
    interval,
)


@gtscript.stencil(backend=BACKEND)
def get_phi_fv3_stencil(
    gt0: FIELD_FLT, gq0: FIELD_FLT, del_gz: FIELD_FLT, phii: FIELD_FLT, phil: FIELD_FLT
):
    with computation(BACKWARD), interval(-1, None):
        phii = 0.0

    with computation(BACKWARD):
        with interval(-1, None):
            del_gz = (
                del_gz[0, 0, 0]
                * gt0[0, 0, 0]
                * (1.0 + con_fvirt * max(0.0, gq0[0, 0, 0]))
            )
            phil = 0.5 * (phii[0, 0, 0] + phii[0, 0, 0] + del_gz[0, 0, 0])
        with interval(1, -1):
            phii = phii[0, 0, 1] + del_gz[0, 0, 1]
            del_gz = (
                del_gz[0, 0, 0]
                * gt0[0, 0, 0]
                * (1.0 + con_fvirt * max(0.0, gq0[0, 0, 0]))
            )
            phil = 0.5 * (phii[0, 0, 0] + phii[0, 0, 0] + del_gz[0, 0, 0])
        with interval(0, 1):
            phii = phii[0, 0, 1] + del_gz[0, 0, 1]


def get_phi_fv3(
    gt0: FIELD_FLT, gq0: FIELD_FLT, del_gz: FIELD_FLT, phii: FIELD_FLT, phil: FIELD_FLT
):
    with computation(BACKWARD), interval(-1, None):
        phii = 0.0

    with computation(PARALLEL), interval(0, -1):
        del_gz = (
            del_gz[0, 0, 0] * gt0[0, 0, 0] * (1.0 + con_fvirt * max(0.0, gq0[0, 0, 0]))
        )

    # [TODO] For c12 baroclinic mp, phii passes without any changes.
    # But why isn't this passing?
    # with computation(BACKWARD), interval(0, -1):
    #     phii = phii[0, 0, 1] + del_gz[0, 0, 1]

    with computation(PARALLEL), interval(0, -1):
        phil = 0.5 * (phii[0, 0, 0] + phii[0, 0, 1])
