import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
import timeit
from fv3gfs.physics.global_config import *

from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    interval,
)


@gtscript.stencil(backend=BACKEND)
def get_prs_fv3_stencil(
    phii: FIELD_FLT,
    prsi: FIELD_FLT,
    tgrs: FIELD_FLT,
    qgrs: FIELD_FLT,
    del_: FIELD_FLT,
    del_gz: FIELD_FLT,
):
    with computation(PARALLEL), interval(1, None):
        del_ = prsi[0, 0, 0] - prsi[0, 0, -1]
        del_gz = (phii[0, 0, -1] - phii[0, 0, 0]) / (
            tgrs[0, 0, 0] * (1.0 + con_fvirt * max(0.0, qgrs[0, 0, 0]))
        )


def get_prs_fv3(
    phii: FIELD_FLT,
    prsi: FIELD_FLT,
    tgrs: FIELD_FLT,
    qgrs: FIELD_FLT,
    del_: FIELD_FLT,
    del_gz: FIELD_FLT,
):
    with computation(PARALLEL), interval(1, None):
        del_ = prsi[0, 0, 0] - prsi[0, 0, -1]
        del_gz = (phii[0, 0, -1] - phii[0, 0, 0]) / (
            tgrs[0, 0, 0] * (1.0 + con_fvirt * max(0.0, qgrs[0, 0, 0]))
        )
