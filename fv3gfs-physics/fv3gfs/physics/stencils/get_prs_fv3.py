from gt4py.gtscript import PARALLEL, computation, interval

from fv3gfs.physics.global_config import FIELD_FLT
from fv3gfs.physics.global_constants import con_fvirt


def get_prs_fv3(
    phii: FIELD_FLT,
    prsi: FIELD_FLT,
    tgrs: FIELD_FLT,
    qgrs: FIELD_FLT,
    del_: FIELD_FLT,
    del_gz: FIELD_FLT,
):
    # Passing with integration, but zero padding is different from fortran for del_gz
    with computation(PARALLEL), interval(0, -1):
        del_ = prsi[0, 0, 1] - prsi[0, 0, 0]
        del_gz = (phii[0, 0, 0] - phii[0, 0, 1]) / (
            tgrs[0, 0, 0] * (1.0 + con_fvirt * max(0.0, qgrs[0, 0, 0]))
        )
