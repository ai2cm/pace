from typing import Any, Dict, Tuple

import numpy as np
from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, IntFieldIJ


@gtstencil()
def fix_tracer(
    q: FloatField,
    dp: FloatField,
    dm: FloatField,
    dm_pos: FloatField,
    zfix: IntFieldIJ,
    sum0: FloatFieldIJ,
    sum1: FloatFieldIJ,
):
    # reset fields
    with computation(FORWARD), interval(...):
        zfix = 0
        sum0 = 0.0
        sum1 = 0.0
    with computation(PARALLEL), interval(...):
        lower_fix = 0.0
        upper_fix = 0.0
    # fix_top:
    with computation(PARALLEL):
        with interval(0, 1):
            if q < 0:
                q = 0
            dm = q * dp
        with interval(1, 2):
            if q[0, 0, -1] < 0.0:
                q = (
                    q + q[0, 0, -1] * dp[0, 0, -1] / dp
                )  # move enough mass up so that the top layer isn't negative
    # fix_interior:
    with computation(FORWARD), interval(1, -1):
        # if a higher layer borrowed from this one, account for that here
        if lower_fix[0, 0, -1] != 0.0:
            q = q - (lower_fix[0, 0, -1] / dp)
        if q < 0.0:
            zfix += 1
            if q[0, 0, -1] > 0.0:
                # Borrow from the layer above
                dq = (
                    q[0, 0, -1] * dp[0, 0, -1]
                    if q[0, 0, -1] * dp[0, 0, -1] < -(q * dp)
                    else -(q * dp)
                )
                q = q + dq / dp
                upper_fix = dq
            if (q < 0.0) and (q[0, 0, 1] > 0.0):
                # borrow from the layer below
                dq = (
                    q[0, 0, 1] * dp[0, 0, 1]
                    if q[0, 0, 1] * dp[0, 0, 1] < -(q * dp)
                    else -(q * dp)
                )
                q = q + dq / dp
                lower_fix = dq
    with computation(PARALLEL), interval(0, -1):
        if upper_fix[0, 0, 1] != 0.0:
            # If a lower layer borrowed from this one, account for that here
            q = q - upper_fix[0, 0, 1] / dp
        dm = q * dp
        dm_pos = dm if dm > 0.0 else 0.0
    # fix_bottom:
    with computation(FORWARD), interval(-1, None):
        # the 2nd-to-last layer borrowed from this one, account for that here
        if lower_fix[0, 0, -1] != 0.0:
            q = q - (lower_fix[0, 0, -1] / dp)
        qup = q[0, 0, -1] * dp[0, 0, -1]
        qly = -q * dp
        dup = qup if qup < qly else qly
        if (q < 0.0) and (q[0, 0, -1] > 0.0):
            zfix += 1
            q = q + (dup / dp)
            upper_fix = dup
        dm = q * dp
        dm_pos = dm if dm > 0.0 else 0.0
    with computation(PARALLEL), interval(-2, -1):
        # if the bottom layer borrowed from this one, adjust
        if upper_fix[0, 0, 1] != 0.0:
            q = q - (upper_fix[0, 0, 1] / dp)
            dm = q * dp
            dm_pos = dm if dm > 0.0 else 0.0  # now we gotta update these too
    with computation(FORWARD), interval(1, None):
        sum0 += dm
        sum1 += dm_pos
    # final_check
    with computation(PARALLEL), interval(1, None):
        fac = sum0 / sum1 if sum0 > 0.0 else 0.0
        if zfix > 0 and fac > 0.0:
            q = fac * dm / dp if fac * dm / dp > 0.0 else 0.0


def compute(
    dp2: FloatField,
    tracers: Dict[str, Any],
    im: int,
    km: int,
    nq: int,
    jslice: Tuple[int],
):
    # Same as above, but with multiple tracer fields
    i1 = spec.grid.is_
    js = jslice.start
    jext = jslice.stop - jslice.start

    tracer_list = [tracers[q] for q in utils.tracer_variables[0:nq]]
    shape = tracer_list[0].shape
    shape_ij = shape[0:2]

    dm = utils.make_storage_from_shape(shape, origin=(0, 0, 0), cache_key="fillz_dm")
    dm_pos = utils.make_storage_from_shape(
        shape, origin=(0, 0, 0), cache_key="fillz_dm_pos"
    )
    # setting initial value of upper_fix to zero is only needed
    # for validation. The values in the compute domain are set to zero in the stencil.
    zfix = utils.make_storage_from_shape(
        shape_ij, dtype=np.int, origin=(0, 0), cache_key="fillz_zfix"
    )
    sum0 = utils.make_storage_from_shape(
        shape_ij, origin=(0, 0), cache_key="fillz_sum0"
    )
    sum1 = utils.make_storage_from_shape(
        shape_ij, origin=(0, 0), cache_key="fillz_sum1"
    )
    # TODO: Implement dev_gfs_physics ifdef when we implement compiler defs.

    for tracer in tracer_list:
        fix_tracer(
            tracer,
            dp2,
            dm,
            dm_pos,
            zfix,
            sum0,
            sum1,
            origin=(i1, js, 0),
            domain=(im, jext, km),
        )
    return tracer_list
