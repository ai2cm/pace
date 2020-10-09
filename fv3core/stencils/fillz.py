import math as math

import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.remap_profile as remap_profile
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.corners import fill2_4corners, fill_4corners


sd = utils.sd


def grid():
    return spec.grid


@gtstencil()
def fix_top(q: sd, dp: sd, dm: sd):
    with computation(PARALLEL), interval(1, 2):
        if q[0, 0, -1] < 0.0:
            q = (
                q + q[0, 0, -1] * dp[0, 0, -1] / dp
            )  # move enough mass up so that the top layer isn't negative
    with computation(PARALLEL), interval(0, 1):
        if q < 0:
            q = 0
        dm = q * dp


@gtstencil()
def fix_interior(
    q: sd, dp: sd, zfix: sd, upper_fix: sd, lower_fix: sd, dm: sd, dm_pos: sd
):
    with computation(FORWARD), interval(1, -1):
        # if a higher layer borrowed from this one, account for that here
        if lower_fix[0, 0, -1] != 0.0:
            q = q - (lower_fix[0, 0, -1] / dp)
        dq = q * dp
        if q < 0.0:
            zfix = 1.0
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
    with computation(PARALLEL), interval(...):
        if upper_fix[0, 0, 1] != 0.0:
            # If a lower layer borrowed from this one, account for that here
            q = q - upper_fix[0, 0, 1] / dp
        dm = q * dp
        dm_pos = dm if dm > 0.0 else 0.0


@gtstencil()
def fix_bottom(
    q: sd, dp: sd, zfix: sd, upper_fix: sd, lower_fix: sd, dm: sd, dm_pos: sd
):
    with computation(PARALLEL), interval(1, 2):
        if (
            lower_fix[0, 0, -1] != 0.0
        ):  # the 2nd-to-last layer borrowed from this one, account for that here
            q = q - (lower_fix[0, 0, -1] / dp)
        qup = q[0, 0, -1] * dp[0, 0, -1]
        qly = -q * dp
        dup = qup if qup < qly else qly
        if (q < 0.0) and (q[0, 0, -1] > 0.0):
            zfix = 1.0
            q = q + (dup / dp)
            upper_fix = dup
        dm = q * dp
        dm_pos = dm if dm > 0.0 else 0.0
    with computation(PARALLEL), interval(0, 1):
        if (
            upper_fix[0, 0, 1] != 0.0
        ):  # if the bottom layer borrowed from this one, adjust
            q = q - (upper_fix[0, 0, 1] / dp)
            dm = q * dp
            dm_pos = dm if dm > 0.0 else 0.0  # now we gotta update these too


@gtstencil()
def final_check(q: sd, dp: sd, dm: sd, zfix: sd, fac: sd):
    with computation(PARALLEL), interval(...):
        if zfix > 0:
            if fac > 0:
                q = fac * dm / dp if fac * dm / dp > 0.0 else 0.0


def compute(dp2, tracers, im, km, nq, jslice):
    # Same as above, but with multiple tracer fields
    shape = tracers[utils.tracer_variables[0]].shape
    i1 = grid().is_
    js = jslice.start
    j_extent = jslice.stop - jslice.start
    orig = (i1, js, 0)
    zfix = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
    upper_fix = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
    lower_fix = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
    dm = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
    dm_pos = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
    fac = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
    # TODO: implement dev_gfs_physics ifdef when we implement compiler defs

    for q in utils.tracer_variables[0:nq]:
        # reset fields
        zfix[:] = utils.zeros(shape, type(zfix))
        fac[:] = utils.zeros(shape, type(fac))
        lower_fix[:] = utils.zeros(shape, type(lower_fix))
        upper_fix[:] = utils.zeros(shape, type(upper_fix))
        fix_top(tracers[q], dp2, dm, origin=orig, domain=(im, j_extent, 2))
        fix_interior(
            tracers[q],
            dp2,
            zfix,
            upper_fix,
            lower_fix,
            dm,
            dm_pos,
            origin=(i1, js, 0),
            domain=(im, j_extent, km),
        )
        fix_bottom(
            tracers[q],
            dp2,
            zfix,
            upper_fix,
            lower_fix,
            dm,
            dm_pos,
            origin=(i1, js, km - 2),
            domain=(im, j_extent, 2),
        )
        fix_cols = utils.sum(zfix[:], axis=2)
        zfix[:] = utils.repeat(fix_cols[:, :, np.newaxis], km + 1, axis=2)
        sum0 = utils.sum(dm[:, :, 1:], axis=2)
        sum1 = utils.sum(dm_pos[:, :, 1:], axis=2)
        adj_factor = utils.zeros(sum0.shape, type(fac))
        adj_factor[sum0 > 0] = sum0[sum0 > 0] / sum1[sum0 > 0]
        fac[:] = utils.repeat(adj_factor[:, :, np.newaxis], km + 1, axis=2)
        final_check(
            tracers[q],
            dp2,
            dm,
            zfix,
            fac,
            origin=(i1, js, 1),
            domain=(im, j_extent, km - 1),
        )
    return [tracers[tracer] for tracer in tracers.keys()]
