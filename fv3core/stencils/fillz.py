import typing
from typing import Any, Dict

from gt4py.gtscript import FORWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.utils.typing import FloatField, FloatFieldIJ, IntFieldIJ


@typing.no_type_check
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


class FillNegativeTracerValues:
    """
    Fix tracer values to prevent negative masses.

    Fortran name is `fillz`
    """

    def __init__(
        self,
        im: int,
        jm: int,
        km: int,
        nq: int,
    ):
        grid = spec.grid
        self._nq = nq
        self._fix_tracer_stencil = FrozenStencil(
            fix_tracer, origin=grid.compute_origin(), domain=(im, jm, km)
        )

        shape = grid.domain_shape_full(add=(1, 1, 1))
        shape_ij = shape[0:2]

        self._dm = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
        self._dm_pos = utils.make_storage_from_shape(shape, origin=(0, 0, 0))
        # Setting initial value of upper_fix to zero is only needed for validation.
        # The values in the compute domain are set to zero in the stencil.
        self._zfix = utils.make_storage_from_shape(shape_ij, dtype=int, origin=(0, 0))
        self._sum0 = utils.make_storage_from_shape(shape_ij, origin=(0, 0))
        self._sum1 = utils.make_storage_from_shape(shape_ij, origin=(0, 0))

    def __call__(
        self,
        dp2: FloatField,
        tracers: Dict[str, Any],
    ):
        tracer_list = [tracers[name] for name in utils.tracer_variables[0 : self._nq]]
        for tracer in tracer_list:
            self._fix_tracer_stencil(
                tracer,
                dp2,
                self._dm,
                self._dm_pos,
                self._zfix,
                self._sum0,
                self._sum1,
            )
        return tracer_list
