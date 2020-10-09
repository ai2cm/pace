#!/usr/bin/env python3
import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def vorticity(u: sd, dx: sd, vt: sd):
    with computation(PARALLEL), interval(...):
        vt[0, 0, 0] = u * dx


@gtstencil()
def volume_mean_relative_vorticity(ut: sd, vt: sd, rarea: sd, wk: sd):
    with computation(PARALLEL), interval(...):
        wk[0, 0, 0] = rarea * (vt - vt[0, 1, 0] - ut + ut[1, 0, 0])


def compute(u, v, ut, vt, wk):
    vorticity(
        u,
        spec.grid.dx,
        vt,
        origin=spec.grid.default_origin(),
        domain=spec.grid.domain_shape_y(),
    )
    vorticity(
        v,
        spec.grid.dy,
        ut,
        origin=spec.grid.default_origin(),
        domain=spec.grid.domain_shape_x(),
    )
    volume_mean_relative_vorticity(
        ut,
        vt,
        spec.grid.rarea,
        wk,
        origin=spec.grid.default_origin(),
        domain=spec.grid.domain_shape_standard(),
    )
