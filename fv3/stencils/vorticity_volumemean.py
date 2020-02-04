#!/usr/bin/env python3
from fv3.utils.gt4py_utils import sd, halo, exec_backend, make_storage_from_shape
import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
from .base_stencil import BaseStencil
from fv3._config import grid, namelist


@gtscript.stencil(backend=exec_backend)
def vorticity(u: sd, dx: sd, vt: sd):
    with computation(PARALLEL), interval(...):
        vt = u * dx


@gtscript.stencil(backend=exec_backend)
def volume_mean_relative_vorticity(ut: sd, vt: sd, rarea: sd, wk: sd):
    with computation(PARALLEL), interval(...):
        wk = rarea * (vt - vt[0, 1, 0] - ut + ut[1, 0, 0])


def compute(u, v, ut, vt, wk):
    vorticity(u, grid.dx, vt, origin=grid.default_origin(), domain=grid.domain_shape_y())
    vorticity(v, grid.dy, ut, origin=grid.default_origin(), domain=grid.domain_shape_x())
    volume_mean_relative_vorticity(ut, vt, grid.rarea, wk, origin=grid.default_origin(), domain=grid.domain_shape_standard())
