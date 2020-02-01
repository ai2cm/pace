#!/usr/bin/env python3
from fv3.utils.gt4py_utils import sd, halo, backend, make_storage_from_shape
import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
from .base_stencil import BaseStencil
from fv3._config import grid, namelist
origin = (1, 1, 0)


@gtscript.stencil(backend=backend)
def main_vb(vc: sd, uc: sd, cosa: sd, rsina: sd, vb: sd, *, dt5: float):
    with computation(PARALLEL), interval(...):
        vb = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina


@gtscript.stencil(backend=backend)
def y_edge(vt: sd, vb: sd, *, dt5: float):
    with computation(PARALLEL), interval(...):
        vb = dt5 * (vt[-1, 0, 0] + vt)


@gtscript.stencil(backend=backend)
def x_edge(vt: sd, vb: sd, *, dt4: float):
    with computation(PARALLEL), interval(...):
        vb = dt4 * (-vt[-2, 0, 0] + 3.0 * (vt[-1, 0, 0] + vt) - vt[1, 0, 0])


def compute(uc, vc, vt, vb, dt5, dt4):
    js2 = max(4, grid.js)
    is2 = max(4, grid.is_)
    je1 = min(grid.npy + 1, grid.je + 1)
    ie1 = min(grid.npx + 1, grid.je + 1)
    jdiff = je1 - js2 + 1
    idiff = ie1 - is2 + 1
    if namelist['grid_type'] < 3 and not grid.nested:
        domain_y = (grid.nic + 1, 1, grid.npz)
        domain_x = (1, jdiff, grid.npz)
        if grid.south_edge:
            y_edge(vt, vb, dt5=dt5, origin=(grid.is_, grid.js, 0), domain=domain_y)
        main_vb(vc, uc, grid.cosa, grid.rsina, vb, dt5=dt5, origin=(is2, js2, 0), domain=(idiff, jdiff, grid.npz))
        if grid.west_edge:
            x_edge(vt, vb, dt4=dt4, origin=(grid.is_, js2, 0), domain=domain_x)
        if grid.east_edge:
            x_edge(vt, vb, dt4=dt4, origin=(grid.ie + 1, js2, 0), domain=domain_x)
        if grid.north_edge:
            y_edge(vt, vb, dt5=dt5, origin=(grid.is_, grid.je + 1, 0), domain=domain_y)
        
    else:
        # should be a stencil like vb = dt5 * (vc[-1, 0,0] + vc)
        raise Exception('unimplemented grid_type >= 3 or nested')
