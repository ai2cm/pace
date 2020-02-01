#!/usr/bin/env python3
from fv3.utils.gt4py_utils import sd, halo, backend, make_storage_from_shape
import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
from .base_stencil import BaseStencil
from fv3._config import grid, namelist
origin = (1, 1, 0)

# TODO: merge with vbke
@gtscript.stencil(backend=backend)
def main_ub(uc: sd, vc: sd, cosa: sd, rsina: sd, ub: sd, *, dt5: float):
    with computation(PARALLEL), interval(...):
        ub = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina


@gtscript.stencil(backend=backend)
def x_edge(ut: sd, ub: sd, *, dt5: float):
    with computation(PARALLEL), interval(...):
        ub = dt5 * (ut[0, -1, 0] + ut)


@gtscript.stencil(backend=backend)
def y_edge(ut: sd, ub: sd, *, dt4: float):
    with computation(PARALLEL), interval(...):
        ub = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])


def compute(uc, vc, ut, ub, dt5, dt4):
    js2 = max(4, grid.js)
    is2 = max(4, grid.is_)
    je1 = min(grid.npy + 2, grid.je + 1)
    ie1 = min(grid.npy + 2, grid.je + 1)
    jdiff = je1 - js2 + 1
    idiff = ie1 - is2 + 1
   
    if namelist['grid_type'] < 3:
        domain_y = (idiff, 1, grid.npz)
        domain_x = (1, grid.njc + 1, grid.npz)
        if grid.west_edge:
            x_edge(ut, ub, dt5=dt5, origin=(grid.is_, grid.js, 0), domain=domain_x)
        main_ub(uc, vc, grid.cosa, grid.rsina, ub, dt5=dt5, origin=(is2, grid.js, 0), domain=(idiff, grid.njc+1, grid.npz))
        if grid.south_edge:
            y_edge(ut, ub, dt4=dt4, origin=(is2, grid.js, 0), domain=domain_y)
        if grid.north_edge:
            y_edge(ut, ub, dt4=dt4, origin=(is2, grid.je + 1, 0), domain=domain_y)
        if grid.east_edge:
            x_edge(ut, ub, dt5=dt5, origin=(grid.ie + 1, grid.js, 0), domain=domain_x)
        
        
    else:
        # should be a stencil like ub = dt5 * (vc[-1, 0,0] + vc)
        raise Exception('untested glad grid_type not < 3')
    
