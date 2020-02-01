#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
from ..utils.corners import copy_corners
import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
backend = utils.backend

def fill_corner_2d_bgrid(q:sd):
    from __externals__ import i, j
    with computation(PARALLEL), interval(...):
        q = q[i, j, 0]
def fill_corners_2d(q, direction, grid):
    if grid == 'B':
        if dir == 'x':
            if grid.sw_corner:
               corner_stencil = gtscript.stencil(definition=fill_corner_2d_bgrid, backend=backend, externals={'i':3, 'j':})
               corner_stencil(q, origin=grid.default_origin(), domain=(grid.halo, grid.halo, grid.npz))
            
