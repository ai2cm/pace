#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from fv3.stencils.pe_halo import edge_pe
import numpy as np

sd = utils.sd


# TODO: exp and log aren't in gt4py, if they ever are could push slicing into a stencil
def compute(pk3, delp, ptop, akap):
    grid = spec.grid
    pei = utils.make_storage_from_shape(pk3.shape, grid.default_origin())
    edge_domain_x = (1, grid.njc, grid.npz + 1)
    jslice = slice(grid.js, grid.je + 1)
    kslice = slice(1, grid.npz + 1)
    for iedge in [grid.is_ - 2, grid.is_ - 1, grid.ie + 1, grid.ie + 2]:
        edge_pe(pei, delp, ptop, origin=(iedge, grid.js, 0), domain=edge_domain_x)
        pk3[iedge, jslice, kslice] = np.exp(akap * np.log(pei[iedge, jslice, kslice]))
    pej = utils.make_storage_from_shape(pk3.shape, grid.default_origin())
    edge_domain_y = (grid.nic + 4, 1, grid.npz + 1)
    islice = slice(grid.is_ - 2, grid.ie + 3)
    for jedge in [grid.js - 2, grid.js - 1, grid.je + 1, grid.je + 2]:
        edge_pe(pej, delp, ptop, origin=(grid.is_ - 2, jedge, 0), domain=edge_domain_y)
        pk3[islice, jedge, kslice] = np.exp(akap * np.log(pej[islice, jedge, kslice]))
