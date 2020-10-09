#!/usr/bin/env python3
import gt4py.gtscript as gtscript

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def edge_pe(pe: sd, delp: sd, ptop: float):
    with computation(FORWARD):
        with interval(0, 1):
            pe[0, 0, 0] = ptop
        with interval(1, None):
            pe[0, 0, 0] = pe[0, 0, -1] + delp[0, 0, -1]


def compute(pe, delp, ptop):
    grid = spec.grid
    edge_domain_x = (1, grid.njc, grid.npz + 1)
    edge_pe(pe, delp, ptop, origin=(grid.is_ - 1, grid.js, 0), domain=edge_domain_x)
    edge_pe(pe, delp, ptop, origin=(grid.ie + 1, grid.js, 0), domain=edge_domain_x)
    edge_domain_y = (grid.nic + 2, 1, grid.npz + 1)
    edge_pe(pe, delp, ptop, origin=(grid.is_ - 1, grid.js - 1, 0), domain=edge_domain_y)
    edge_pe(pe, delp, ptop, origin=(grid.is_ - 1, grid.je + 1, 0), domain=edge_domain_y)
