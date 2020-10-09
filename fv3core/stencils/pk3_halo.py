#!/usr/bin/env python3
import gt4py.gtscript as gtscript

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.pe_halo import edge_pe


sd = utils.sd
# TODO merge with pe_halo? reuse partials?
@gtstencil()
def edge_pe(pe: sd, delp: sd, pk3: sd, ptop: float, akap: float):
    with computation(FORWARD):
        with interval(0, 1):
            pe[0, 0, 0] = ptop
        with interval(1, None):
            pe[0, 0, 0] = pe[0, 0, -1] + delp[0, 0, -1]
    with computation(PARALLEL), interval(1, None):
        pk3 = exp(akap * log(pe))


def compute(pk3, delp, ptop, akap):
    grid = spec.grid
    pei = utils.make_storage_from_shape(pk3.shape, grid.default_origin())
    edge_domain_x = (2, grid.njc, grid.npz + 1)
    edge_pe(
        pei,
        delp,
        pk3,
        ptop,
        akap,
        origin=(grid.is_ - 2, grid.js, 0),
        domain=edge_domain_x,
    )
    edge_pe(
        pei,
        delp,
        pk3,
        ptop,
        akap,
        origin=(grid.ie + 1, grid.js, 0),
        domain=edge_domain_x,
    )
    pej = utils.make_storage_from_shape(pk3.shape, grid.default_origin())
    edge_domain_y = (grid.nic + 4, 2, grid.npz + 1)
    edge_pe(
        pej,
        delp,
        pk3,
        ptop,
        akap,
        origin=(grid.is_ - 2, grid.js - 2, 0),
        domain=edge_domain_y,
    )

    edge_pe(
        pej,
        delp,
        pk3,
        ptop,
        akap,
        origin=(grid.is_ - 2, grid.je + 1, 0),
        domain=edge_domain_y,
    )
