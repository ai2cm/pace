#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from fv3.stencils.pe_halo import edge_pe
import numpy as np

sd = utils.sd
# TODO merge with pe_halo? reuse partials?
@utils.stencil()
def edge_pe(pe: sd, delp: sd, pk3: sd, ptop: float, akap: float):
    with computation(FORWARD):
        with interval(0, 1):
            pe[0, 0, 0] = ptop
        with interval(1, None):
            pe[0, 0, 0] = pe[0, 0, -1] + delp[0, 0, -1]
    # with computation(PARALLEL), interval(1, None):
    #    # pk3 = exp(akap * log(pe))


def hack_np_compute_pk3(pk3, akap, pe, islice, jslice):
    grid = spec.grid
    kslice = slice(1, grid.npz + 1)
    pk3[islice, jslice, kslice] = np.exp(akap * np.log(pe[islice, jslice, kslice]))


# TODO: exp and log aren't in gt4py, if they ever are could push slicing into a stencil
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
    hack_np_compute_pk3(
        pk3, akap, pei, slice(grid.is_ - 2, grid.is_), slice(grid.js, grid.je + 1)
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
    hack_np_compute_pk3(
        pk3, akap, pei, slice(grid.ie + 1, grid.ie + 3), slice(grid.js, grid.je + 1)
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
    hack_np_compute_pk3(
        pk3, akap, pej, slice(grid.is_ - 2, grid.ie + 3), slice(grid.js - 2, grid.js)
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
    hack_np_compute_pk3(
        pk3,
        akap,
        pej,
        slice(grid.is_ - 2, grid.ie + 3),
        slice(grid.je + 1, grid.je + 3),
    )
