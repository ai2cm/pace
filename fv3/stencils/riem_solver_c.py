#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.sim1_solver as sim1_solver
import fv3.stencils.copy_stencil as cp

sd = utils.sd


@utils.stencil()
def precompute(
    cp3: sd,
    gz: sd,
    dm: sd,
    q_con: sd,
    pem: sd,
    peg: sd,
    dz: sd,
    gm: sd,
    pef: sd,
    ptop: float,
):
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peg = ptop
            pef = ptop
        with interval(1, None):
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            pef = ptop
    with computation(PARALLEL), interval(0, -1):
        dz = gz[0, 0, 1] - gz
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cp3)
        dm = dm / constants.GRAV


@utils.stencil()
def finalize(pe2: sd, pem: sd, hs_0: sd, dz: sd, pef: sd, gz: sd):
    with computation(PARALLEL), interval(1, None):
        pef = pe2 + pem
    with computation(BACKWARD):
        with interval(-1, None):
            gz = hs_0
        with interval(0, -1):
            gz = gz[0, 0, 1] - dz * constants.GRAV


# TODO: this is totally inefficient, can we use stencils?
def compute(ms, dt2, akap, cappa, ptop, hs, w3, ptc, q_con, delpc, gz, pef, ws):
    grid = spec.grid
    is1 = grid.is_ - 1
    ie1 = grid.ie + 1
    js1 = grid.js - 1
    je1 = grid.je + 1
    km = spec.grid.npz - 1
    islice = slice(is1, ie1 + 1)
    kslice = slice(0, km + 1)
    kslice_shift = slice(1, km + 2)
    shape = w3.shape
    domain = (spec.grid.nic + 2, grid.njc + 2, km + 2)
    riemorigin = (is1, js1, 0)
    dm = cp.copy(delpc, (0, 0, 0))
    cp3 = cp.copy(cappa, (0, 0, 0))
    w = cp.copy(w3, (0, 0, 0))

    pem = utils.make_storage_from_shape(shape, riemorigin)
    peg = utils.make_storage_from_shape(shape, riemorigin)
    pe = utils.make_storage_from_shape(shape, riemorigin)
    gm = utils.make_storage_from_shape(shape, riemorigin)
    dz = utils.make_storage_from_shape(shape, riemorigin)
    pm = utils.make_storage_from_shape(shape, riemorigin)
    precompute(
        cp3,
        gz,
        dm,
        q_con,
        pem,
        peg,
        dz,
        gm,
        pef,
        ptop,
        origin=riemorigin,
        domain=domain,
    )
    # TODO add to stencil when we have math functions
    jslice = slice(js1, je1 + 1)
    tmpslice_shift = (islice, jslice, kslice_shift)
    tmpslice = (islice, jslice, kslice)
    pm[tmpslice] = (peg[tmpslice_shift] - peg[tmpslice]) / np.log(
        peg[tmpslice_shift] / peg[tmpslice]
    )
    sim1_solver.solve(is1, ie1, js1, je1, dt2, gm, cp3, pe, dm, pm, pem, w, dz, ptc, ws)
    hs_0 = utils.make_storage_data(hs[:, :, 0].data, shape)
    finalize(pe, pem, hs_0, dz, pef, gz, origin=riemorigin, domain=domain)
