#!/usr/bin/env python3
import numpy as np

import fv3core._config as spec
import fv3core.stencils.sim1_solver as sim1_solver
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.basic_operations import copy


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
    pm: sd,
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
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / log(peg[0, 0, 1] / peg)


@utils.stencil()
def finalize(pe2: sd, pem: sd, hs: sd, dz: sd, pef: sd, gz: sd):
    # TODO: we only want to bottom level of hd, so this could be removed once hd0 is a 2d field
    with computation(FORWARD):
        with interval(0, 1):
            hs_0 = hs
        with interval(1, None):
            hs_0 = hs_0[0, 0, -1]

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
    dm = copy(delpc)
    cp3 = copy(cappa)
    w = copy(w3)

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
        pm,
        ptop,
        origin=riemorigin,
        domain=domain,
    )
    sim1_solver.solve(is1, ie1, js1, je1, dt2, gm, cp3, pe, dm, pm, pem, w, dz, ptc, ws)
    finalize(pe, pem, hs, dz, pef, gz, origin=riemorigin, domain=domain)
