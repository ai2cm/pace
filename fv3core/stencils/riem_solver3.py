#!/usr/bin/env python3
import math

import numpy as np

import fv3core._config as spec
import fv3core.decorators as decorators
import fv3core.stencils.sim1_solver as sim1_solver
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.basic_operations import copy


sd = utils.sd


@utils.stencil()
def precompute(
    cp3: sd,
    dm: sd,
    zh: sd,
    q_con: sd,
    pem: sd,
    peln: sd,
    pk3: sd,
    peg: sd,
    pelng: sd,
    gm: sd,
    dz: sd,
    pm: sd,
    ptop: float,
    peln1: float,
    ptk: float,
    rgrav: float,
    akap: float,
):
    with computation(FORWARD):
        with interval(0, 1):
            pem = ptop
            peln = peln1
            pk3 = ptk
            peg = ptop
            pelng = peln1
        with interval(1, None):
            # TODO consolidate with riem_solver_c, same functions, math functions
            pem = pem[0, 0, -1] + dm[0, 0, -1]
            peln = log(pem)
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            pelng = log(peg)
            pk3 = exp(akap * peln)
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1.0 - cp3)
        dm = dm * rgrav
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / (pelng[0, 0, 1] - pelng)
        dz = zh[0, 0, 1] - zh


@utils.stencil()
def last_call_copy(peln_run: sd, peln: sd, pk3: sd, pk: sd, pem: sd, pe: sd):
    with computation(PARALLEL), interval(...):
        peln = peln_run
        pk = pk3
        pe = pem


@utils.stencil()
def finalize(
    zs: sd,
    dz: sd,
    zh: sd,
    peln_run: sd,
    peln: sd,
    pk3: sd,
    pk: sd,
    pem: sd,
    pe: sd,
    ppe: sd,
    pe_init: sd,
    last_call: bool,
):
    with computation(PARALLEL), interval(...):
        if __INLINED(spec.namelist.use_logp):
            pk3 = peln_run
        if __INLINED(spec.namelist.beta < -0.1):
            ppe = pe + pem
        else:
            ppe = pe
        if last_call:
            peln = peln_run
            pk = pk3
            pe = pem
        else:
            pe = pe_init
    with computation(BACKWARD):
        with interval(-1, None):
            zh = zs
        with interval(0, -1):
            zh = zh[0, 0, 1] - dz


def compute(
    last_call,
    dt,
    akap,
    cappa,
    ptop,
    zs,
    w,
    delz,
    q_con,
    delp,
    pt,
    zh,
    pe,
    ppe,
    pk3,
    pk,
    peln,
    wsd,
):
    grid = spec.grid
    rgrav = 1.0 / constants.GRAV
    km = grid.npz - 1
    peln1 = math.log(ptop)
    ptk = math.exp(akap * peln1)
    islice = slice(grid.is_, grid.ie + 1)
    kslice = slice(0, km + 1)
    kslice_shift = slice(1, km + 2)
    shape = w.shape
    domain = (grid.nic, grid.njc, km + 2)
    riemorigin = (grid.is_, grid.js, 0)
    dm = copy(delp)
    cp3 = copy(cappa)
    pe_init = copy(pe)
    pm = utils.make_storage_from_shape(shape, riemorigin)
    pem = utils.make_storage_from_shape(shape, riemorigin)
    peln_run = utils.make_storage_from_shape(shape, riemorigin)
    peg = utils.make_storage_from_shape(shape, riemorigin)
    pelng = utils.make_storage_from_shape(shape, riemorigin)
    gm = utils.make_storage_from_shape(shape, riemorigin)
    precompute(
        cp3,
        dm,
        zh,
        q_con,
        pem,
        peln_run,
        pk3,
        peg,
        pelng,
        gm,
        delz,
        pm,
        ptop,
        peln1,
        ptk,
        rgrav,
        akap,
        origin=riemorigin,
        domain=domain,
    )
    sim1_solver.solve(
        grid.is_,
        grid.ie,
        grid.js,
        grid.je,
        dt,
        gm,
        cp3,
        pe,
        dm,
        pm,
        pem,
        w,
        delz,
        pt,
        wsd,
    )

    finalize(
        zs,
        delz,
        zh,
        peln_run,
        peln,
        pk3,
        pk,
        pem,
        pe,
        ppe,
        pe_init,
        last_call,
        origin=riemorigin,
        domain=domain,
    )
