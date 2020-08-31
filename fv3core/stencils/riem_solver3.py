#!/usr/bin/env python3
import fv3core.utils.gt4py_utils as utils
import fv3core._config as spec
import fv3core.utils.global_constants as constants
import numpy as np
import fv3core.stencils.sim1_solver as sim1_solver
import fv3core.stencils.copy_stencil as cp
import fv3core.stencils.basic_operations as basic
import math
import fv3core.decorators as decorators

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
    ptop: float,
    peln1: float,
    ptk: float,
    rgrav: float,
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
            # peln = log(pem)
            peg = peg[0, 0, -1] + dm[0, 0, -1] * (1.0 - q_con[0, 0, -1])
            # pelng = log(peg)
            # pk3 = exp(akap * peln)
    with computation(PARALLEL), interval(0, -1):
        dz = zh[0, 0, 1] - zh
    with computation(PARALLEL), interval(...):
        gm = 1.0 / (1 - cp3)
        dm = dm * rgrav


@utils.stencil()
def compute_pm(peg: sd, pelng: sd, pm: sd):
    with computation(PARALLEL), interval(0, -1):
        pm = (peg[0, 0, 1] - peg) / (pelng[0, 0, 1] - pelng)


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
    dm = cp.copy(delp, (0, 0, 0))
    cp3 = cp.copy(cappa, (0, 0, 0))
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
        ptop,
        peln1,
        ptk,
        rgrav,
        origin=riemorigin,
        domain=domain,
    )
    # TODO put into stencil when have math functions
    tmpslice = (islice, slice(grid.js, grid.je + 1), kslice_shift)
    peln_run[tmpslice] = np.log(pem[tmpslice])
    pelng[tmpslice] = np.log(peg[tmpslice])
    pk3[tmpslice] = np.exp(akap * peln_run[tmpslice])
    compute_pm(peg, pelng, pm, origin=riemorigin, domain=domain)
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
        last_call,
        origin=riemorigin,
        domain=domain,
    )
