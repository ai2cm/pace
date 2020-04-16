#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.sim1_solver as sim1_solver
import fv3.stencils.copy_stencil as cp
import math

sd = utils.sd

# TODO: this is totally inefficient, can we use stencils?
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
    rgrav = 1.0 / constants.GRAV
    km = spec.grid.npz - 1
    peln1 = math.log(ptop)
    ptk = math.exp(akap * peln1)
    islice = slice(spec.grid.is_, spec.grid.ie + 1)
    kslice = slice(0, km + 1)
    kslice_shift = slice(1, km + 2)
    shape1 = (spec.grid.nic, km + 2)
    dm = cp.copy(delp, (0, 0, 0))
    cp3 = cp.copy(cappa, (0, 0, 0))
    # pk3[islice, spec.grid.js:spec.grid.je+1, 0] = ptk
    pm2 = np.zeros(shape1)
    pe2 = np.zeros(shape1)
    pem = np.zeros(shape1)
    peln2 = np.zeros(shape1)
    peg = np.zeros(shape1)
    pelng = np.zeros(shape1)
    for j in range(spec.grid.js, spec.grid.je + 1):
        dm2 = np.squeeze(dm.data[islice, j, kslice])
        cp2 = np.squeeze(cp3.data[islice, j, kslice])
        pem[:, 0] = ptop
        peln2[:, 0] = peln1
        pk3[islice, j, 0] = ptk
        peg[:, 0] = ptop
        pelng[:, 0] = peln1
        for k in range(1, km + 2):
            pem[:, k] = pem[:, k - 1] + dm2[:, k - 1]
            peln2[:, k] = np.log(pem[:, k])
            peg[:, k] = peg[:, k - 1] + dm2[:, k - 1] * (1.0 - q_con[islice, j, k - 1])
            pelng[:, k] = np.log(peg[:, k])
            pk3[islice, j, k] = np.exp(akap * peln2[:, k])

        pm2 = (peg[:, kslice_shift] - peg[:, kslice]) / (
            pelng[:, kslice_shift] - pelng[:, kslice]
        )

        gm2 = 1.0 / (1 - cp2)
        dm2 = dm2 * rgrav
        dz2 = np.squeeze(zh[islice, j, kslice_shift] - zh[islice, j, kslice])
        w2 = w[islice, j, kslice]
        pt2 = pt.data[islice, j, kslice]
        ws2 = wsd[islice, j, :]
        sim1_solver.solve(
            spec.grid.is_,
            spec.grid.ie,
            dt,
            gm2,
            cp2,
            pe2,
            dm2,
            pm2,
            pem,
            w2,
            dz2,
            pt2,
            ws2,
        )

        w[islice, j, kslice] = w2
        delz[islice, j, kslice] = dz2
        if last_call:
            peln[islice, j, :] = peln2
            pk[islice, j, :] = pk3[islice, j, :]
            pe[islice, j, :] = pem

        if spec.namelist["beta"] < -0.1:  # fp_out
            ppe[islice, j, :] = pe2 + pem
        else:
            ppe[islice, j, :] = pe2

        if spec.namelist["use_logp"]:
            pk3[islice, j, kslice_shift] = peln2[:, kslice_shift]

        zh[islice, j, km + 1] = zs[islice, j, km + 1]
        for k in range(km, -1, -1):
            zh[islice, j, k] = zh[islice, j, k + 1] - dz2[:, k]
