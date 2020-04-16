#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import fv3._config as spec
import fv3.utils.global_constants as constants
import numpy as np
import fv3.stencils.sim1_solver as sim1_solver
import fv3.stencils.copy_stencil as cp

sd = utils.sd


# TODO: this is totally inefficient, can we use stencils?
def compute(ms, dt2, akap, cappa, ptop, hs, w3, ptc, q_con, delpc, gz, pef, ws):
    is1 = spec.grid.is_ - 1
    ie1 = spec.grid.ie + 1
    km = spec.grid.npz - 1
    islice = slice(is1, ie1 + 1)
    kslice = slice(0, km + 1)
    kslice_shift = slice(1, km + 2)
    shape1 = (spec.grid.nic + 2, km + 2)
    dm = cp.copy(delpc, (0, 0, 0))
    cp3 = cp.copy(cappa, (0, 0, 0))
    pef[islice, spec.grid.js - 1 : spec.grid.je + 2, 0] = ptop
    pem = np.zeros(shape1)
    peg = np.zeros(shape1)
    pe2 = np.zeros(shape1)
    for j in range(spec.grid.js - 1, spec.grid.je + 2):
        dm2 = np.squeeze(dm.data[islice, j, kslice])
        cp2 = np.squeeze(cp3[islice, j, kslice])
        ptr = ptc.data[islice, j, kslice]
        wsr = ws[islice, j, :]
        pem[:, 0] = ptop
        peg[:, 0] = ptop
        for k in range(1, km + 2):
            pem[:, k] = pem[:, k - 1] + dm2[:, k - 1]
            peg[:, k] = peg[:, k - 1] + dm2[:, k - 1] * (1.0 - q_con[islice, j, k - 1])
        dz2 = gz[islice, j, kslice_shift] - gz[islice, j, kslice]
        pm2 = (peg[:, kslice_shift] - peg[:, kslice]) / np.log(
            peg[:, kslice_shift] / peg[:, kslice]
        )
        gm2 = 1.0 / (1 - cp2)
        dm2 = dm2 / constants.GRAV
        w2 = np.copy(w3[islice, j, kslice])

        sim1_solver.solve(
            is1, ie1, dt2, gm2, cp2, pe2, dm2, pm2, pem, w2, dz2, ptr, wsr
        )

        pef[islice, j, kslice_shift] = pe2[:, kslice_shift] + pem[:, kslice_shift]
        gz[islice, j, km + 1] = hs[islice, j, 0]
        for k in range(km, -1, -1):
            gz[islice, j, k] = gz[islice, j, k + 1] - dz2[:, k] * constants.GRAV
