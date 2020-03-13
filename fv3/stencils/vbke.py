#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd


@gtscript.stencil(backend=utils.backend, rebuild=utils.rebuild)
def main_vb(vc: sd, uc: sd, cosa: sd, rsina: sd, vb: sd, dt5: float):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt5 * (vc[-1, 0, 0] + vc - (uc[0, -1, 0] + uc) * cosa) * rsina


@gtscript.stencil(backend=utils.backend, rebuild=utils.rebuild)
def y_edge(vt: sd, vb: sd, dt5: float):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt5 * (vt[-1, 0, 0] + vt)


@gtscript.stencil(backend=utils.backend, rebuild=utils.rebuild)
def x_edge(vt: sd, vb: sd, dt4: float):
    with computation(PARALLEL), interval(...):
        vb[0, 0, 0] = dt4 * (-vt[-2, 0, 0] + 3.0 * (vt[-1, 0, 0] + vt) - vt[1, 0, 0])


def compute(uc, vc, vt, vb, dt5, dt4):
    grid = spec.grid
    js2 = 4 if grid.south_edge else grid.js
    is2 = 4 if grid.west_edge else grid.is_
    je1 = grid.npy + 1 if grid.north_edge else grid.je + 1
    ie1 = grid.npx + 1 if grid.east_edge else grid.ie + 1
    jdiff = je1 - js2 + 1
    idiff = ie1 - is2 + 1
    if spec.namelist["grid_type"] < 3 and not grid.nested:
        domain_y = (grid.nic + 1, 1, grid.npz)
        domain_x = (1, jdiff, grid.npz)
        if grid.south_edge:
            y_edge(vt, vb, dt5=dt5, origin=grid.compute_origin(), domain=domain_y)
        main_vb(
            vc,
            uc,
            grid.cosa,
            grid.rsina,
            vb,
            dt5=dt5,
            origin=(is2, js2, 0),
            domain=(idiff, jdiff, grid.npz),
        )
        if grid.west_edge:
            x_edge(vt, vb, dt4=dt4, origin=(grid.is_, js2, 0), domain=domain_x)
        if grid.east_edge:
            x_edge(vt, vb, dt4=dt4, origin=(grid.ie + 1, js2, 0), domain=domain_x)
        if grid.north_edge:
            y_edge(vt, vb, dt5=dt5, origin=(grid.is_, grid.je + 1, 0), domain=domain_y)
    else:
        # should be a stencil like vb = dt5 * (vc[-1, 0,0] + vc)
        raise Exception("unimplemented grid_type >= 3 or nested")
