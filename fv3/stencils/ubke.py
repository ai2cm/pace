#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
import fv3._config as spec
sd = utils.sd


# TODO: merge with vbke?
@gtscript.stencil(backend=utils.exec_backend, rebuild=utils.rebuild)
def main_ub(uc: sd, vc: sd, cosa: sd, rsina: sd, ub: sd, *, dt5: float):
    with computation(PARALLEL), interval(...):
        ub = dt5 * (uc[0, -1, 0] + uc - (vc[-1, 0, 0] + vc) * cosa) * rsina


@gtscript.stencil(backend=utils.exec_backend, rebuild=utils.rebuild)
def x_edge(ut: sd, ub: sd, *, dt5: float):
    with computation(PARALLEL), interval(...):
        ub = dt5 * (ut[0, -1, 0] + ut)


@gtscript.stencil(backend=utils.exec_backend, rebuild=utils.rebuild)
def y_edge(ut: sd, ub: sd, *, dt4: float):
    with computation(PARALLEL), interval(...):
        ub = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])


def compute(uc, vc, ut, ub, dt5, dt4):
    grid = spec.grid
    is2 = 4 if grid.west_edge else grid.is_
    ie1 = grid.npx + 1 if grid.east_edge else grid.ie + 1
    idiff = ie1 - is2 + 1
    if spec.namelist['grid_type'] < 3 and not grid.nested:
        domain_y = (idiff, 1, grid.npz)
        domain_x = (1, grid.njc + 1, grid.npz)
        if grid.west_edge:
            x_edge(ut, ub, dt5=dt5, origin=(grid.is_, grid.js, 0), domain=domain_x)
        main_ub(uc, vc, grid.cosa, grid.rsina, ub, dt5=dt5, origin=(is2, grid.js, 0), domain=(idiff, grid.njc+1, grid.npz))
        if grid.south_edge:
            y_edge(ut, ub, dt4=dt4, origin=(is2, grid.js, 0), domain=domain_y)
        if grid.north_edge:
            y_edge(ut, ub, dt4=dt4, origin=(is2, grid.je + 1, 0), domain=domain_y)
        if grid.east_edge:
            x_edge(ut, ub, dt5=dt5, origin=(grid.ie + 1, grid.js, 0), domain=domain_x)


    else:
        # should be a stencil like ub = dt5 * (vc[-1, 0,0] + vc)
        raise Exception('untested glad grid_type not < 3')
