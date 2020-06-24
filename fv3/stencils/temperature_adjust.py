#!/usr/bin/env python3
import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
import fv3.utils.global_constants as constants
from gt4py.gtscript import computation, interval, PARALLEL
import numpy as np
import math

sd = utils.sd


@utils.stencil()
def dtmp_stencil(heat_source: sd, delp: sd, cv_air: float):
    with computation(PARALLEL), interval(...):
        dtmp[0, 0, 0] = heat_source / (cv_air * delp)


@utils.stencil()
def x_edge(ut: sd, ub: sd, *, dt5: float):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = dt5 * (ut[0, -1, 0] + ut)


@utils.stencil()
def y_edge(ut: sd, ub: sd, *, dt4: float):
    with computation(PARALLEL), interval(...):
        ub[0, 0, 0] = dt4 * (-ut[0, -2, 0] + 3.0 * (ut[0, -1, 0] + ut) - ut[0, 1, 0])


# TODO use stencils. limited by functions exp, log and variable that depends on k
def compute(pt, pkz, heat_source, delz, delp, cappa, n_con, bdt):
    grid = spec.grid
    isl = slice(grid.is_, grid.ie + 1)
    jsl = slice(grid.js, grid.je + 1)
    ksl = slice(0, n_con)
    delt_column = np.ones(n_con) * abs(bdt * spec.namelist["delt_max"])
    delt_column[0] *= 0.1
    delt_column[1] *= 0.5
    dshape = (grid.nic, grid.njc, n_con)
    delt = utils.make_storage_data_from_1d(
        delt_column, dshape, origin=grid.default_origin()
    )
    pkz[isl, jsl, ksl] = np.exp(
        cappa[isl, jsl, ksl]
        / (1 - cappa[isl, jsl, ksl])
        * np.log(
            constants.RDG
            * delp[isl, jsl, ksl]
            / delz[isl, jsl, ksl]
            * pt[isl, jsl, ksl]
        )
    )

    dtmp = heat_source[isl, jsl, ksl] / (constants.CV_AIR * delp[isl, jsl, ksl])
    deltmin = np.minimum(delt, np.abs(dtmp)) * dtmp / np.abs(dtmp)
    pt[isl, jsl, ksl] = pt[isl, jsl, ksl] + deltmin / pkz[isl, jsl, ksl]
