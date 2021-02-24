import numpy as np
from gt4py.gtscript import PARALLEL, computation, exp, interval, log

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import sign


sd = utils.sd


@gtstencil()
def compute_pkz_tempadjust(
    delp: sd, delz: sd, cappa: sd, heat_source: sd, delt: sd, pt: sd, pkz: sd
):
    with computation(PARALLEL), interval(...):
        pkz = exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
        pkz = (constants.RDG * delp / delz * pt) ** (cappa / (1.0 - cappa))
        dtmp = heat_source / (constants.CV_AIR * delp)
        deltmin = sign(min(delt, abs(dtmp)), dtmp)
        pt = pt + deltmin / pkz


# TODO use stencils. limited by functions exp, log and variable that depends on k
def compute(pt, pkz, heat_source, delz, delp, cappa, n_con, bdt):
    grid = spec.grid
    delt_column = np.ones(delz.shape[2]) * abs(bdt * spec.namelist.delt_max)
    delt_column[0] *= 0.1
    delt_column[1] *= 0.5
    delt = utils.make_storage_data(delt_column, delz.shape, origin=grid.full_origin())
    compute_pkz_tempadjust(
        delp,
        delz,
        cappa,
        heat_source,
        delt,
        pt,
        pkz,
        origin=grid.compute_origin(),
        domain=(grid.nic, grid.njc, n_con),
    )
