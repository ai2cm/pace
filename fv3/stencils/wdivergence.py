import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
from fv3._config import namelist, grid
sd = utils.sd
backend = utils.backend
origin = (grid.is_, grid.js, 0)

@gtscript.stencil(backend=backend, rebuild=True)
def wnonhydrostatic_stencil(w: sd, delp: sd, gx: sd, gy: sd, rarea:sd):
    with computation(PARALLEL), interval(...):
        w = delp * w + (gx - gx[1, 0, 0] + gy - gy[0, 1, 0])*rarea
