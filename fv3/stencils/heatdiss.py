import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
from fv3._config import namelist, grid
sd = utils.sd
backend = utils.backend
origin = utils.origin

@gtscript.stencil(backend=backend, rebuild=True)
def heat_diss(fx2: sd, fy2: sd, w: sd, rarea: sd, heat_source: sd, diss_est: sd, dw: sd, dd8: float):
    with computation(PARALLEL), interval(...):
        dw = (fx2-fx2[1, 0, 0]+fy2-fy2[0, 1, 0]) * rarea
        heat_source = dd8 - dw * (w + 0.5 * dw)
        diss_est = heat_source

def compute(fx2, fy2, w, dd8, dw, heat_source, diss_est):
    heat_diss(fx2, fy2, w, grid.rarea, heat_source, diss_est, dw, dd8,
              origin=origin, domain=grid.domain_shape_compute())
