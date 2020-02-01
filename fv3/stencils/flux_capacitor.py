import fv3.utils.gt4py_utils as utils
import gt4py as gt
import gt4py.gtscript as gtscript
from fv3._config import namelist, grid

sd = utils.sd
backend = utils.backend


@gtscript.stencil(backend=backend, rebuild=True)
def flux_capacitor(cx: sd, cy: sd, xflux: sd, yflux: sd, crx_adv: sd, cry_adv: sd, fx: sd, fy: sd):
    with computation(PARALLEL), interval(0, None):
        cx = cx + crx_adv
        cy = cy + cry_adv
        xflux = xflux + fx
        yflux = yflux + fy


def compute(cx, cy, xflux, yflux, crx_adv, cry_adv, fx, fy):
    # this overcomputes, could split into 2 stencils for x and y directions if this is an issue
    flux_capacitor(cx, cy, xflux, yflux, crx_adv, cry_adv, fx, fy, origin=grid.default_origin(), domain=grid.domain_shape_standard())
