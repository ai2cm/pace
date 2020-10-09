import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil


sd = utils.sd


@gtstencil()
def flux_capacitor(
    cx: sd, cy: sd, xflux: sd, yflux: sd, crx_adv: sd, cry_adv: sd, fx: sd, fy: sd
):
    with computation(PARALLEL), interval(0, None):
        cx = cx + crx_adv
        cy = cy + cry_adv
        xflux = xflux + fx
        yflux = yflux + fy


def compute(cx, cy, xflux, yflux, crx_adv, cry_adv, fx, fy):
    # this overcomputes, could split into 2 stencils for x and y directions if this is an issue
    flux_capacitor(
        cx,
        cy,
        xflux,
        yflux,
        crx_adv,
        cry_adv,
        fx,
        fy,
        origin=spec.grid.default_origin(),
        domain=spec.grid.domain_shape_standard(),
    )
