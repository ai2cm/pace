import fv3.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript
import fv3._config as spec
from gt4py.gtscript import computation, interval, PARALLEL

sd = utils.sd
origin = utils.origin


@utils.stencil()
def heat_diss(
    fx2: sd,
    fy2: sd,
    w: sd,
    rarea: sd,
    heat_source: sd,
    diss_est: sd,
    dw: sd,
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        dw[0, 0, 0] = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
        heat_source[0, 0, 0] = dd8 - dw * (w + 0.5 * dw)
        diss_est[0, 0, 0] = heat_source


def compute(fx2, fy2, w, dd8, dw, heat_source, diss_est):
    heat_diss(
        fx2,
        fy2,
        w,
        spec.grid.rarea,
        heat_source,
        diss_est,
        dw,
        dd8,
        origin=origin,
        domain=spec.grid.domain_shape_compute(),
    )
