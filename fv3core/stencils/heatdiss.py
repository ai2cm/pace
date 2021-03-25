from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtstencil()
def heat_diss(
    fx2: FloatField,
    fy2: FloatField,
    w: FloatField,
    rarea: FloatFieldIJ,
    heat_source: FloatField,
    diss_est: FloatField,
    dw: FloatField,
    dd8: float,
):
    with computation(PARALLEL), interval(...):
        dw[0, 0, 0] = (fx2 - fx2[1, 0, 0] + fy2 - fy2[0, 1, 0]) * rarea
        heat_source[0, 0, 0] = dd8 - dw * (w + 0.5 * dw)
        diss_est[0, 0, 0] = heat_source


def compute(
    fx2: FloatField,
    fy2: FloatField,
    w: FloatField,
    dd8: float,
    dw: FloatField,
    heat_source: FloatField,
    diss_est: FloatField,
) -> None:
    heat_diss(
        fx2,
        fy2,
        w,
        spec.grid.rarea,
        heat_source,
        diss_est,
        dw,
        dd8,
        origin=utils.origin,
        domain=spec.grid.domain_shape_compute(),
    )
