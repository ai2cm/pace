from typing import Dict

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.fillz import FillNegativeTracerValues
from fv3core.stencils.map_single import MapSingle
from fv3core.utils.typing import FloatField


def compute(
    pe1: FloatField,
    pe2: FloatField,
    dp2: FloatField,
    tracers: Dict[str, "FloatField"],
    nq: int,
    q_min: float,
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    kord: int,
):
    qs = utils.make_storage_from_shape(
        pe1.shape, origin=(0, 0, 0), cache_key="mapn_tracer_qs"
    )
    map_single = utils.cached_stencil_class(MapSingle)(
        kord, 0, i1, i2, j1, j2, cache_key=f"mapntracer-single-j{j2}"
    )

    for q in utils.tracer_variables[0:nq]:
        map_single(tracers[q], pe1, pe2, qs)

    if spec.namelist.fill:
        fillz = utils.cached_stencil_class(FillNegativeTracerValues)(
            cache_key="mapntracer-fillz"
        )
        fillz(dp2, tracers, map_single.i_extent, map_single.j_extent, spec.grid.npz, nq)
