from typing import Dict

from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.fillz as fillz
import fv3core.stencils.map_single as map_single
import fv3core.stencils.remap_profile as remap_profile
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


@gtstencil
def set_components(
    tracer: FloatField,
    a4_1: FloatField,
    a4_2: FloatField,
    a4_3: FloatField,
    a4_4: FloatField,
):
    with computation(PARALLEL), interval(...):
        a4_1 = tracer
        a4_2 = 0.0
        a4_3 = 0.0
        a4_4 = 0.0


def compute(
    pe1: FloatField,
    pe2: FloatField,
    dp2: FloatField,
    tracers: Dict[str, type(FloatField)],
    nq: int,
    q_min: float,
    i1: int,
    i2: int,
    j1: int,
    j2: int,
    kord: int,
    version: str = "stencil",
):
    domain_compute = (
        spec.grid.ie - spec.grid.is_ + 1,
        spec.grid.je - spec.grid.js + 1,
        spec.grid.npz + 1,
    )

    qs = utils.make_storage_from_shape(
        pe1.shape, origin=(0, 0, 0), cache_key="mapn_tracer_qs"
    )
    (
        dp1,
        q4_1,
        q4_2,
        q4_3,
        q4_4,
        origin,
        domain,
        i_extent,
        j_extent,
    ) = map_single.setup_data(tracers[utils.tracer_variables[0]], pe1, i1, i2, j1, j2)

    # transliterated fortran 3d or 2d validate, not bit-for bit
    tracer_list = [tracers[q] for q in utils.tracer_variables[0:nq]]
    for tracer in tracer_list:
        set_components(
            tracer,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            origin=(spec.grid.is_, spec.grid.js, 0),
            domain=domain_compute,
        )

        q4_1, q4_2, q4_3, q4_4 = remap_profile.compute(
            qs,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            spec.grid.npz,
            i1,
            i2,
            j1,
            j2,
            0,
            kord,
            q_min,
        )
        map_single.do_lagrangian_contributions(
            tracer,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            i1,
            i2,
            j1,
            j2,
            kord,
            origin,
            domain,
            version,
        )
    if spec.namelist.fill:
        fillz.compute(dp2, tracers, i_extent, j_extent, spec.grid.npz, nq)
