from typing import Dict, Optional

import fv3core._config as spec
import fv3core.stencils.fillz as fillz
import fv3core.stencils.map_single as map_single
import fv3core.stencils.remap_profile as remap_profile
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import FloatField


def compute(
    pe1: FloatField,
    pe2: FloatField,
    dp2: FloatField,
    tracers: Dict[str, type(FloatField)],
    nq: int,
    q_min: float,
    i1: int,
    i2: int,
    kord: int,
    j_2d: Optional[int] = None,
    version: str = "stencil",
):
    qs = utils.make_storage_from_shape(pe1.shape, origin=(0, 0, 0))
    (
        dp1,
        q4_1,
        q4_2,
        q4_3,
        q4_4,
        origin,
        domain,
        jslice,
        i_extent,
    ) = map_single.setup_data(tracers[utils.tracer_variables[0]], pe1, i1, i2, j_2d)

    # transliterated fortran 3d or 2d validate, not bit-for bit
    tracer_list = [tracers[q] for q in utils.tracer_variables[0:nq]]
    for tracer in tracer_list:
        q4_1[:] = tracer[:]
        q4_2[:] = 0.0
        q4_3[:] = 0.0
        q4_4[:] = 0.0
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
            0,
            kord,
            jslice,
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
            kord,
            jslice,
            origin,
            domain,
            version,
        )
    if spec.namelist.fill:
        fillz.compute(dp2, tracers, i_extent, spec.grid.npz, nq, jslice)
