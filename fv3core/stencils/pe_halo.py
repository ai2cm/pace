from gt4py.gtscript import FORWARD, computation, horizontal, interval, region

import fv3core._config as spec
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField


@gtstencil()
def edge_pe(pe: FloatField, delp: FloatField, ptop: float):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(
                region[local_is - 1, local_js : local_je + 1],
                region[local_ie + 1, local_js : local_je + 1],
                region[local_is - 1 : local_ie + 2, local_js - 1],
                region[local_is - 1 : local_ie + 2, local_je + 1],
            ):
                pe[0, 0, 0] = ptop
        with interval(1, None):
            with horizontal(
                region[local_is - 1, local_js : local_je + 1],
                region[local_ie + 1, local_js : local_je + 1],
                region[local_is - 1 : local_ie + 2, local_js - 1],
                region[local_is - 1 : local_ie + 2, local_je + 1],
            ):
                pe[0, 0, 0] = pe[0, 0, -1] + delp[0, 0, -1]


def compute(pe, delp, ptop):
    grid = spec.grid
    edge_pe(
        pe,
        delp,
        ptop,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(add=(0, 0, 1)),
    )
