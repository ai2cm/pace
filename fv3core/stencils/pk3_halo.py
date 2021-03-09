from gt4py.gtscript import FORWARD, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.utils.typing import FloatField, FloatFieldIJ


# TODO merge with pe_halo? reuse partials?
# NOTE: This is different from fv3core.stencils.pe_halo.edge_pe
@gtstencil()
def edge_pe_update(
    pe: FloatFieldIJ, delp: FloatField, pk3: FloatField, ptop: float, akap: float
):
    from __externals__ import local_ie, local_is, local_je, local_js

    with computation(FORWARD):
        with interval(0, 1):
            with horizontal(
                region[local_is - 2 : local_is, local_js : local_je + 1],
                region[local_ie + 1 : local_ie + 3, local_js : local_je + 1],
                region[local_is - 2 : local_ie + 3, local_js - 2 : local_js],
                region[local_is - 2 : local_ie + 3, local_je + 1 : local_je + 3],
            ):
                pe = ptop
        with interval(1, None):
            with horizontal(
                region[local_is - 2 : local_is, local_js : local_je + 1],
                region[local_ie + 1 : local_ie + 3, local_js : local_je + 1],
                region[local_is - 2 : local_ie + 3, local_js - 2 : local_js],
                region[local_is - 2 : local_ie + 3, local_je + 1 : local_je + 3],
            ):
                pe = pe + delp[0, 0, -1]
                pk3 = pe ** akap


def compute(pk3, delp, ptop, akap):
    grid = spec.grid
    pe_tmp = utils.make_storage_from_shape(pk3.shape[0:2], grid.full_origin())

    edge_pe_update(
        pe_tmp,
        delp,
        pk3,
        ptop,
        akap,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(add=(0, 0, 1)),
    )
