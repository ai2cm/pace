from gt4py.gtscript import FORWARD, computation, horizontal, interval, region

import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import FrozenStencil
from fv3core.utils.grid import axis_offsets
from fv3core.utils.typing import FloatField, FloatFieldIJ


# TODO merge with pe_halo? reuse partials?
# NOTE: This is different from fv3core.stencils.pe_halo.edge_pe
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


class PK3Halo:
    """
    Fortran name is pk3_halo
    """

    def __init__(self, grid):
        shape_2D = grid.domain_shape_full(add=(1, 1, 1))[0:2]
        origin = grid.full_origin()
        domain = grid.domain_shape_full(add=(0, 0, 1))
        ax_offsets = axis_offsets(grid, origin, domain)
        self._pe_tmp = utils.make_storage_from_shape(shape_2D, grid.full_origin())
        self._edge_pe_update = FrozenStencil(
            func=edge_pe_update,
            externals={
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )

    def __call__(self, pk3: FloatField, delp: FloatField, ptop: float, akap: float):
        """Update pressure (pk3) in halo region

        Args:
            pk3: Interface pressure raised to power of kappa using constant kappa
            delp: Vertical delta in pressure
            ptop: The pressure level at the top of atmosphere
            akap: Poisson constant (KAPPA)
        """
        self._edge_pe_update(self._pe_tmp, delp, pk3, ptop, akap)
