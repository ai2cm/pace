from gt4py.gtscript import FORWARD, computation, horizontal, interval, region

from pace.dsl.typing import FloatField


def edge_pe(pe: FloatField, delp: FloatField, ptop: float):
    """
    This corresponds to the pe_halo routine in FV3core
    Updading the interface pressure from the pressure differences

    Args:
        pe (out): The pressure on the interfaces of the cell
        delp (in): The pressure difference between vertical grid cells
        ptop (in): The pressure level at the top of the grid
    """
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
