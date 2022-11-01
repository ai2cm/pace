from gt4py.gtscript import PARALLEL, computation, exp, interval, log

import pace.util.constants as constants
from pace.dsl.typing import FloatField
from pace.fv3core.stencils.basic_operations import sign


def compute_pkz_tempadjust(
    delp: FloatField,
    delz: FloatField,
    cappa: FloatField,
    heat_source: FloatField,
    pt: FloatField,
    pkz: FloatField,
    delt_time_factor: float,
):
    """
    Adjust air temperature from heating due to vorticity damping.
        Heating is limited by deltmax times the length of a timestep, with the
        highest levels limited further.
    Args:
        delp (in): Pressure thickness of atmosphere layers
        delz (in): Vertical thickness of atmosphere layers
        cappa (in): R/Cp
        heat_source (in): heat source from vorticity damping implied by
            energy conservation
        pt (inout): Air potential temperature
        pkz (out): Layer mean pressure raised to the power of Kappa
        delta_time_factor (in): scaled time step
    """
    with computation(PARALLEL), interval(...):
        pkz = exp(cappa / (1.0 - cappa) * log(constants.RDG * delp / delz * pt))
        pkz = (constants.RDG * delp / delz * pt) ** (cappa / (1.0 - cappa))
        dtmp = heat_source / (constants.CV_AIR * delp)
    with computation(PARALLEL):
        with interval(0, 1):
            deltmin = sign(min(delt_time_factor * 0.1, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
        with interval(1, 2):
            deltmin = sign(min(delt_time_factor * 0.5, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
        with interval(2, None):
            deltmin = sign(min(delt_time_factor, abs(dtmp)), dtmp)
            pt = pt + deltmin / pkz
