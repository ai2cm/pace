from gt4py.gtscript import PARALLEL, computation, exp, interval, log

import pace.util.constants as constants
from fv3core.stencils.basic_operations import sign
from pace.dsl.typing import FloatField


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
        delp: Pressure thickness of atmosphere layers (in)
        delz: Vertical thickness of atmosphere layers (in)
        cappa: Power to raise pressure to (in)
        heat_source: heat source from vorticity damping implied by
            energy conservation (in)
        pt: Air potential temperature (inout)
        pkz: Layer mean pressure raised to the power of Kappa (in)
        delta_time_factor: scaled time step (in)
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
