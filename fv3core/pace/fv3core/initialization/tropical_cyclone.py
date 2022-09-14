from pace.util.grid import MetricTerms
from pace.fv3core.initialization.dycore_state import DycoreState
import pace.util as fv3util


def init_tc_state(metric_terms: MetricTerms,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:



    return "A"