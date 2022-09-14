from pace.util.grid import MetricTerms
from pace.fv3core.initialization.dycore_state import DycoreState
import pace.util as fv3util


def init_tc_state(metric_terms: MetricTerms,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized to the
    FV3 tropical cyclone test case (test_case 55).

    This case involves a grid_transformation (done on metric terms)
    to locally increase resolution.
    """



    return "A"