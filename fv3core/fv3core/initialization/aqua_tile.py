import math  # noqa: F401

import numpy as np

import fv3core.initialization.baroclinic as baroclinic  # noqa: F401
import fv3core.utils.global_constants as constants  # noqa: F401
import fv3gfs.util as fv3util
from fv3core.grid import MetricTerms, lon_lat_midpoint  # noqa: F401
from fv3core.initialization.dycore_state import DycoreState  # noqa: F401


def initialize_dry_atmosphere(u: np.ndarray, v: np.ndarray, phis: np.ndarray):
    u[:, :, :] = 0.0
    v[:, :, :] = 0.0
    phis[:, :, :] = 0.0


def set_hydrostatic_equilibrium():
    pass


def perturb_ics():
    pass


def init_doubly_periodic_state(
    metric_terms: MetricTerms,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.TileCommunicator,
):
    pass
