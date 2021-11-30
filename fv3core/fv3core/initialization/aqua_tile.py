import math  # noqa: F401

import numpy as np

import fv3core.initialization.baroclinic as baroclinic
import fv3gfs.util as fv3util
from fv3core.fv3core.utils.global_constants import PI  # noqa: F401
from fv3core.grid import MetricTerms
from fv3core.initialization.dycore_state import DycoreState


def initialize_dry_atmosphere(u: np.ndarray, v: np.ndarray, phis: np.ndarray):
    u[:, :, :] = 0.0
    v[:, :, :] = 0.0
    phis[:, :, :] = 0.0


def _set_hydrostatic_equilibrium():
    pass


def perturb_ics(metric_terms: MetricTerms, ps, pt):
    r0 = 100.0 * (metric_terms._dx_const ** 2 + metric_terms._dy_const) ** 0.5
    i_center = int(metric_terms._npx / 2)
    j_center = int(metric_terms._npy / 2)
    i_indices = np.arange(metric_terms._npx)
    j_indices = np.arange(metric_terms._npy)
    i_dist = (i_indices - i_center) * metric_terms._dx_const
    j_dist = (j_indices - j_center) * metric_terms._dy_const
    distances = np.add.outer(i_dist ** 2, j_dist ** 2)
    distances[distances > r0] = r0

    ps_3d = np.repeat(ps[:, :, np.newaxis], len(metric_terms.ak), axis=2)
    prf = np.multiply.outer(ps, metric_terms.bk) + metric_terms.ak
    threshold = prf > 100.0
    pt[threshold] = (
        pt[threshold]
        + 0.01 * (1.0 - (distances[threshold] / r0)) * prf[threshold] / ps_3d[threshold]
    )


def set_tracers():
    q = 0.0
    pass


def init_doubly_periodic_state(
    metric_terms: MetricTerms,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.TileCommunicator,
    do_bubble: bool = True,
):
    assert metric_terms._grid_type == 4
    state = DycoreState.init_empty(metric_terms.quantity_factory)
    state.ua.data[:] = 0.0
    state.va.data[:] = 0.0
    state.uc.data[:] = 0.0
    state.vc.data[:] = 0.0
    state.phis.data[:] = 0.0
    pass
    if do_bubble is True:
        perturb_ics(metric_terms, state.ps.data[:], state.pt.data[:])
    if hydrostatic is True:
        baroclinic.p_var()
    else:
        state.w.data[:] = 0.0
        baroclinic.p_var()
    set_tracers()
