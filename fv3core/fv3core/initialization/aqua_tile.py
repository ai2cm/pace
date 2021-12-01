import math  # noqa: F401

import numpy as np

import fv3core.initialization.baroclinic as baroclinic
import fv3gfs.util as fv3util
from fv3core.fv3core.utils.global_constants import GRAV, PI  # noqa: F401
from fv3core.grid import MetricTerms
from fv3core.initialization.dycore_state import DycoreState


def initialize_dry_atmosphere(u: np.ndarray, v: np.ndarray, phis: np.ndarray):
    u[:, :, :] = 0.0
    v[:, :, :] = 0.0
    phis[:, :, :] = 0.0


def set_hydrostatic_equilibrium(
    ps,
    phis,
    dry_mass,
    delp,
    ak,
    bk,
    pt,
    delz,
    area,
    nhalo,
    mountain: bool,
    hydrostatic: bool,
    hybrid_z: bool,
):
    if mountain is True:
        raise NotImplementedError(
            "mountain is not implemented in hydrostatic equilibrium setup"
        )
    p1 = 25000.0
    z1 = 10000 * GRAV
    t1 = 200.0
    t0 = 300.0
    a0 = (t1 - t0) / z1
    c0 = t0 / a0

    if hybrid_z is True:
        ptop = 100.0
    else:
        ptop = ak[0]
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


def set_tracers(qvapor, qliquid, qrain, qice, qsnow, qgraupel):
    qvapor = 0.0
    qliquid = 0.0
    qrain = 0.0
    qice = 0.0
    qsnow = 0.0
    qgraupel = 0.0
    # TODO: Port qsmith from fortran here.
    qvapor = 2.0e-6


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
    nx, ny, nz = baroclinic.local_compute_size(state.delp.data.shape)
    islice, jslice, slice_3d, slice_2d = baroclinic.compute_slices(nx, ny)
    # Slices with extra buffer points in the horizontal dimension
    # to accomodate averaging over shifted calculations on the grid
    _, _, slice_3d_buffer, slice_2d_buffer = baroclinic.compute_slices(nx + 1, ny + 1)

    state.ua.data[slice_3d] = 0.0
    state.va.data[slice_3d] = 0.0
    state.uc.data[slice_3d] = 0.0
    state.vc.data[slice_3d] = 0.0
    state.phis.data[slice_3d] = 0.0
    pass
    if do_bubble is True:
        perturb_ics(
            metric_terms, ps=state.ps.data[slice_2d], pt=state.pt.data[slice_3d]
        )
    if hydrostatic is True:
        raise NotImplementedError("Hydrostatic mode has not been implemented")
        baroclinic.p_var(
            delp=state.delp.data[slice_3d],
            delz=state.delz.data[slice_3d],
            pt=state.pt.data[slice_3d],
            ps=state.ps.data[slice_2d],
            qvapor=state.qvapor.data[slice_3d],
            pe=state.pe.data[slice_3d],
            peln=state.peln.data[slice_3d],
            pkz=state.pkz.data[slice_3d],
            ptop=metric_terms.ptop,
            moist_phys=moist_phys,
            make_nh=(not hydrostatic),
        )
    else:
        state.w.data[slice_3d] = 0.0
        baroclinic.p_var(
            delp=state.delp.data[slice_3d],
            delz=state.delz.data[slice_3d],
            pt=state.pt.data[slice_3d],
            ps=state.ps.data[slice_2d],
            qvapor=state.qvapor.data[slice_3d],
            pe=state.pe.data[slice_3d],
            peln=state.peln.data[slice_3d],
            pkz=state.pkz.data[slice_3d],
            ptop=metric_terms.ptop,
            moist_phys=moist_phys,
            make_nh=(not hydrostatic),
        )
    set_tracers(
        qvapor=state.qvapor.data[slice_3d],
        qliquid=state.qliquid.data[slice_3d],
        qrain=state.qrain.data[slice_3d],
        qice=state.qice.data[slice_3d],
        qsnow=state.qsnow.data[slice_3d],
        qgraupel=state.qgraupel.data[:],
    )
