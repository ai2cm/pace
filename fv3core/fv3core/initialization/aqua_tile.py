import numpy as np

import fv3core.initialization.baroclinic as baroclinic
import pace.util as fv3util
from fv3core.grid import MetricTerms
from fv3core.initialization.dycore_state import DycoreState
from pace.util.constants import DRY_MASS, GRAV, RDGAS


nhalo = fv3util.N_HALO_DEFAULT


def initialize_dry_atmosphere(u: np.ndarray, v: np.ndarray, phis: np.ndarray):
    u[:, :, :] = 0.0
    v[:, :, :] = 0.0
    phis[:, :, :] = 0.0


def set_hydrostatic_equilibrium(
    slice_2d,
    ps,
    hs,
    dry_mass,
    delp,
    ak,
    bk,
    pt,
    delz,
    area,
    mountain: bool,
    hydrostatic: bool,
    hybrid_z: bool,
):
    istart = slice_2d[0].start
    iend = slice_2d[0].stop
    jstart = slice_2d[1].start
    jend = slice_2d[1].stop
    n_i = iend - istart
    n_j = jend - jstart
    n_k = len(ak)

    hs_3d = np.repeat(hs[:, :, np.newaxis], n_k, axis=2)

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

    ztop = z1 * (RDGAS * t1) * np.log(p1 / ptop)

    if mountain is True:
        raise NotImplementedError(
            "mountain is not implemented in hydrostatic equilibrium setup"
        )
    else:
        mslp = dry_mass  # 100000.
        ps = mslp
        dps = 0

    ps = ps + dps
    ps_3d = np.repeat(ps[:, :, np.newaxis], n_k, axis=2)

    gz = np.zeros((n_i, n_j, n_k))
    ph = np.zeros((n_i, n_j, n_k))

    gz[:, :, 0] = ztop
    gz[:, :, -1] = hs
    ph[:, :, 0] = ptop
    ph[:, :, -1] = ps

    # set k-level heights and pressures:
    if hybrid_z is True:
        raise NotImplementedError("hybrid-z coordinates have not been implemented")
    else:
        ph[:, :, 1:] = ak[1:] + np.multiply.outer(bk[1:] * ps[:, :])
        p_threshold = ph[:, :, 1:-1] <= p1
        gz[:, :, 1:-1][p_threshold] = ztop + (RDGAS * t1) / np.log(
            ptop / ph[:, :, 1:-1][p_threshold]
        )
        gz[:, :, 1:-1][~p_threshold] = (hs_3d[:, :, 1:-1][~p_threshold] + c0) / (
            ph[:, :, 1:-1][~p_threshold] / ps_3d[:, :, 1:-1][~p_threshold]
        ) ** (a0 * RDGAS) - c0

        if hydrostatic is False:
            delz[:, :, :-1] = (gz[:, :, 1:] - gz[:, :, :-1]) / GRAV

    geopotential_from_temperature(pt, gz, ph, delp, t1)


def geopotential_from_temperature(pt, gz, ph, delp, t1):
    pt[:, :, :-1] = (gz[:, :, 1:] - gz[:, :, :-1]) / (
        RDGAS * np.log(ph[:, :, 1:] / ph[:, :, :-1])
    )
    pt[pt < t1] = t1
    delp[:, :, :-1] = ph[:, :, 1:] - ph[:, :, :-1]


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


def derive_pressure_fields(
    delp,
    pe,
    peln,
    pk,
    pkz,
    ptop,
):

    pe[:] = baroclinic.initialize_edge_pressure(delp, ptop)
    peln[:] = baroclinic.initialize_log_pressure_interfaces(pe, ptop)
    pk[:], pkz[:] = baroclinic.initialize_kappa_pressures(pe, peln, ptop)


def init_doubly_periodic_state(
    metric_terms: MetricTerms,
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

    state.u.data[:] = 0.0
    state.v.data[:] = 0.0
    state.ua.data[:] = 0.0
    state.va.data[:] = 0.0
    state.uc.data[:] = 0.0
    state.vc.data[:] = 0.0
    state.phis.data[:] = 0.0
    set_hydrostatic_equilibrium(
        slice_2d=slice_2d,
        ps=state.ps.data[slice_2d],
        hs=state.phis.data[slice_2d_buffer],
        dry_mass=DRY_MASS,
        delp=state.delp.data[slice_3d],
        ak=metric_terms.ak.data,
        bk=metric_terms.bk.data,
        pt=state.pt.data[slice_3d_buffer],
        delz=state.delz.data[slice_3d_buffer],
        area=metric_terms.area,
        mountain=False,
        hydrostatic=False,
        hybrid_z=False,
    )

    if do_bubble is True:
        perturb_ics(
            metric_terms, ps=state.ps.data[slice_2d], pt=state.pt.data[slice_3d]
        )
    if hydrostatic is True:
        derive_pressure_fields(
            delp=state.delp.data[slice_3d],
            pe=state.pe.data[slice_3d],
            peln=state.peln.data[slice_3d],
            pk=state.pk.data[slice_3d],
            pkz=state.pkz.data[slice_3d],
            ptop=metric_terms.ptop,
        )
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
        derive_pressure_fields(
            delp=state.delp.data[slice_3d],
            pe=state.pe.data[slice_3d],
            peln=state.peln.data[slice_3d],
            pk=state.pk.data[slice_3d],
            pkz=state.pkz.data[slice_3d],
            ptop=metric_terms.ptop,
        )
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
        qgraupel=state.qgraupel.data[slice_3d],
    )

    comm.halo_update(state.phis_quantity, n_points=nhalo)

    comm.vector_halo_update(state.u_quantity, state.v_quantity, n_points=nhalo)
